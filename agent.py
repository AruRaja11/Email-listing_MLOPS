import os 
import pandas as pd
import numpy as np
import base64
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import pickle
import json
import string

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_emails(service, count):

    print(f'Fetching last {count} emails...')
    results = service.users().messages().list(userId='me', maxResults=count).execute()
    messages = results.get('messages', [])

    if not messages:
        return "No messages found."
    
    email_data = {'sender':[], 'subject': [], 'body': [], 'text':[]}
    for msg in messages:
        m = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()

        headers = m['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown Sender")
        
        body = m.get("snippet", "No content summary available")
        email_data['sender'].append(sender)
        email_data['subject'].append(subject)
        email_data['body'].append(body)
        email_data['text'].append(subject+ " "+ body)

    return pd.DataFrame(email_data, columns=['sender', 'subject', 'body', 'text'])


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w.isalpha() and len(w) > 1]
    return " ".join(words)


# ---------------- LOAD MODEL & TOKENIZER ----------------
def load_model_and_tokenizer():
    with open("/home/arun-raja/Documents/VSC/Learnings/Email_Listing/email_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("/home/arun-raja/Documents/VSC/Learnings/Email_Listing/tokenizer.json") as f:
        tokenizer = tokenizer_from_json(json.load(f))

    return model, tokenizer


# ---------------- PREDICT CATEGORY ----------------
def predict_category(data):
    model, tokenizer = load_model_and_tokenizer()

    categories = {
        0: "forum",
        1: "promotions",
        2: "social media",
        3: "spam",
        4: "updates",
        5: "verify code"
    }

    # ---- Batch preprocessing (FAST & CORRECT) ----
    subjects = data["subject"].apply(clean_text).tolist()
    bodies = data["body"].apply(clean_text).tolist()
    texts = data["text"].apply(clean_text).tolist()

    subject_seq = pad_sequences(
        tokenizer.texts_to_sequences(subjects),
        maxlen=10
    )

    body_seq = pad_sequences(
        tokenizer.texts_to_sequences(bodies),
        maxlen=194
    )

    text_seq = pad_sequences(
        tokenizer.texts_to_sequences(texts),
        maxlen=201
    )

    # ---- Predict ONCE ----
    predictions = model.predict([subject_seq, body_seq, text_seq])
    pred_ids = np.argmax(predictions, axis=1)

    prediction_data = pd.DataFrame({
        "category": [categories[i] for i in pred_ids]
    })

    # ---- Merge results ----
    merged_data = pd.concat(
        [data.reset_index(drop=True),
         prediction_data.reset_index(drop=True)],
        axis=1
    )

    return merged_data

def get_label_map(service):
    labels = service.users().labels().list(userId="me").execute()
    return {label["name"]: label["id"] for label in labels["labels"]}


# adding labels in mail
def find_message_id(service, sender, subject):
    query = f'from:{sender} subject:"{subject}"'
    
    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=1
    ).execute()

    messages = results.get("messages", [])
    if not messages:
        return None

    return messages[0]["id"]

CATEGORY_TO_LABEL = {
    "promotions": "AI_Promotions",
    "social media": "AI_Social",
    "updates": "AI_Updates",
    "verify code": "AI_Verify_Code",
    "forum":"AI_Forum",
    "spam": "SPAM"
}

def apply_category(service, msg_id, category, label_map):
    label_name = CATEGORY_TO_LABEL.get(category)
    if not label_name:
        return

    label_id = label_map[label_name]

    body = {"addLabelIds": [label_id]}

    if category in ["spam", "promotions", "social media"]:
        body["removeLabelIds"] = ["INBOX"]

    service.users().messages().modify(
        userId="me",
        id=msg_id,
        body=body
    ).execute()

def push_categories_to_gmail(service, df):
    label_map = get_label_map(service)

    for _, row in df.iterrows():
        msg_id = find_message_id(
            service,
            sender=row["sender"],
            subject=row["subject"]
        )

        if not msg_id:
            print("Email not found:", row["subject"])
            continue

        apply_category(
            service,
            msg_id=msg_id,
            category=row["category"],
            label_map=label_map
        )


def main(listables):
    try:
        print("🔐 Authenticating with Gmail API...")
        
        # ---- AUTH (reuse logic from get_emails) ----
        SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
        creds = None
        token_path = "token.json"
        cred_path = "/home/arun-raja/Documents/VSC/Learnings/Email_Listing/credentials.json"

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(token_path, "w") as token:
                token.write(creds.to_json())

        service = build("gmail", "v1", credentials=creds)

        # ---- FETCH EMAILS ----
        print("📩 Fetching emails...")
        raw_emails = get_emails(service, count=listables)

        if isinstance(raw_emails, str):
            print(raw_emails)
            return

        print("🧠 Predicting categories...")
        categorized_emails = predict_category(raw_emails)

        print("\n📊 Prediction Results:")
        print(categorized_emails[["sender", "subject", "category"]])

        # ---- APPLY LABELS ----
        print("\n🏷️ Applying labels to Gmail...")
        push_categories_to_gmail(service, categorized_emails)

        print("\n✅ Email categorization & labeling completed successfully!")

    except Exception as e:
        print("❌ Error occurred:", e)


if __name__ == "__main__":
    main(10)