from zenml.client import Client
import sys

print("Checking active stack...")
try:
    client = Client()
    stack = client.active_stack
    print(f"Active stack: {stack.name}")
    tracker = stack.experiment_tracker
    if tracker:
        print(f"Experiment tracker found: {tracker.name}")
        try:
            uri = tracker.get_tracking_uri()
            print(f"Tracking URI: {uri}")
        except Exception as e:
            print(f"Error getting URI: {e}")
    else:
        print("No experiment tracker found in active stack.")
except Exception as e:
    print(f"Error initializing Client: {e}")
