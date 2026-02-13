from pipelines.training_pipeline import training_line
from zenml.client import Client

if __name__ == "__main__":
    if Client().active_stack.experiment_tracker:
        print(Client().active_stack.experiment_tracker.get_tracking_uri())
    else:
        print("No experiment tracker found in the active stack.")
    training_line("data/email_listing.csv")
