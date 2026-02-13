from pipelines.training_pipeline import training_line
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_line("data/email_listing.csv")