import yaml
from src.data.loader import TwiBot22Loader
from src.data.preprocessor import TwiBot22Preprocessor

def load_config():
    with open("config/base.yaml") as f:
        base = yaml.safe_load(f)

    with open("config/twibot22.yaml") as f:
        dataset = yaml.safe_load(f)

    config = {**base, **dataset}
    return config


if __name__ == "__main__":
    config = load_config()

    loader = TwiBot22Loader(config)
    data = loader.load_all()

    print("\n===== LOADER CHECK =====")
    print("Users sample:", data["users"][0])
    print("Tweets sample:", data["tweets"][0])
    print("Edges sample:\n", data["edges"].head())
    print("Labels sample:\n", data["labels"].head())
    
    preprocessor = TwiBot22Preprocessor(config)
    processed = preprocessor.process_all(data)

    print("\n===== AFTER PREPROCESSING =====")
    print(processed["users"].head())
    print(processed["tweets"].head())
    print(processed["edges"].head())
    print(processed["labels"].head())
    
    print("\n===== SANITY CHECKS =====")

    users_df = processed["users"]
    tweets_df = processed["tweets"]
    edges_df = processed["edges"]
    labels_df = processed["labels"]

    # ---------------- USERS ----------------
    print("\n[USERS]")
    print("Columns:", users_df.columns.tolist())
    print("Count:", len(users_df))
    print("Duplicate IDs:", users_df["id"].duplicated().sum())

    # ---------------- TWEETS ----------------
    print("\n[TWEETS]")
    print("Columns:", tweets_df.columns.tolist())
    print("Count:", len(tweets_df))
    print("Missing text:", tweets_df["text"].isna().sum())
    print("Missing author_id:", tweets_df["author_id"].isna().sum())

    # ---------------- EDGES ----------------
    print("\n[EDGES]")
    print("Columns:", edges_df.columns.tolist())
    print("Count:", len(edges_df))
    print(edges_df.head())

    # ---------------- LABELS ----------------
    print("\n[LABELS]")
    print("Columns:", labels_df.columns.tolist())
    print("Count:", len(labels_df))
    print(labels_df.head())