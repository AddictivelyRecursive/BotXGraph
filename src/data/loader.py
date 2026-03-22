import os
import json
import pandas as pd
import ijson
import glob

class TwiBot22Loader:
    def __init__(self, config):
        """
        config: merged config dict
        """
        self.config = config

        base_path = config["paths"]["raw_data"]
        dataset_cfg = config["dataset"]

        self.dataset_path = os.path.join(base_path, dataset_cfg["name"])

        self.user_file = os.path.join(self.dataset_path, dataset_cfg["files"]["users"])
        self.tweets_pattern = os.path.join(
            self.dataset_path,
            dataset_cfg["files"]["tweets_pattern"]
        )
        self.edge_file = os.path.join(self.dataset_path, dataset_cfg["files"]["edges"])
        self.label_file = os.path.join(self.dataset_path, dataset_cfg["files"]["labels"])

        # debug mode (optional)
        self.debug = config.get("debug", {}).get("enabled", False)
        self.max_users = config.get("debug", {}).get("max_users", None)
        self.max_tweets = config.get("debug", {}).get("max_tweets", None)
        self.max_edges = config.get("debug", {}).get("max_edges", None)

    def load_users(self):
        print("Loading users...")
        with open(self.user_file, "r", encoding="utf-8") as f:
            users = json.load(f)

        if self.debug and self.max_users:
            users = users[:self.max_users]

        print(f"Loaded {len(users)} users")
        return users

    def load_tweets(self):
        print("Loading tweets (STREAMING mode)...")

        tweet_files = sorted(glob.glob(self.tweets_pattern))

        all_tweets = []
        count = 0

        for file in tweet_files:
            print(f"Streaming {file}...")

            with open(file, "rb") as f:
                # assumes JSON is a list of tweet objects
                parser = ijson.items(f, "item")

                for tweet in parser:
                    all_tweets.append(tweet)
                    count += 1

                    if self.debug and self.max_tweets and count >= self.max_tweets:
                        print(f"Stopped early at {count} tweets (debug mode)")
                        return all_tweets

            print(f"Finished file: {file}")

        print(f"Total tweets loaded: {len(all_tweets)}")
        return all_tweets

    def load_edges(self):
        print("Loading edges...")
        edges = pd.read_csv(self.edge_file)

        if self.debug and self.max_edges:
            edges = edges.iloc[:self.max_edges]

        print(f"Loaded {len(edges)} edges")
        return edges

    def load_labels(self):
        print("Loading labels...")
        labels = pd.read_csv(self.label_file)
        print(f"Loaded {len(labels)} labels")
        return labels

    def load_all(self):
        users = self.load_users()
        tweets = self.load_tweets()
        edges = self.load_edges()
        labels = self.load_labels()

        return {
            "users": users,
            "tweets": tweets,
            "edges": edges,
            "labels": labels,
        }