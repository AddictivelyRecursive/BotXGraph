import os
import json
import re
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

        # if self.debug and self.max_users:
        #     users = users[:self.max_users]

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

        if self.debug and self.max_edges:
            edges = pd.read_csv(self.edge_file, nrows=self.max_edges)
        else:
            edges = pd.read_csv(self.edge_file)

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


class TwitterHumanBotsLoader:
    """
    Loader for the compact Twitter Human-Bots CSV dataset.

    The source CSV has one row per account. To fit the existing heterogeneous
    KG pipeline, each account is represented as a user node and one synthetic
    tweet node using the profile description text.
    """

    def __init__(self, config):
        self.config = config

        base_path = config["paths"]["raw_data"]
        dataset_cfg = config["dataset"]

        self.dataset_path = os.path.join(base_path, dataset_cfg["name"])
        self.account_file = os.path.join(
            self.dataset_path,
            dataset_cfg["files"]["accounts"]
        )

        self.debug = config.get("debug", {}).get("enabled", False)
        self.max_users = config.get("debug", {}).get("max_users", None)

    def load_accounts(self):
        print("Loading Twitter Human-Bots accounts...")

        if self.debug and self.max_users:
            accounts = pd.read_csv(self.account_file, nrows=self.max_users)
        else:
            accounts = pd.read_csv(self.account_file)

        print(f"Loaded {len(accounts)} accounts")
        return accounts

    def _extract_entities_from_text(self, text):
        text = "" if pd.isna(text) else str(text)

        hashtags = [
            {"tag": tag.lower()}
            for tag in re.findall(r"(?<!\w)#([A-Za-z0-9_]+)", text)
        ]
        urls = [
            {"expanded_url": url}
            for url in re.findall(r"https?://[^\s]+", text)
        ]

        return {
            "hashtags": hashtags,
            "urls": urls,
        }

    def build_synthetic_tweets(self, accounts):
        print("Creating synthetic profile-description tweets...")

        description_col = "description" if "description" in accounts.columns else None
        tweets = []

        for _, row in accounts.iterrows():
            uid = str(row["id"])
            text = row[description_col] if description_col else ""

            tweets.append({
                "id": f"profile_{uid}",
                "author_id": uid,
                "text": "" if pd.isna(text) else str(text),
                "entities": self._extract_entities_from_text(text),
            })

        print(f"Created {len(tweets)} synthetic tweets")
        return tweets

    def build_labels(self, accounts):
        label_col = "account_type" if "account_type" in accounts.columns else "label"
        if label_col not in accounts.columns:
            raise ValueError("Expected account_type or label column in account CSV")

        labels = accounts[["id", label_col]].copy()
        labels.rename(columns={label_col: "label"}, inplace=True)
        labels["label"] = labels["label"].astype(str).str.lower()
        return labels

    def load_all(self):
        accounts = self.load_accounts()

        if "id" not in accounts.columns:
            raise ValueError("Expected id column in account CSV")

        users = accounts.to_dict(orient="records")
        tweets = self.build_synthetic_tweets(accounts)
        labels = self.build_labels(accounts)
        edges = pd.DataFrame(columns=["src", "dst", "type"])

        return {
            "users": users,
            "tweets": tweets,
            "edges": edges,
            "labels": labels,
        }


def get_loader(config):
    dataset_type = config.get("dataset", {}).get("type")
    dataset_name = config.get("dataset", {}).get("name", "")

    if dataset_type == "twitter_human_bots" or dataset_name == "twitter-human-bots":
        return TwitterHumanBotsLoader(config)

    return TwiBot22Loader(config)
