import pandas as pd


class TwiBot22Preprocessor:
    def __init__(self, config):
        self.config = config

    # ---------------------------
    # USERS
    # ---------------------------
    def process_users(self, users):
        print("Processing users...")

        users_df = pd.DataFrame(users)

        # ensure ID exists
        assert "id" in users_df.columns, "User ID missing!"

        users_df["id"] = users_df["id"].astype(str)

        users_df = users_df.drop_duplicates(subset=["id"])

        print(f"Users processed: {len(users_df)}")
        return users_df

    # ---------------------------
    # TWEETS
    # ---------------------------
    def process_tweets(self, tweets):
        print("Processing tweets...")

        tweets_df = pd.DataFrame(tweets)

        assert "id" in tweets_df.columns, "Tweet ID missing!"

        tweets_df["id"] = tweets_df["id"].astype(str)

        # normalize text
        if "text" in tweets_df.columns:
            tweets_df["text"] = tweets_df["text"].fillna("").astype(str)
        else:
            tweets_df["text"] = ""

        # normalize author (important!)
        if "author_id" in tweets_df.columns:
            tweets_df["author_id"] = tweets_df["author_id"].astype(str)
            tweets_df["author_id"] = tweets_df["author_id"].replace("None", None)
        else:
            tweets_df["author_id"] = None

        tweets_df = tweets_df.drop_duplicates(subset=["id"])

        print(f"Tweets processed: {len(tweets_df)}")
        return tweets_df

    # ---------------------------
    # EDGES
    # ---------------------------
    def process_edges(self, edges):
        print("Processing edges...")

        edges_df = edges.copy()
        edges_df.columns = [col.lower() for col in edges_df.columns]

        # Handle TwiBot-22 format
        if "source_id" in edges_df.columns and "target_id" in edges_df.columns:
            edges_df.rename(
                columns={
                    "source_id": "src",
                    "target_id": "dst",
                    "relation": "type"
                },
                inplace=True
            )
        else:
            # fallback for other datasets
            possible_src = ["source", "src", "from"]
            possible_dst = ["target", "dst", "to"]

            src_col = next((c for c in possible_src if c in edges_df.columns), None)
            dst_col = next((c for c in possible_dst if c in edges_df.columns), None)

            assert src_col is not None, "No source column found in edges"
            assert dst_col is not None, "No target column found in edges"

            edges_df.rename(columns={src_col: "src", dst_col: "dst"}, inplace=True)

            if "type" not in edges_df.columns:
                edges_df["type"] = "unknown"

        # normalize types
        edges_df["src"] = edges_df["src"].astype(str)
        edges_df["dst"] = edges_df["dst"].astype(str)
        edges_df["type"] = edges_df["type"].astype(str)

        edges_df = edges_df.drop_duplicates()

        print(f"Edges processed: {len(edges_df)}")
        return edges_df

    # ---------------------------
    # LABELS
    # ---------------------------
    def process_labels(self, labels):
        print("Processing labels...")

        labels_df = labels.copy()

        labels_df.columns = [col.lower() for col in labels_df.columns]

        # detect id column
        id_col = None
        for c in ["id", "user_id"]:
            if c in labels_df.columns:
                id_col = c
                break

        assert id_col is not None, "No user id column in labels"

        labels_df.rename(columns={id_col: "id"}, inplace=True)

        labels_df["id"] = labels_df["id"].astype(str)

        # label column
        if "label" not in labels_df.columns:
            raise ValueError("Label column missing")

        print(f"Labels processed: {len(labels_df)}")
        return labels_df

    # ---------------------------
    # FULL PIPELINE
    # ---------------------------
    def process_all(self, data):
        users_df = self.process_users(data["users"])
        tweets_df = self.process_tweets(data["tweets"])
        edges_df = self.process_edges(data["edges"])
        labels_df = self.process_labels(data["labels"])

        return {
            "users": users_df,
            "tweets": tweets_df,
            "edges": edges_df,
            "labels": labels_df,
        }