from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class EdgeAugmentor:
    def __init__(self, config):
        self.config = config
        aug_cfg = config.get("augmentation", {})

        self.knn_k = int(aug_cfg.get("knn_k", 20))
        self.min_similarity = float(aug_cfg.get("min_similarity", 0.5))
        self.min_shared_hashtags = int(aug_cfg.get("min_shared_hashtags", 2))
        self.min_shared_urls = int(aug_cfg.get("min_shared_urls", 1))
        self.max_users_per_token = int(aug_cfg.get("max_users_per_token", 200))

    def augment(self, users_df, tweets_df, hashtag_edges_df, url_edges_df):
        print("Building augmented user-user edges...")

        users_df = users_df.copy()
        users_df["id"] = users_df["id"].astype(str)
        users_df = users_df.drop_duplicates(subset=["id"])

        user_ids = users_df["id"].tolist()
        if len(user_ids) < 2:
            return pd.DataFrame(columns=["src", "dst", "type"])

        edges = []
        edges.extend(self._build_similarity_edges(users_df, user_ids))
        edges.extend(self._build_cooccurrence_edges(
            tweets_df, hashtag_edges_df, relation="co_hashtag", min_shared=self.min_shared_hashtags
        ))
        edges.extend(self._build_cooccurrence_edges(
            tweets_df, url_edges_df, relation="co_url", min_shared=self.min_shared_urls
        ))

        edges_df = pd.DataFrame(edges, columns=["src", "dst", "type"])
        if edges_df.empty:
            return pd.DataFrame(columns=["src", "dst", "type"])

        edges_df["src"] = edges_df["src"].astype(str)
        edges_df["dst"] = edges_df["dst"].astype(str)
        edges_df["type"] = edges_df["type"].astype(str)
        edges_df = edges_df.drop_duplicates()

        print(f"Augmented edges created: {len(edges_df)}")
        for relation, count in edges_df["type"].value_counts().to_dict().items():
            print(f"  - {relation}: {count}")

        return edges_df

    def _build_similarity_edges(self, users_df, user_ids):
        feature_df = self._profile_feature_frame(users_df)
        if feature_df.shape[0] < 2:
            return []

        x = StandardScaler().fit_transform(feature_df.values.astype(float))
        k = min(self.knn_k + 1, len(user_ids))
        if k < 2:
            return []

        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(x)
        distances, indices = nn.kneighbors(x)

        edges = []
        for row_idx, neighbors in enumerate(indices):
            src = user_ids[row_idx]
            for pos, nbr_idx in enumerate(neighbors):
                if nbr_idx == row_idx:
                    continue
                similarity = 1.0 - float(distances[row_idx][pos])
                if similarity < self.min_similarity:
                    continue
                dst = user_ids[nbr_idx]
                edges.append((src, dst, "similar_profile"))

        return edges

    def _profile_feature_frame(self, users_df):
        numeric_cols = [
            "followers_count",
            "friends_count",
            "statuses_count",
            "favourites_count",
            "average_tweets_per_day",
            "account_age_days",
        ]
        bool_cols = [
            "verified",
            "geo_enabled",
            "default_profile",
            "default_profile_image",
        ]

        frame = pd.DataFrame(index=users_df.index)

        for col in numeric_cols:
            values = pd.to_numeric(users_df.get(col, 0), errors="coerce").fillna(0.0)
            if col in {"followers_count", "friends_count", "statuses_count", "favourites_count"}:
                values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
            frame[col] = values

        for col in bool_cols:
            values = users_df.get(col, False)
            frame[col] = values.apply(self._to_bool).astype(float)

        return frame.fillna(0.0)

    def _build_cooccurrence_edges(self, tweets_df, relation_edges_df, relation, min_shared):
        if relation_edges_df.empty:
            return []

        tweet_authors = tweets_df[["id", "author_id"]].dropna(subset=["author_id"]).copy()
        tweet_authors["id"] = tweet_authors["id"].astype(str)
        tweet_authors["author_id"] = tweet_authors["author_id"].astype(str)
        tweet_to_author = dict(zip(tweet_authors["id"], tweet_authors["author_id"]))

        relation_edges_df = relation_edges_df.copy()
        if "tweet_id" not in relation_edges_df.columns:
            return []

        key_column = "hashtag" if "hashtag" in relation_edges_df.columns else "url"
        if key_column not in relation_edges_df.columns:
            return []

        item_to_users = defaultdict(set)
        for _, row in relation_edges_df.iterrows():
            tweet_id = str(row["tweet_id"])
            uid = tweet_to_author.get(tweet_id)
            if uid is None:
                continue
            item = str(row[key_column]).strip().lower()
            if not item:
                continue
            item_to_users[item].add(uid)

        pair_counts = defaultdict(int)
        for users in item_to_users.values():
            users = sorted(users)
            if len(users) < 2 or len(users) > self.max_users_per_token:
                continue
            for a, b in combinations(users, 2):
                pair_counts[(a, b)] += 1

        edges = []
        for (a, b), count in pair_counts.items():
            if count < min_shared:
                continue
            edges.append((a, b, relation))
            edges.append((b, a, relation))

        return edges

    def _to_bool(self, value):
        if pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)
