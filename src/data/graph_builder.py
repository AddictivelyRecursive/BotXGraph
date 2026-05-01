import torch

from src.data.text_encoder import TextFeatureEncoder

try:
    from torch_geometric.data import HeteroData
except ImportError:
    class _Storage:
        def __repr__(self):
            return repr(self.__dict__)

    class HeteroData(dict):
        def __getitem__(self, key):
            if key not in self:
                self[key] = _Storage()
            return dict.__getitem__(self, key)

        @property
        def node_types(self):
            return [key for key in self.keys() if isinstance(key, str)]

        @property
        def edge_types(self):
            return [key for key in self.keys() if isinstance(key, tuple)]

        def __repr__(self):
            parts = []
            for key in self.node_types:
                num_nodes = getattr(self[key], "num_nodes", "?")
                parts.append(f"{key}={{ num_nodes={num_nodes} }}")
            for key in self.edge_types:
                shape = tuple(getattr(self[key], "edge_index", torch.empty(2, 0)).shape)
                parts.append(f"{key}={{ edge_index={shape} }}")
            return "HeteroData(" + ", ".join(parts) + ")"


class GraphBuilder:
    def __init__(self, config):
        self.config = config
        feature_cfg = config.get("features", {})
        self.text_embedding_dim = int(feature_cfg.get("text_embedding_dim", 64))
        self.text_encoder = TextFeatureEncoder(embedding_dim=self.text_embedding_dim)

    # ---------------------------
    # CREATE ID MAPS
    # ---------------------------
    def build_id_maps(self, users_df, tweets_df, hashtags_df, urls_df):
        print("Building ID maps...")

        user_map = {uid: i for i, uid in enumerate(users_df["id"])}
        tweet_map = {tid: i for i, tid in enumerate(tweets_df["id"])}
        hashtag_map = {h: i for i, h in enumerate(hashtags_df["hashtag"])}
        url_map = {u: i for i, u in enumerate(urls_df["url"])}

        return user_map, tweet_map, hashtag_map, url_map

    # ---------------------------
    # BUILD EDGE INDEX
    # ---------------------------
    def build_edge_index(self, src_list, dst_list, src_map, dst_map):
        src_idx = []
        dst_idx = []

        for s, d in zip(src_list, dst_list):
            if s in src_map and d in dst_map:
                src_idx.append(src_map[s])
                dst_idx.append(dst_map[d])

        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        return edge_index

    def add_relation(self, data, src_type, rel_type, dst_type, src_values, dst_values, src_map, dst_map):
        edge_index = self.build_edge_index(src_values, dst_values, src_map, dst_map)
        data[(src_type, rel_type, dst_type)].edge_index = edge_index
        return edge_index

    def build_user_features(self, users_df, tweets_df, hashtag_edges_df, url_edges_df, user_map):
        numeric_columns = [
            "followers_count",
            "friends_count",
            "statuses_count",
            "favourites_count",
            "average_tweets_per_day",
            "account_age_days",
        ]
        boolean_columns = [
            "verified",
            "geo_enabled",
            "default_profile",
            "default_profile_image",
        ]

        feature_names = [
            "log1p_followers_count",
            "log1p_friends_count",
            "log1p_statuses_count",
            "log1p_favourites_count",
            "average_tweets_per_day",
            "account_age_days",
            "verified",
            "geo_enabled",
            "default_profile",
            "default_profile_image",
            "hashtag_count",
            "url_count",
            "description_length",
        ]

        raw_features = torch.zeros((len(user_map), len(feature_names)), dtype=torch.float)

        tweet_authors = tweets_df[["id", "author_id"]].dropna().copy()
        tweet_to_author = dict(zip(tweet_authors["id"].astype(str), tweet_authors["author_id"].astype(str)))

        hashtag_counts = self._count_entities_by_user(hashtag_edges_df, "hashtag", tweet_to_author)
        url_counts = self._count_entities_by_user(url_edges_df, "url", tweet_to_author)
        description_lengths = self._description_length_by_user(tweets_df)

        for _, row in users_df.iterrows():
            uid = str(row["id"])
            if uid not in user_map:
                continue

            idx = user_map[uid]
            values = []

            for column in numeric_columns:
                value = self._safe_float(row.get(column, 0.0))
                if column in {
                    "followers_count",
                    "friends_count",
                    "statuses_count",
                    "favourites_count",
                }:
                    value = torch.log1p(torch.tensor(max(value, 0.0))).item()
                values.append(value)

            for column in boolean_columns:
                values.append(1.0 if self._safe_bool(row.get(column, False)) else 0.0)

            values.extend([
                float(hashtag_counts.get(uid, 0)),
                float(url_counts.get(uid, 0)),
                float(description_lengths.get(uid, 0)),
            ])

            raw_features[idx] = torch.tensor(values, dtype=torch.float)

        features, feature_stats = self._zscore_normalize(raw_features)
        return features, feature_names, feature_stats

    def build_tweet_features(self, tweets_df, tweet_map):
        texts = [""] * len(tweet_map)
        for _, row in tweets_df.iterrows():
            tid = str(row["id"])
            if tid not in tweet_map:
                continue
            texts[tweet_map[tid]] = row.get("text", "")

        embeddings, source = self.text_encoder.encode(texts)
        features = torch.tensor(embeddings, dtype=torch.float)
        feature_names = [f"tweet_text_{i}" for i in range(features.shape[1])]
        return features, feature_names, source

    def build_url_features(self, urls_df, url_map):
        feature_names = [
            "domain_length",
            "token_count",
            "subdomain_count",
            "has_www",
            "has_digits",
            "has_hyphen",
            "is_shortener_like",
            "tld_is_com",
            "tld_is_org",
            "tld_is_net",
        ]
        raw_features = torch.zeros((len(url_map), len(feature_names)), dtype=torch.float)
        shortener_tokens = {"bit", "tinyurl", "t", "goo", "owly", "buff", "short", "lnk", "cutt"}

        for _, row in urls_df.iterrows():
            domain = str(row["url"]).strip().lower()
            if domain not in url_map:
                continue

            parts = [p for p in domain.split(".") if p]
            tld = parts[-1] if parts else ""
            idx = url_map[domain]
            values = [
                float(len(domain)),
                float(len(parts)),
                float(max(0, len(parts) - 2)),
                1.0 if domain.startswith("www.") else 0.0,
                1.0 if any(ch.isdigit() for ch in domain) else 0.0,
                1.0 if "-" in domain else 0.0,
                1.0 if any(tok in domain for tok in shortener_tokens) else 0.0,
                1.0 if tld == "com" else 0.0,
                1.0 if tld == "org" else 0.0,
                1.0 if tld == "net" else 0.0,
            ]
            raw_features[idx] = torch.tensor(values, dtype=torch.float)

        features, feature_stats = self._zscore_normalize(raw_features)
        return features, feature_names, feature_stats

    def build_hashtag_features(self, hashtags_df, hashtag_map):
        feature_names = [
            "hashtag_length",
            "has_digit",
            "has_upper_like_token",
            "token_parts",
        ]
        raw_features = torch.zeros((len(hashtag_map), len(feature_names)), dtype=torch.float)

        for _, row in hashtags_df.iterrows():
            hashtag = str(row["hashtag"]).strip()
            if hashtag not in hashtag_map:
                continue

            token_parts = [part for part in hashtag.replace("_", " ").split() if part]
            values = [
                float(len(hashtag)),
                1.0 if any(ch.isdigit() for ch in hashtag) else 0.0,
                1.0 if any(ch.isalpha() and ch.isupper() for ch in hashtag) else 0.0,
                float(len(token_parts) if token_parts else 1),
            ]
            raw_features[hashtag_map[hashtag]] = torch.tensor(values, dtype=torch.float)

        features, feature_stats = self._zscore_normalize(raw_features)
        return features, feature_names, feature_stats

    def _zscore_normalize(self, features):
        if features.numel() == 0:
            return features, {"mean": [], "std": []}

        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        normalized = (features - mean) / std
        stats = {
            "mean": [float(x) for x in mean.tolist()],
            "std": [float(x) for x in std.tolist()],
        }
        return normalized, stats

    def _count_entities_by_user(self, entity_edges_df, entity_column, tweet_to_author):
        counts = {}

        if entity_edges_df.empty or entity_column not in entity_edges_df.columns:
            return counts

        for _, row in entity_edges_df.iterrows():
            uid = tweet_to_author.get(str(row["tweet_id"]))
            if uid is None:
                continue
            counts[uid] = counts.get(uid, 0) + 1

        return counts

    def _description_length_by_user(self, tweets_df):
        lengths = {}

        if "author_id" not in tweets_df.columns or "text" not in tweets_df.columns:
            return lengths

        for _, row in tweets_df.dropna(subset=["author_id"]).iterrows():
            uid = str(row["author_id"])
            text = "" if row.get("text") is None else str(row.get("text"))
            lengths[uid] = lengths.get(uid, 0) + len(text)

        return lengths

    def _safe_float(self, value):
        try:
            if value is None:
                return 0.0
            value = float(value)
            if value != value:
                return 0.0
            return value
        except (TypeError, ValueError):
            return 0.0

    def _safe_bool(self, value):
        if value is None:
            return False
        try:
            if value != value:
                return False
        except TypeError:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    # ---------------------------
    # MAIN GRAPH BUILD
    # ---------------------------
    def build_graph(self, processed_data, entity_data):
        print("Building Heterogeneous Graph...")

        users_df = processed_data["users"]
        tweets_df = processed_data["tweets"]
        edges_df = processed_data["edges"]
        labels_df = processed_data["labels"]

        hashtags_df = entity_data["hashtags"]
        hashtag_edges_df = entity_data["hashtag_edges"]

        urls_df = entity_data["urls"]
        url_edges_df = entity_data["url_edges"]

        # ---------------------------
        # ID MAPS
        # ---------------------------
        user_map, tweet_map, hashtag_map, url_map = self.build_id_maps(
            users_df, tweets_df, hashtags_df, urls_df
        )

        data = HeteroData()

        # ---------------------------
        # NODES (no features yet)
        # ---------------------------
        data["user"].num_nodes = len(user_map)
        data["tweet"].num_nodes = len(tweet_map)
        data["hashtag"].num_nodes = len(hashtag_map)
        data["url"].num_nodes = len(url_map)

        user_x, feature_names, user_feature_stats = self.build_user_features(
            users_df,
            tweets_df,
            hashtag_edges_df,
            url_edges_df,
            user_map
        )
        data["user"].x = user_x
        data["user"].feature_names = feature_names
        data["user"].feature_stats = user_feature_stats

        tweet_x, tweet_feature_names, tweet_feature_source = self.build_tweet_features(tweets_df, tweet_map)
        data["tweet"].x = tweet_x
        data["tweet"].feature_names = tweet_feature_names
        data["tweet"].feature_source = tweet_feature_source

        hashtag_x, hashtag_feature_names, hashtag_feature_stats = self.build_hashtag_features(hashtags_df, hashtag_map)
        data["hashtag"].x = hashtag_x
        data["hashtag"].feature_names = hashtag_feature_names
        data["hashtag"].feature_stats = hashtag_feature_stats

        url_x, url_feature_names, url_feature_stats = self.build_url_features(urls_df, url_map)
        data["url"].x = url_x
        data["url"].feature_names = url_feature_names
        data["url"].feature_stats = url_feature_stats

        print("Node counts:")
        print(data)

        # ---------------------------
        # USER → USER (dynamic relations)
        # ---------------------------
        if not edges_df.empty and {"src", "dst", "type"}.issubset(edges_df.columns):
            edge_types = sorted(set(edges_df["type"].astype(str)))
            for rel in edge_types:
                rel_edges = edges_df[edges_df["type"].astype(str) == rel]
                edge_index = self.add_relation(
                    data,
                    "user",
                    rel,
                    "user",
                    rel_edges["src"],
                    rel_edges["dst"],
                    user_map,
                    user_map,
                )
                if not rel.startswith("rev_"):
                    data["user", f"rev_{rel}", "user"].edge_index = torch.flip(edge_index, dims=[0])
        else:
            # keep a stable empty relation so downstream code has a predictable edge key
            data["user", "follows", "user"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # ---------------------------
        # USER → TWEET (posts via author_id)
        # ---------------------------
        tweet_authors = tweets_df.dropna(subset=["author_id"])

        edge_index = self.add_relation(
            data,
            "user",
            "posts",
            "tweet",
            tweet_authors["author_id"],
            tweet_authors["id"],
            user_map,
            tweet_map
        )
        data["tweet", "rev_posts", "user"].edge_index = torch.flip(edge_index, dims=[0])

        # ---------------------------
        # TWEET → HASHTAG
        # ---------------------------
        edge_index = self.add_relation(
            data,
            "tweet",
            "contains",
            "hashtag",
            hashtag_edges_df["tweet_id"],
            hashtag_edges_df["hashtag"],
            tweet_map,
            hashtag_map
        )
        data["hashtag", "rev_contains", "tweet"].edge_index = torch.flip(edge_index, dims=[0])

        # ---------------------------
        # TWEET → URL
        # ---------------------------
        edge_index = self.add_relation(
            data,
            "tweet",
            "links",
            "url",
            url_edges_df["tweet_id"],
            url_edges_df["url"],
            tweet_map,
            url_map
        )
        data["url", "rev_links", "tweet"].edge_index = torch.flip(edge_index, dims=[0])

        # ---------------------------
        # LABELS (user classification)
        # ---------------------------
        print("Processing labels...")

        label_map = {
            "human": 0,
            "nonbot": 0,
            "non-bot": 0,
            "0": 0,
            0: 0,
            "bot": 1,
            "1": 1,
            1: 1,
        }

        y = torch.full((len(user_map),), -1, dtype=torch.long)

        for _, row in labels_df.iterrows():
            uid = row["id"]
            if uid in user_map:
                label = row["label"]
                if isinstance(label, str):
                    label = label.lower()
                y[user_map[uid]] = label_map.get(label, -1)

        data["user"].y = y

        print("Graph construction complete.")
        return data
