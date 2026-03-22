import torch
from torch_geometric.data import HeteroData


class GraphBuilder:
    def __init__(self, config):
        self.config = config

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

        print("Node counts:")
        print(data)

        # ---------------------------
        # USER → USER (follows)
        # ---------------------------
        follow_edges = edges_df[
            edges_df["type"].isin(["following", "followers"])
        ]

        edge_index = self.build_edge_index(
            follow_edges["src"],
            follow_edges["dst"],
            user_map,
            user_map
        )

        data["user", "follows", "user"].edge_index = edge_index

        # ---------------------------
        # USER → TWEET (posts via author_id)
        # ---------------------------
        tweet_authors = tweets_df.dropna(subset=["author_id"])

        edge_index = self.build_edge_index(
            tweet_authors["author_id"],
            tweet_authors["id"],
            user_map,
            tweet_map
        )

        data["user", "posts", "tweet"].edge_index = edge_index

        # ---------------------------
        # TWEET → HASHTAG
        # ---------------------------
        edge_index = self.build_edge_index(
            hashtag_edges_df["tweet_id"],
            hashtag_edges_df["hashtag"],
            tweet_map,
            hashtag_map
        )

        data["tweet", "contains", "hashtag"].edge_index = edge_index

        # ---------------------------
        # TWEET → URL
        # ---------------------------
        edge_index = self.build_edge_index(
            url_edges_df["tweet_id"],
            url_edges_df["url"],
            tweet_map,
            url_map
        )

        data["tweet", "links", "url"].edge_index = edge_index

        # ---------------------------
        # LABELS (user classification)
        # ---------------------------
        print("Processing labels...")

        label_map = {"human": 0, "bot": 1}

        y = torch.full((len(user_map),), -1, dtype=torch.long)

        for _, row in labels_df.iterrows():
            uid = row["id"]
            if uid in user_map:
                y[user_map[uid]] = label_map.get(row["label"], -1)

        data["user"].y = y

        print("Graph construction complete.")
        return data