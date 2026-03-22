import yaml
from src.data.loader import TwiBot22Loader
from src.data.preprocessor import TwiBot22Preprocessor
from src.data.entity_extractor import EntityExtractor
from src.data.graph_builder import GraphBuilder

def load_config():
    with open("config/base.yaml") as f:
        base = yaml.safe_load(f)

    with open("config/twibot22.yaml") as f:
        dataset = yaml.safe_load(f)

    config = {**base, **dataset}
    return config


if __name__ == "__main__":
    config = load_config()
    print("\n===== STAGE: LOADING =====")
    loader = TwiBot22Loader(config)
    data = loader.load_all()

    # print("\n===== LOADER CHECK =====")
    # print("Users sample:", data["users"][0])
    # print("Tweets sample:", data["tweets"][0])
    # print("Edges sample:\n", data["edges"].head())
    # print("Labels sample:\n", data["labels"].head())
    
    print("\n===== STAGE: PREPROCESSING =====")
    preprocessor = TwiBot22Preprocessor(config)
    processed = preprocessor.process_all(data)
    edges_df = processed["edges"]
    
    print("\n===== ID DEBUG =====")

    print("Sample user IDs:")
    print(processed["users"]["id"].head().tolist())

    print("\nSample tweet author_ids:")
    print(processed["tweets"]["author_id"].head().tolist())

    print("\n===== EDGE TYPES =====")
    print(edges_df["type"].unique())

    # print("\n===== AFTER PREPROCESSING =====")
    # print(processed["users"].head())
    # print(processed["tweets"].head())
    # print(processed["edges"].head())
    # print(processed["labels"].head())
    
    print("\n===== STAGE: ENTITY EXTRACTION =====")
    extractor = EntityExtractor(config)
    entities = extractor.extract_all(processed["tweets"])

    # print("\n===== EXTRACTION CHECK =====")
    # print(entities["hashtags"].head())
    # print(entities["hashtag_edges"].head())
    # print(entities["urls"].head())
    # print(entities["url_edges"].head())
    
    # print("\n===== PREPROCESSING CHECKS =====")
    # users_df = processed["users"]
    # tweets_df = processed["tweets"]
    # edges_df = processed["edges"]
    # labels_df = processed["labels"]
    
    print("\n===== STAGE: GRAPH BUILDING =====")
    builder = GraphBuilder(config)
    graph = builder.build_graph(processed, entities)

    print("\n===== GRAPH SUMMARY =====")
    print(graph)
    print("User nodes:", graph["user"].num_nodes)
    print("Tweet nodes:", graph["tweet"].num_nodes)
    print("Hashtag nodes:", graph["hashtag"].num_nodes)
    print("URL nodes:", graph["url"].num_nodes)

    print("\nEdge types:")
    for edge_type in graph.edge_types:
        print(edge_type, graph[edge_type].edge_index.shape)
    
    
    # ---------------- For Debugging ----------------
    # # ---------------- USERS ----------------
    # print("\n[USERS]")
    # print("Columns:", users_df.columns.tolist())
    # print("Count:", len(users_df))
    # print("Duplicate IDs:", users_df["id"].duplicated().sum())

    # # ---------------- TWEETS ----------------
    # print("\n[TWEETS]")
    # print("Columns:", tweets_df.columns.tolist())
    # print("Count:", len(tweets_df))
    # print("Missing text:", tweets_df["text"].isna().sum())
    # print("Missing author_id:", tweets_df["author_id"].isna().sum())

    # # ---------------- EDGES ----------------
    # print("\n[EDGES]")
    # print("Columns:", edges_df.columns.tolist())
    # print("Count:", len(edges_df))
    # print(edges_df.head())

    # # ---------------- LABELS ----------------
    # print("\n[LABELS]")
    # print("Columns:", labels_df.columns.tolist())
    # print("Count:", len(labels_df))
    # print(labels_df.head())