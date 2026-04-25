import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.entity_extractor import EntityExtractor
from src.data.graph_builder import GraphBuilder
from src.data.loader import get_loader
from src.data.preprocessor import TwiBot22Preprocessor


def load_config(base_path, dataset_path):
    with open(base_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = yaml.safe_load(f)

    config = {**base, **dataset}
    return config


def ensure_dataset(config, csv_path, download):
    if csv_path:
        config["dataset"]["files"]["accounts"] = os.path.abspath(csv_path)
        return config

    dataset_path = Path(config["paths"]["raw_data"]) / config["dataset"]["name"]
    account_file = dataset_path / config["dataset"]["files"]["accounts"]

    if account_file.exists():
        return config

    if not download:
        raise FileNotFoundError(
            f"Dataset CSV not found at {account_file}. "
            "Pass --csv /path/to/twitter_human_bots_dataset.csv, or rerun with --download."
        )

    dataset_path.mkdir(parents=True, exist_ok=True)
    download_url = config["dataset"]["source"]["download_url"]
    print(f"Downloading dataset from {download_url}...")
    urllib.request.urlretrieve(download_url, account_file)
    print(f"Saved dataset to {account_file}")
    return config


def set_runtime_limits(config, limit):
    config.setdefault("debug", {})

    if limit is None:
        config["debug"]["enabled"] = False
        config["debug"]["max_users"] = None
        return config

    config["debug"]["enabled"] = True
    config["debug"]["max_users"] = limit
    return config


def build_graph(config):
    print("\n===== STAGE: LOADING =====")
    loader = get_loader(config)
    raw_data = loader.load_all()

    print("\n===== STAGE: PREPROCESSING =====")
    preprocessor = TwiBot22Preprocessor(config)
    processed = preprocessor.process_all(raw_data)

    print("\n===== STAGE: ENTITY EXTRACTION =====")
    extractor = EntityExtractor(config)
    entities = extractor.extract_all(processed["tweets"])

    print("\n===== STAGE: GRAPH BUILDING =====")
    builder = GraphBuilder(config)
    graph = builder.build_graph(processed, entities)

    return graph, processed, entities


def train_baseline(graph, random_state):
    x = graph["user"].x.numpy()
    y = graph["user"].y.numpy()

    labeled_mask = y >= 0
    x = x[labeled_mask]
    y = y[labeled_mask]

    if len(set(y.tolist())) < 2:
        raise ValueError("Need both human and bot labels for training")

    first_stratify = y if min_class_count(y) >= 2 else None
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        stratify=first_stratify,
        random_state=random_state,
    )
    second_stratify = y_temp if min_class_count(y_temp) >= 2 else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=second_stratify,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    results = {
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_macro_f1": f1_score(y_test, test_pred, average="macro"),
        "test_bot_precision": precision_score(y_test, test_pred, pos_label=1, zero_division=0),
        "test_bot_recall": recall_score(y_test, test_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            test_pred,
            labels=[0, 1],
            target_names=["human", "bot"],
            zero_division=0,
        ),
    }

    return results


def min_class_count(y):
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if len(counts) else 0


def graph_summary(graph):
    summary = {
        "user_nodes": graph["user"].num_nodes,
        "tweet_nodes": graph["tweet"].num_nodes,
        "hashtag_nodes": graph["hashtag"].num_nodes,
        "url_nodes": graph["url"].num_nodes,
        "edge_counts": {},
    }

    for edge_type in graph.edge_types:
        summary["edge_counts"][edge_type] = graph[edge_type].edge_index.shape[1]

    return summary


def write_report(output_path, summary, results, feature_names):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = results["confusion_matrix"]
    edge_lines = [
        f"- `{edge_type}`: {count}"
        for edge_type, count in summary["edge_counts"].items()
    ]

    report = f"""# Fast Bot/No-Bot KG Baseline

## Dataset / Graph Summary
- User nodes: {summary["user_nodes"]}
- Tweet nodes: {summary["tweet_nodes"]}
- Hashtag nodes: {summary["hashtag_nodes"]}
- URL nodes: {summary["url_nodes"]}

## Edge Counts
{chr(10).join(edge_lines)}

## Feature Set
{", ".join(feature_names)}

## Split
- Train: {results["train_size"]}
- Validation: {results["val_size"]}
- Test: {results["test_size"]}

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | {results["val_accuracy"]:.4f} | {results["val_macro_f1"]:.4f} |
| Test | {results["test_accuracy"]:.4f} | {results["test_macro_f1"]:.4f} |

Bot precision: {results["test_bot_precision"]:.4f}  
Bot recall: {results["test_bot_recall"]:.4f}

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | {cm[0][0]} | {cm[0][1]} |
| True bot | {cm[1][0]} | {cm[1][1]} |

## Classification Report
```text
{results["classification_report"]}
```

Note: This is a fast KG-compatible baseline, not final deep GNN training.
"""

    output_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a fast bot/no-bot KG baseline.")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--dataset-config", default="config/twitter_human_bots.yaml")
    parser.add_argument("--csv", help="Path to twitter_human_bots_dataset.csv")
    parser.add_argument("--download", action="store_true", help="Download the CSV if it is missing")
    parser.add_argument("--limit", type=int, help="Use only the first N users for a smoke run")
    parser.add_argument("--output", default="reports/fast_bot_baseline.md")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.base_config, args.dataset_config)
    config = ensure_dataset(config, args.csv, args.download)
    config = set_runtime_limits(config, args.limit)

    graph, _, _ = build_graph(config)
    summary = graph_summary(graph)

    print("\n===== GRAPH SUMMARY =====")
    print(summary)

    print("\n===== STAGE: TRAINING BASELINE =====")
    results = train_baseline(graph, args.random_state)

    print("\n===== TEST METRICS =====")
    print(f"Accuracy: {results['test_accuracy']:.4f}")
    print(f"Macro-F1: {results['test_macro_f1']:.4f}")
    print(f"Bot precision: {results['test_bot_precision']:.4f}")
    print(f"Bot recall: {results['test_bot_recall']:.4f}")
    print("Confusion matrix [human, bot]:")
    print(results["confusion_matrix"])

    write_report(args.output, summary, results, graph["user"].feature_names)


if __name__ == "__main__":
    main()
