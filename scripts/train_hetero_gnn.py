import argparse
import copy
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
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

from src.data.edge_augmentor import EdgeAugmentor
from src.data.entity_extractor import EntityExtractor
from src.data.graph_builder import GraphBuilder
from src.data.loader import get_loader
from src.data.preprocessor import TwiBot22Preprocessor


def load_config(base_path, dataset_path):
    with open(base_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(dataset_path, encoding="utf-8") as f:
        dataset = yaml.safe_load(f)
    return {**base, **dataset}


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
    else:
        config["debug"]["enabled"] = True
        config["debug"]["max_users"] = int(limit)
    return config


def set_augmentation_config(config, args):
    config.setdefault("augmentation", {})
    config["augmentation"]["knn_k"] = args.knn_k
    config["augmentation"]["min_similarity"] = args.min_similarity
    config["augmentation"]["min_shared_hashtags"] = args.min_shared_hashtags
    config["augmentation"]["min_shared_urls"] = args.min_shared_urls
    config["augmentation"]["max_users_per_token"] = args.max_users_per_token
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def min_class_count(y):
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if len(counts) else 0


def build_split_masks(labels, random_state):
    labels = np.asarray(labels)
    labeled_idx = np.where(labels >= 0)[0]
    labeled_y = labels[labeled_idx]

    first_stratify = labeled_y if min_class_count(labeled_y) >= 2 else None
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        labeled_idx,
        labeled_y,
        test_size=0.30,
        stratify=first_stratify,
        random_state=random_state,
    )
    second_stratify = y_temp if min_class_count(y_temp) >= 2 else None
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.50,
        stratify=second_stratify,
        random_state=random_state,
    )

    n = len(labels)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    return train_mask, val_mask, test_mask


def relation_key(edge_type):
    return "__".join(edge_type)


class RelationLayer(nn.Module):
    def __init__(self, node_types, edge_types, hidden_dim):
        super().__init__()
        self.node_types = node_types
        self.rel_linears = nn.ModuleDict()
        for edge_type in edge_types:
            self.rel_linears[relation_key(edge_type)] = nn.Linear(hidden_dim, hidden_dim)
        self.self_linears = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types
        })

    def forward(self, x_dict, edge_index_dict):
        out_dict = {
            ntype: self.self_linears[ntype](x_dict[ntype])
            for ntype in self.node_types
        }
        degree_dict = {
            ntype: torch.zeros(x_dict[ntype].size(0), device=x_dict[ntype].device)
            for ntype in self.node_types
        }

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if edge_index.numel() == 0:
                continue

            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            src_x = x_dict[src_type]
            messages = self.rel_linears[relation_key(edge_type)](src_x[src_idx])

            out_dict[dst_type].index_add_(0, dst_idx, messages)
            degree_dict[dst_type].index_add_(
                0,
                dst_idx,
                torch.ones(dst_idx.size(0), device=dst_idx.device),
            )

        for ntype in self.node_types:
            degree = degree_dict[ntype].clamp(min=1.0).unsqueeze(-1)
            out_dict[ntype] = out_dict[ntype] / degree

        return out_dict


class HeteroBotGNN(nn.Module):
    def __init__(self, data, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        self.dropout = dropout

        self.input_linears = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()

        for ntype in self.node_types:
            node_store = data[ntype]
            if hasattr(node_store, "x"):
                in_dim = int(node_store.x.size(-1))
                self.input_linears[ntype] = nn.Linear(in_dim, hidden_dim)
            else:
                self.embeddings[ntype] = nn.Embedding(int(node_store.num_nodes), hidden_dim)

        self.layer1 = RelationLayer(self.node_types, self.edge_types, hidden_dim)
        self.layer2 = RelationLayer(self.node_types, self.edge_types, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def _node_features(self, data):
        x_dict = {}
        for ntype in self.node_types:
            node_store = data[ntype]
            if ntype in self.input_linears:
                x = node_store.x.float()
                x_dict[ntype] = self.input_linears[ntype](x)
            else:
                x_dict[ntype] = self.embeddings[ntype].weight
        return x_dict

    def forward(self, data):
        x_dict = self._node_features(data)
        edge_index_dict = {
            edge_type: data[edge_type].edge_index.long()
            for edge_type in self.edge_types
        }

        x_dict = self.layer1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        x_dict = self.layer2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        return self.classifier(x_dict["user"])


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "bot_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "bot_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["human", "bot"],
            zero_division=0,
        ),
    }


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data)
    y = data["user"].y[mask].cpu().numpy()
    preds = logits[mask].argmax(dim=-1).cpu().numpy()
    return compute_metrics(y, preds)


def train_model(data, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = move_graph_to_device(data, device)
    labels = data["user"].y.cpu().numpy()
    train_mask, val_mask, test_mask = build_split_masks(labels, args.random_state)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    model = HeteroBotGNN(data, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    y_train = data["user"].y[train_mask]
    counts = torch.bincount(y_train, minlength=2).float().clamp(min=1.0)
    class_weights = (counts.sum() / (2.0 * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[train_mask], data["user"].y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == 1:
            val_metrics = evaluate(model, data, val_mask)
            val_f1 = val_metrics["macro_f1"]
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = args.patience
            else:
                patience_left -= args.eval_every
                if patience_left <= 0:
                    print("Early stopping triggered.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, data, val_mask)
    test_metrics = evaluate(model, data, test_mask)
    return {
        "best_epoch": best_epoch,
        "train_size": int(train_mask.sum().item()),
        "val_size": int(val_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def move_graph_to_device(data, device):
    if hasattr(data, "to"):
        return data.to(device)

    for key in data.keys():
        store = data[key]
        for attr_name, value in store.__dict__.items():
            if torch.is_tensor(value):
                setattr(store, attr_name, value.to(device))
    return data


def graph_summary(graph):
    summary = {
        "user_nodes": graph["user"].num_nodes,
        "tweet_nodes": graph["tweet"].num_nodes,
        "hashtag_nodes": graph["hashtag"].num_nodes,
        "url_nodes": graph["url"].num_nodes,
        "edge_counts": {},
    }
    for edge_type in graph.edge_types:
        summary["edge_counts"][edge_type] = int(graph[edge_type].edge_index.shape[1])
    return summary


def write_report(output_path, summary, results):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    val = results["val_metrics"]
    test = results["test_metrics"]
    cm = test["confusion_matrix"]
    edge_lines = [f"- `{etype}`: {count}" for etype, count in summary["edge_counts"].items()]

    report = f"""# Hetero-GNN Bot/No-Bot Results

## Dataset / Graph Summary
- User nodes: {summary["user_nodes"]}
- Tweet nodes: {summary["tweet_nodes"]}
- Hashtag nodes: {summary["hashtag_nodes"]}
- URL nodes: {summary["url_nodes"]}

## Edge Counts
{chr(10).join(edge_lines)}

## Split
- Train: {results["train_size"]}
- Validation: {results["val_size"]}
- Test: {results["test_size"]}
- Best epoch: {results["best_epoch"]}

## Metrics
| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Validation | {val["accuracy"]:.4f} | {val["macro_f1"]:.4f} |
| Test | {test["accuracy"]:.4f} | {test["macro_f1"]:.4f} |

Bot precision: {test["bot_precision"]:.4f}  
Bot recall: {test["bot_recall"]:.4f}

## Confusion Matrix
Rows are true labels, columns are predicted labels, ordered as human, bot.

|  | Pred human | Pred bot |
| --- | ---: | ---: |
| True human | {cm[0][0]} | {cm[0][1]} |
| True bot | {cm[1][0]} | {cm[1][1]} |

## Classification Report
```text
{test["classification_report"]}
```
"""
    output_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report to {output_path}")


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

    print("\n===== STAGE: EDGE AUGMENTATION =====")
    augmentor = EdgeAugmentor(config)
    augmented_edges = augmentor.augment(
        processed["users"],
        processed["tweets"],
        entities["hashtag_edges"],
        entities["url_edges"],
    )
    if not augmented_edges.empty:
        processed["edges"] = (
            pd.concat([processed["edges"], augmented_edges], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
    print(f"Total user-user edges after augmentation: {len(processed['edges'])}")

    print("\n===== STAGE: GRAPH BUILDING =====")
    builder = GraphBuilder(config)
    graph = builder.build_graph(processed, entities)
    return graph


def parse_args():
    parser = argparse.ArgumentParser(description="Train a hetero-GNN bot/no-bot classifier.")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--dataset-config", default="config/twitter_human_bots.yaml")
    parser.add_argument("--csv", help="Path to twitter_human_bots_dataset.csv")
    parser.add_argument("--download", action="store_true", help="Download the CSV if it is missing")
    parser.add_argument("--limit", type=int, help="Use only the first N users for a smoke run")
    parser.add_argument("--output", default="reports/hetero_gnn_results.md")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.5)
    parser.add_argument("--min-shared-hashtags", type=int, default=2)
    parser.add_argument("--min-shared-urls", type=int, default=1)
    parser.add_argument("--max-users-per-token", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_state)

    config = load_config(args.base_config, args.dataset_config)
    config = ensure_dataset(config, args.csv, args.download)
    config = set_runtime_limits(config, args.limit)
    config = set_augmentation_config(config, args)

    graph = build_graph(config)
    summary = graph_summary(graph)
    print("\n===== GRAPH SUMMARY =====")
    print(summary)

    print("\n===== STAGE: TRAINING HETERO-GNN =====")
    results = train_model(graph, args)

    test = results["test_metrics"]
    print("\n===== TEST METRICS =====")
    print(f"Accuracy: {test['accuracy']:.4f}")
    print(f"Macro-F1: {test['macro_f1']:.4f}")
    print(f"Bot precision: {test['bot_precision']:.4f}")
    print(f"Bot recall: {test['bot_recall']:.4f}")
    print("Confusion matrix [human, bot]:")
    print(test["confusion_matrix"])

    write_report(args.output, summary, results)


if __name__ == "__main__":
    main()
