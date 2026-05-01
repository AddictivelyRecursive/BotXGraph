import argparse
import copy
import json
import math
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
from src.visualization.report_plots import generate_all_plots


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
    def __init__(self, node_types, edge_types, hidden_dim, num_heads=4, num_global_tokens=8, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = max(1, int(num_heads))
        self.head_dim = max(1, hidden_dim // self.num_heads)
        self.dropout = dropout

        self.rel_src_linears = nn.ModuleDict()
        self.rel_dst_linears = nn.ModuleDict()
        for edge_type in edge_types:
            key = relation_key(edge_type)
            self.rel_src_linears[key] = nn.Linear(hidden_dim, hidden_dim)
            self.rel_dst_linears[key] = nn.Linear(hidden_dim, hidden_dim)
        self.self_linears = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types
        })
        self.rel_queries = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types
        })
        self.global_queries = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types
        })
        self.feed_forward = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for node_type in node_types
        })
        self.norm1 = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in node_types})
        self.norm2 = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in node_types})
        self.global_centroids = nn.Parameter(torch.randn(num_global_tokens, hidden_dim) * 0.02)

    def forward(self, x_dict, edge_index_dict):
        relation_messages = {ntype: [] for ntype in self.node_types}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if edge_index.numel() == 0:
                continue

            key = relation_key(edge_type)
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            src_x = self.rel_src_linears[key](x_dict[src_type])[src_idx]
            dst_x = self.rel_dst_linears[key](x_dict[dst_type])[dst_idx]
            edge_messages = 0.5 * (src_x + dst_x)

            agg = torch.zeros(
                x_dict[dst_type].size(0),
                self.hidden_dim,
                device=x_dict[dst_type].device,
            )
            degree = torch.zeros(x_dict[dst_type].size(0), device=x_dict[dst_type].device)
            agg.index_add_(0, dst_idx, edge_messages)
            degree.index_add_(0, dst_idx, torch.ones(dst_idx.size(0), device=dst_idx.device))
            agg = agg / degree.clamp(min=1.0).unsqueeze(-1)
            relation_messages[dst_type].append((key, agg))

        out_dict = {}
        relation_importance = {}
        centroid_scale = math.sqrt(float(self.hidden_dim))
        for ntype in self.node_types:
            base = self.self_linears[ntype](x_dict[ntype])
            combined = base

            if relation_messages[ntype]:
                rel_names = [name for name, _ in relation_messages[ntype]]
                rel_stack = torch.stack([msg for _, msg in relation_messages[ntype]], dim=1)
                query = self.rel_queries[ntype](x_dict[ntype]).unsqueeze(1)
                scores = (query * rel_stack).sum(dim=-1) / centroid_scale
                weights = torch.softmax(scores, dim=1)
                rel_mix = (weights.unsqueeze(-1) * rel_stack).sum(dim=1)
                combined = combined + rel_mix
                relation_importance[ntype] = {
                    rel_name: float(weights[:, idx].mean().detach().cpu().item())
                    for idx, rel_name in enumerate(rel_names)
                }

            global_query = self.global_queries[ntype](x_dict[ntype])
            global_scores = global_query @ self.global_centroids.t() / centroid_scale
            global_weights = torch.softmax(global_scores, dim=-1)
            global_context = global_weights @ self.global_centroids
            combined = self.norm1[ntype](combined + global_context)
            combined = self.norm2[ntype](combined + self.feed_forward[ntype](combined))
            out_dict[ntype] = F.dropout(combined, p=self.dropout, training=self.training)

        return out_dict, relation_importance


class HeteroBotGNN(nn.Module):
    def __init__(self, data, hidden_dim=64, dropout=0.3, num_heads=4, num_global_tokens=8):
        super().__init__()
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_global_tokens = num_global_tokens
        self.last_relation_importance = {}

        self.input_linears = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()

        for ntype in self.node_types:
            node_store = data[ntype]
            if hasattr(node_store, "x"):
                in_dim = int(node_store.x.size(-1))
                self.input_linears[ntype] = nn.Linear(in_dim, hidden_dim)
            else:
                self.embeddings[ntype] = nn.Embedding(int(node_store.num_nodes), hidden_dim)

        self.layer1 = RelationLayer(
            self.node_types,
            self.edge_types,
            hidden_dim,
            num_heads=num_heads,
            num_global_tokens=num_global_tokens,
            dropout=dropout,
        )
        self.layer2 = RelationLayer(
            self.node_types,
            self.edge_types,
            hidden_dim,
            num_heads=num_heads,
            num_global_tokens=num_global_tokens,
            dropout=dropout,
        )
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

    def forward_with_state(self, data):
        x_dict = self._node_features(data)
        edge_index_dict = {
            edge_type: data[edge_type].edge_index.long()
            for edge_type in self.edge_types
        }

        x_dict, rel1 = self.layer1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict, rel2 = self.layer2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        user_rel = {}
        for layer_rel in (rel1.get("user", {}), rel2.get("user", {})):
            for rel_name, score in layer_rel.items():
                user_rel.setdefault(rel_name, []).append(score)
        self.last_relation_importance = {
            rel_name: float(np.mean(scores))
            for rel_name, scores in user_rel.items()
        }

        logits = self.classifier(x_dict["user"])
        return logits, x_dict

    def forward(self, data):
        logits, _ = self.forward_with_state(data)
        return logits


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


def metrics_to_jsonable(metrics):
    payload = dict(metrics)
    payload["confusion_matrix"] = np.asarray(payload["confusion_matrix"]).tolist()
    return payload


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mask_to_indices(mask):
    return mask.nonzero(as_tuple=False).view(-1).detach().cpu().numpy()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data)
    y = data["user"].y[mask].cpu().numpy()
    preds = logits[mask].argmax(dim=-1).cpu().numpy()
    return compute_metrics(y, preds)


@torch.no_grad()
def predict_mask(model, data, mask):
    model.eval()
    logits = model(data)
    preds = logits[mask].argmax(dim=-1).cpu().numpy()
    y_true = data["user"].y[mask].cpu().numpy()
    indices = mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    return y_true, preds, indices


@torch.no_grad()
def collect_epoch_state(model, data, masks):
    model.eval()
    logits, x_dict = model.forward_with_state(data)
    user_logits = logits.detach().cpu()
    user_probs = torch.softmax(logits, dim=-1).detach().cpu()
    user_embeddings = x_dict["user"].detach().cpu()
    all_labels = data["user"].y.detach().cpu().numpy()

    split_payload = {}
    for split_name, mask in masks.items():
        idx = mask_to_indices(mask)
        split_logits = user_logits[idx].numpy()
        split_probs = user_probs[idx].numpy()
        split_preds = split_probs.argmax(axis=-1)
        split_labels = all_labels[idx]
        split_payload[split_name] = {
            "indices": idx,
            "labels": split_labels,
            "predictions": split_preds,
            "logits": split_logits,
            "probabilities": split_probs,
            "metrics": compute_metrics(split_labels, split_preds),
        }

    return {
        "user_logits": user_logits.numpy(),
        "user_probabilities": user_probs.numpy(),
        "user_embeddings": user_embeddings.numpy(),
        "split_payload": split_payload,
        "relation_importance": dict(sorted(model.last_relation_importance.items(), key=lambda x: x[1], reverse=True)),
    }


def save_epoch_artifacts(artifacts_dir, epoch, state, loss_value, save_embeddings=True, save_checkpoint_payload=None):
    artifacts_dir = Path(artifacts_dir)
    epoch_dir = artifacts_dir / "epochs" / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    val_metrics = metrics_to_jsonable(state["split_payload"]["val"]["metrics"])
    test_metrics = metrics_to_jsonable(state["split_payload"]["test"]["metrics"])
    train_metrics = metrics_to_jsonable(state["split_payload"]["train"]["metrics"])

    meta = {
        "epoch": int(epoch),
        "loss": float(loss_value),
        "relation_importance": state["relation_importance"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(epoch_dir / "metrics.json", meta)

    np.savez_compressed(
        epoch_dir / "user_logits_probs.npz",
        logits=state["user_logits"],
        probabilities=state["user_probabilities"],
    )

    for split_name, split in state["split_payload"].items():
        np.savez_compressed(
            epoch_dir / f"{split_name}_predictions.npz",
            indices=split["indices"],
            labels=split["labels"],
            predictions=split["predictions"],
            logits=split["logits"],
            probabilities=split["probabilities"],
        )

    if save_embeddings:
        np.savez_compressed(
            epoch_dir / "user_embeddings.npz",
            embeddings=state["user_embeddings"],
        )

    if save_checkpoint_payload is not None:
        torch.save(save_checkpoint_payload, epoch_dir / "checkpoint.pt")


def train_model(data, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = move_graph_to_device(data, device)
    labels = data["user"].y.cpu().numpy()
    train_mask, val_mask, test_mask = build_split_masks(labels, args.random_state)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    model = HeteroBotGNN(
        data,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_heads=args.num_heads,
        num_global_tokens=args.num_global_tokens,
    ).to(device)
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
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mask_cpu = {
        "train": train_mask.detach().cpu(),
        "val": val_mask.detach().cpu(),
        "test": test_mask.detach().cpu(),
    }
    history = {
        "epoch": [],
        "loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }
    save_json(
        artifacts_dir / "run_config.json",
        {
            "device": str(device),
            "epochs": args.epochs,
            "patience": args.patience,
            "eval_every": args.eval_every,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_heads": args.num_heads,
            "num_global_tokens": args.num_global_tokens,
            "save_epoch_artifacts": args.save_epoch_artifacts,
            "save_checkpoints": args.save_checkpoints,
            "save_embeddings": args.save_embeddings,
            "split_sizes": {
                "train": int(train_mask.sum().item()),
                "val": int(val_mask.sum().item()),
                "test": int(test_mask.sum().item()),
            },
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[train_mask], data["user"].y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == 1:
            epoch_state = collect_epoch_state(model, data, mask_cpu)
            val_metrics = epoch_state["split_payload"]["val"]["metrics"]
            val_f1 = val_metrics["macro_f1"]
            history["epoch"].append(epoch)
            history["loss"].append(float(loss.item()))
            history["val_accuracy"].append(float(val_metrics["accuracy"]))
            history["val_macro_f1"].append(float(val_f1))
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_f1:.4f}"
            )

            if args.save_epoch_artifacts:
                checkpoint_payload = None
                if args.save_checkpoints:
                    checkpoint_payload = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": float(loss.item()),
                        "val_metrics": metrics_to_jsonable(val_metrics),
                    }
                save_epoch_artifacts(
                    artifacts_dir=artifacts_dir,
                    epoch=epoch,
                    state=epoch_state,
                    loss_value=float(loss.item()),
                    save_embeddings=args.save_embeddings,
                    save_checkpoint_payload=checkpoint_payload,
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

    final_state = collect_epoch_state(model, data, mask_cpu)
    val_metrics = final_state["split_payload"]["val"]["metrics"]
    test_metrics = final_state["split_payload"]["test"]["metrics"]
    test_y_true = final_state["split_payload"]["test"]["labels"]
    test_y_pred = final_state["split_payload"]["test"]["predictions"]
    test_indices = final_state["split_payload"]["test"]["indices"]
    feature_saliency = compute_feature_saliency(model, data, test_mask)
    save_json(
        artifacts_dir / "history.json",
        {
            "best_epoch": int(best_epoch),
            "history": history,
            "final_val_metrics": metrics_to_jsonable(val_metrics),
            "final_test_metrics": metrics_to_jsonable(test_metrics),
            "relation_importance": final_state["relation_importance"],
        },
    )
    np.savez_compressed(
        artifacts_dir / "final_outputs.npz",
        test_indices=test_indices,
        test_labels=test_y_true,
        test_predictions=test_y_pred,
        user_logits=final_state["user_logits"],
        user_probabilities=final_state["user_probabilities"],
    )
    if args.save_embeddings:
        np.savez_compressed(
            artifacts_dir / "final_user_embeddings.npz",
            embeddings=final_state["user_embeddings"],
        )
    if args.save_checkpoints and best_state is not None:
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "final_val_metrics": metrics_to_jsonable(val_metrics),
                "final_test_metrics": metrics_to_jsonable(test_metrics),
            },
            artifacts_dir / "best_model.pt",
        )
    return {
        "model_name": "RelGT-inspired relational attention network",
        "best_epoch": best_epoch,
        "train_size": int(train_mask.sum().item()),
        "val_size": int(val_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "artifacts_dir": str(artifacts_dir),
        "save_embeddings": bool(args.save_embeddings),
        "save_checkpoints": bool(args.save_checkpoints),
        "feature_saliency": feature_saliency,
        "relation_importance": final_state["relation_importance"],
        "test_labels": test_y_true.tolist(),
        "test_predictions": test_y_pred.tolist(),
        "test_indices": test_indices.tolist(),
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
        "tweet_feature_source": getattr(graph["tweet"], "feature_source", "unknown"),
        "user_feature_names": list(getattr(graph["user"], "feature_names", [])),
        "edge_counts": {},
    }
    for edge_type in graph.edge_types:
        summary["edge_counts"][edge_type] = int(graph[edge_type].edge_index.shape[1])
    return summary


def compute_feature_saliency(model, data, mask):
    model.eval()
    original_x = data["user"].x
    user_x = original_x.detach().clone().requires_grad_(True)
    data["user"].x = user_x

    logits = model(data)
    objective = logits[mask, 1].sum()
    model.zero_grad()
    objective.backward()
    scores = user_x.grad[mask].abs().mean(dim=0).detach().cpu().tolist()

    data["user"].x = original_x
    return scores


def build_project_recommendation(summary):
    return (
        "Current `twitter-human-bots` data is sufficient for a wrap-up demo because it trains fast, "
        "produces clean bot/human metrics, and already supports user-text-URL heterogeneous modeling. "
        "If you want one extra easy next step later, reuse the already-wired `TwiBot-22` pipeline as the "
        "follow-up dataset rather than adding a brand-new source right now."
    )


def write_report(output_path, summary, results, plot_paths=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    val = results["val_metrics"]
    test = results["test_metrics"]
    cm = test["confusion_matrix"]
    edge_lines = [f"- `{etype}`: {count}" for etype, count in summary["edge_counts"].items()]
    relation_lines = [
        f"- `{rel}`: {score:.4f}" for rel, score in results["relation_importance"].items()
    ] or ["- No user-targeted relations received attention mass."]

    saliency = results.get("feature_saliency", [])
    feature_ranking = sorted(
        zip(summary["user_feature_names"], saliency),
        key=lambda x: x[1],
        reverse=True,
    )
    top_features = [f"- `{name}`: {score:.4f}" for name, score in feature_ranking[:8]]
    recommendation = build_project_recommendation(summary)
    plot_lines = [f"- `{Path(plot_path).name}`" for plot_path in (plot_paths or [])]
    artifact_lines = [
        "- `run_config.json`",
        "- `history.json`",
        "- `final_outputs.npz`",
        "- `final_user_embeddings.npz`" if results.get("save_embeddings", True) else None,
        "- `best_model.pt`" if results.get("save_checkpoints", True) else None,
        "- `epochs/epoch_XXX/metrics.json`",
        "- `epochs/epoch_XXX/*_predictions.npz`",
        "- `epochs/epoch_XXX/user_logits_probs.npz`",
        "- `epochs/epoch_XXX/user_embeddings.npz`" if results.get("save_embeddings", True) else None,
        "- `epochs/epoch_XXX/checkpoint.pt`" if results.get("save_checkpoints", True) else None,
    ]
    artifact_lines = [line for line in artifact_lines if line is not None]

    report = f"""# RelGT-Inspired Hetero-GNN Bot/No-Bot Results

Model: {results["model_name"]}

## Dataset / Graph Summary
- User nodes: {summary["user_nodes"]}
- Tweet nodes: {summary["tweet_nodes"]}
- Hashtag nodes: {summary["hashtag_nodes"]}
- URL nodes: {summary["url_nodes"]}
- Tweet text encoder: `{summary["tweet_feature_source"]}`
- User profile features: normalized z-score features
- URL node features: domain-level lexical features

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

## Explainability Snapshot
Top user feature saliency:
{chr(10).join(top_features)}

Relation importance into user prediction:
{chr(10).join(relation_lines)}

## Visual Outputs
{chr(10).join(plot_lines) if plot_lines else "- No plots generated."}

## Stored Artifacts
Artifact directory: `{results["artifacts_dir"]}`
{chr(10).join(artifact_lines)}

## Wrap-up Recommendation
{recommendation}
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
    return graph, processed, entities


def parse_args():
    parser = argparse.ArgumentParser(description="Train a hetero-GNN bot/no-bot classifier.")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--dataset-config", default="config/twitter_human_bots.yaml")
    parser.add_argument("--csv", help="Path to twitter_human_bots_dataset.csv")
    parser.add_argument("--download", action="store_true", help="Download the CSV if it is missing")
    parser.add_argument("--limit", type=int, help="Use only the first N users for a smoke run")
    parser.add_argument("--output", default="reports/hetero_gnn_results.md")
    parser.add_argument("--plots-dir", default="reports/plots/hetero_gnn")
    parser.add_argument("--artifacts-dir", default="reports/artifacts/hetero_gnn")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-epoch-artifacts", action="store_true", default=True)
    parser.add_argument("--no-save-epoch-artifacts", dest="save_epoch_artifacts", action="store_false")
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    parser.add_argument("--save-embeddings", action="store_true", default=True)
    parser.add_argument("--no-save-embeddings", dest="save_embeddings", action="store_false")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-global-tokens", type=int, default=8)

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

    graph, processed, _ = build_graph(config)
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

    plot_paths = []
    if not args.skip_plots:
        print("\n===== STAGE: GENERATING PLOTS =====")
        plot_paths = generate_all_plots(args.plots_dir, summary, results, graph, processed)
        for plot_path in plot_paths:
            print(f"Saved plot: {plot_path}")

    write_report(args.output, summary, results, plot_paths=plot_paths)


if __name__ == "__main__":
    main()
