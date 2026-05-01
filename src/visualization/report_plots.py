from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def generate_all_plots(output_dir, summary, results, graph, processed_data):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    saved.append(plot_training_curves(results["history"], output_dir / "training_curves.png"))
    saved.append(plot_edge_type_counts(summary["edge_counts"], output_dir / "edge_type_counts.png"))
    saved.append(plot_confusion_matrix(results["test_metrics"]["confusion_matrix"], output_dir / "confusion_matrix.png"))
    saved.append(plot_feature_saliency(summary["user_feature_names"], results["feature_saliency"], output_dir / "feature_saliency.png"))
    saved.append(plot_relation_importance(results["relation_importance"], output_dir / "relation_importance.png"))
    saved.append(plot_graph_schema(summary["edge_counts"], output_dir / "graph_schema.png"))
    saved.append(plot_user_projection(graph, results, output_dir / "user_projection.png"))
    saved.append(plot_similarity_subgraph(processed_data, output_dir / "similarity_subgraph.png"))
    return [str(path) for path in saved if path is not None]


def plot_training_curves(history, output_path):
    epochs = history.get("epoch", [])
    if not epochs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history.get("loss", []), marker="o", linewidth=2, color="#0f766e")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history.get("val_accuracy", []), marker="o", linewidth=2, label="Val Accuracy", color="#1d4ed8")
    axes[1].plot(epochs, history.get("val_macro_f1", []), marker="s", linewidth=2, label="Val Macro-F1", color="#b45309")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_edge_type_counts(edge_counts, output_path):
    items = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{src}\n{rel}\n{dst}" for (src, rel, dst), _ in items]
    values = [count for _, count in items]

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.55)))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Edge Count")
    ax.set_title("Heterogeneous Graph Edge Counts")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(cm, output_path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["human", "bot"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_saliency(feature_names, saliency, output_path, top_k=10):
    pairs = sorted(zip(feature_names, saliency), key=lambda x: x[1], reverse=True)[:top_k]
    labels = [name for name, _ in pairs][::-1]
    values = [score for _, score in pairs][::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels, values, color="#7c3aed")
    ax.set_xlabel("Average absolute gradient")
    ax.set_title("Top User Feature Saliency")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_relation_importance(relation_importance, output_path):
    items = list(relation_importance.items())[:10]
    labels = [name for name, _ in items][::-1]
    values = [score for _, score in items][::-1]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, values, color="#ea580c")
    ax.set_xlabel("Mean attention weight")
    ax.set_title("Relation Importance for User Prediction")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_graph_schema(edge_counts, output_path):
    node_pos = {
        "user": (0.15, 0.55),
        "tweet": (0.48, 0.55),
        "hashtag": (0.78, 0.78),
        "url": (0.78, 0.28),
    }
    node_colors = {
        "user": "#dbeafe",
        "tweet": "#dcfce7",
        "hashtag": "#fef3c7",
        "url": "#fee2e2",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("BotXGraph Heterogeneous Schema", fontsize=16, pad=16)
    ax.axis("off")

    for node_type, (x, y) in node_pos.items():
        rect = plt.Rectangle((x - 0.09, y - 0.07), 0.18, 0.14, fc=node_colors[node_type], ec="#1f2937", lw=1.8)
        ax.add_patch(rect)
        ax.text(x, y, node_type.upper(), ha="center", va="center", fontsize=13, fontweight="bold")

    for edge_type, count in edge_counts.items():
        src, rel, dst = edge_type
        if src not in node_pos or dst not in node_pos:
            continue
        x1, y1 = node_pos[src]
        x2, y2 = node_pos[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2, color="#475569", alpha=0.7))
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(xm, ym, f"{rel}\n{count}", ha="center", va="center", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_user_projection(graph, results, output_path):
    x = graph["user"].x.detach().cpu().numpy()
    y_true = graph["user"].y.detach().cpu().numpy()
    test_idx = np.asarray(results["test_indices"], dtype=int)
    y_pred = np.asarray(results["test_predictions"], dtype=int)

    if x.shape[0] > 4000:
        keep = np.linspace(0, x.shape[0] - 1, 4000, dtype=int)
        x = x[keep]
        y_true = y_true[keep]

    coords = PCA(n_components=2, random_state=42).fit_transform(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    actual_colors = np.where(y_true == 1, "#dc2626", "#2563eb")
    axes[0].scatter(coords[:, 0], coords[:, 1], c=actual_colors, s=12, alpha=0.65, linewidths=0)
    axes[0].set_title("User Feature Projection by True Label")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    pred_colors = np.full(graph["user"].num_nodes, "#94a3b8", dtype=object)
    pred_colors[test_idx] = np.where(y_pred == 1, "#dc2626", "#2563eb")
    if pred_colors.shape[0] > coords.shape[0]:
        pred_colors = pred_colors[:coords.shape[0]]
    axes[1].scatter(coords[:, 0], coords[:, 1], c=pred_colors, s=12, alpha=0.65, linewidths=0)
    axes[1].set_title("Test Predictions on User Projection")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_similarity_subgraph(processed_data, output_path, relation_name="similar_profile", max_nodes=24):
    edges = processed_data["edges"].copy()
    if edges.empty or "type" not in edges.columns:
        return None

    rel_edges = edges[edges["type"].astype(str) == relation_name].copy()
    if rel_edges.empty:
        return None

    degree = {}
    for _, row in rel_edges.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    top_nodes = [node for node, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:max_nodes]]
    top_set = set(top_nodes)
    rel_edges = rel_edges[rel_edges["src"].astype(str).isin(top_set) & rel_edges["dst"].astype(str).isin(top_set)]
    if rel_edges.empty:
        return None

    labels_df = processed_data["labels"].copy()
    labels_df["id"] = labels_df["id"].astype(str)
    label_lookup = dict(zip(labels_df["id"], labels_df["label"].astype(str)))

    angles = np.linspace(0, 2 * np.pi, num=len(top_nodes), endpoint=False)
    radius = 1.0
    pos = {node: (radius * np.cos(angle), radius * np.sin(angle)) for node, angle in zip(top_nodes, angles)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Sample Similar-Profile User Subgraph")
    ax.axis("off")

    for _, row in rel_edges.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", alpha=0.35, linewidth=1.0)

    for node in top_nodes:
        x, y = pos[node]
        label = label_lookup.get(node, "unknown").lower()
        color = "#dc2626" if "bot" in label else "#2563eb"
        ax.scatter([x], [y], s=110, color=color, edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(x, y + 0.08, node[-6:], ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
