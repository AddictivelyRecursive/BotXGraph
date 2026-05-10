import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_fast_bot_baseline import (
    build_graph as build_baseline_graph,
    ensure_dataset as ensure_baseline_dataset,
    load_config as load_baseline_config,
    set_runtime_limits as set_baseline_runtime_limits,
    train_baseline,
)
from train_hetero_gnn import (
    build_graph as build_hetero_graph,
    compute_feature_saliency,
    compute_metrics,
    compute_metrics_from_probabilities,
    ensure_dataset as ensure_hetero_dataset,
    load_config as load_hetero_config,
    predict_probabilities,
    set_augmentation_config,
    set_runtime_limits as set_hetero_runtime_limits,
    set_seed,
    train_model,
)
from src.data.edge_augmentor import EdgeAugmentor
from src.data.entity_extractor import EntityExtractor
from src.data.graph_builder import GraphBuilder
from src.data.loader import get_loader
from src.data.preprocessor import TwiBot22Preprocessor


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def metrics_row(prefix, metrics):
    return {
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_bot_precision": float(metrics["bot_precision"]),
        f"{prefix}_bot_recall": float(metrics["bot_recall"]),
    }


def summarize_metric_frame(df, metric_cols):
    rows = []
    for metric in metric_cols:
        rows.append({
            "metric": metric,
            "mean": float(df[metric].mean()),
            "std": float(df[metric].std(ddof=0)),
            "min": float(df[metric].min()),
            "max": float(df[metric].max()),
        })
    return pd.DataFrame(rows)


def humanize_metric(metric_name):
    mapping = {
        "test_accuracy": "Accuracy",
        "test_macro_f1": "Macro-F1",
        "test_bot_precision": "Bot Precision",
        "test_bot_recall": "Bot Recall",
        "val_accuracy": "Validation Accuracy",
        "val_macro_f1": "Validation Macro-F1",
        "val_bot_precision": "Validation Bot Precision",
        "val_bot_recall": "Validation Bot Recall",
    }
    return mapping.get(metric_name, metric_name.replace("_", " ").title())


def prettify_variant_name(name):
    return {
        "profile_only": "Profile Only",
        "profile_plus_text": "Profile + Text",
        "plus_similar_profile": "+ Similar Profile",
        "plus_co_hashtag": "+ Co-Hashtag",
        "plus_co_url": "+ Co-URL",
        "full": "Full Model",
        "full_graph": "Full Graph",
        "drop_similar_profile": "Drop Similar-Profile",
        "drop_hashtag_and_url_edges": "Drop Hashtag/URL Edges",
        "mask_top_user_features": "Mask Top User Features",
        "clean": "Clean",
        "light_rewrite": "Light Rewrite",
        "hashtag_injection": "Hashtag Injection",
        "url_injection": "URL Injection",
        "combined_attack": "Combined Attack",
    }.get(name, name.replace("_", " ").title())


def write_markdown(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def plot_grouped_metric_bars(df, category_col, metric_cols, output_path, title, rotate_labels=False):
    categories = [prettify_variant_name(v) for v in df[category_col].tolist()]
    x = np.arange(len(categories))
    width = 0.18 if len(metric_cols) >= 4 else 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(categories) * 1.35), 5.8))
    colors = ["#1d4ed8", "#b45309", "#0f766e", "#dc2626"]

    for idx, metric in enumerate(metric_cols):
        values = df[metric].astype(float).to_numpy()
        ax.bar(x + (idx - (len(metric_cols) - 1) / 2.0) * width, values, width=width, label=humanize_metric(metric), color=colors[idx % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20 if rotate_labels else 0, ha="right" if rotate_labels else "center")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_std_comparison(summary_frames, output_path, title):
    metrics = [row["metric"] for row in summary_frames[0].to_dict(orient="records")]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#64748b", "#1d4ed8"]
    labels = ["Baseline", "Hetero-GNN"]

    for idx, summary_df in enumerate(summary_frames):
        means = summary_df["mean"].astype(float).to_numpy()
        stds = summary_df["std"].astype(float).to_numpy()
        ax.bar(
            x + (idx - 0.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=colors[idx],
            label=labels[idx],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([humanize_metric(metric) for metric in metrics], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_probability_shift(df, category_col, output_path, title):
    categories = [prettify_variant_name(v) for v in df[category_col].tolist()]
    values = df["mean_abs_prob_shift"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.25), 5.2))
    ax.bar(categories, values, color="#7c3aed")
    ax.set_ylabel("Mean Absolute Probability Shift")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_markdown_table(df, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.4f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def empty_edge_index_like(edge_index):
    return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)


def zero_out_node_features(graph, node_type):
    if hasattr(graph[node_type], "x"):
        graph[node_type].x = torch.zeros_like(graph[node_type].x)


def zero_out_user_features(graph, feature_names):
    all_names = list(getattr(graph["user"], "feature_names", []))
    if not all_names:
        return
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}
    indices = [name_to_idx[name] for name in feature_names if name in name_to_idx]
    if not indices:
        return
    graph["user"].x[:, indices] = 0.0


def keep_only_relations(graph, allowed_relations):
    allowed_relations = set(allowed_relations)
    for edge_type in list(graph.edge_types):
        src, rel, dst = edge_type
        if src == "user" and dst == "user" and rel not in allowed_relations:
            graph[edge_type].edge_index = empty_edge_index_like(graph[edge_type].edge_index)


def drop_relations(graph, relation_names):
    relation_names = set(relation_names)
    for edge_type in list(graph.edge_types):
        _, rel, _ = edge_type
        if rel in relation_names:
            graph[edge_type].edge_index = empty_edge_index_like(graph[edge_type].edge_index)


def apply_ablation_to_graph(graph, variant):
    if variant == "full":
        return graph
    if variant == "profile_only":
        keep_only_relations(graph, set())
        drop_relations(graph, {"contains", "rev_contains", "links", "rev_links"})
        zero_out_node_features(graph, "tweet")
        zero_out_node_features(graph, "hashtag")
        zero_out_node_features(graph, "url")
        return graph
    if variant == "profile_plus_text":
        keep_only_relations(graph, set())
        drop_relations(graph, {"contains", "rev_contains", "links", "rev_links"})
        zero_out_node_features(graph, "hashtag")
        zero_out_node_features(graph, "url")
        return graph
    if variant == "plus_similar_profile":
        keep_only_relations(graph, {"similar_profile", "rev_similar_profile"})
        return graph
    if variant == "plus_co_hashtag":
        keep_only_relations(
            graph,
            {"similar_profile", "rev_similar_profile", "co_hashtag", "rev_co_hashtag"},
        )
        return graph
    if variant == "plus_co_url":
        keep_only_relations(
            graph,
            {
                "similar_profile",
                "rev_similar_profile",
                "co_hashtag",
                "rev_co_hashtag",
                "co_url",
                "rev_co_url",
            },
        )
        return graph
    raise ValueError(f"Unknown ablation variant: {variant}")


def perturb_text(text, mode):
    text = "" if text is None else str(text)
    if mode == "clean":
        return text
    if mode == "light_rewrite":
        replacements = {
            "official": "trusted",
            "news": "updates",
            "developer": "builder",
            "marketing": "growth",
            "crypto": "blockchain",
        }
        out = text
        for src, dst in replacements.items():
            out = out.replace(src, dst).replace(src.title(), dst.title())
        if out.strip():
            out = f"profile summary: {out}"
        return out
    if mode == "hashtag_injection":
        suffix = " #AI #Breaking #Crypto"
        return (text + suffix).strip()
    if mode == "url_injection":
        suffix = " https://t.co/update https://bit.ly/news"
        return (text + suffix).strip()
    if mode == "combined_attack":
        return perturb_text(perturb_text(perturb_text(text, "light_rewrite"), "hashtag_injection"), "url_injection")
    raise ValueError(f"Unknown perturbation mode: {mode}")


def build_processed_inputs(config, perturbation="clean"):
    loader = get_loader(config)
    raw_data = loader.load_all()

    preprocessor = TwiBot22Preprocessor(config)
    processed = preprocessor.process_all(raw_data)

    if perturbation != "clean":
        tweets_df = processed["tweets"].copy()
        tweets_df["text"] = tweets_df["text"].apply(lambda text: perturb_text(text, perturbation))
        tweets_df["entities"] = [{} for _ in range(len(tweets_df))]
        processed["tweets"] = tweets_df

    extractor = EntityExtractor(config)
    entities = extractor.extract_all(processed["tweets"])

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

    builder = GraphBuilder(config)
    graph = builder.build_graph(processed, entities)
    return graph, processed, entities


def build_hetero_config_from_args(args):
    config = load_hetero_config(args.base_config, args.dataset_config)
    config = ensure_hetero_dataset(config, args.csv, args.download)
    config = set_hetero_runtime_limits(config, args.limit)
    config = set_augmentation_config(config, args)
    return config


def build_baseline_config_from_args(args):
    config = load_baseline_config(args.base_config, args.dataset_config)
    config = ensure_baseline_dataset(config, args.csv, args.download)
    config = set_baseline_runtime_limits(config, args.limit)
    return config


def train_hetero_once(args, seed, variant="full", perturbation="clean", output_dir=None):
    set_seed(seed)
    config = build_hetero_config_from_args(args)
    if perturbation == "clean":
        graph, processed, entities = build_hetero_graph(config)
    else:
        graph, processed, entities = build_processed_inputs(config, perturbation=perturbation)

    graph = copy.deepcopy(graph)
    apply_ablation_to_graph(graph, variant)

    train_args = copy.deepcopy(args)
    if output_dir is not None:
        output_dir = ensure_dir(output_dir)
        train_args.output = str(output_dir / "report.md")
        train_args.plots_dir = str(output_dir / "plots")
        train_args.artifacts_dir = str(output_dir / "artifacts")
    train_args.skip_plots = True
    train_args.save_epoch_artifacts = False
    train_args.save_checkpoints = False
    train_args.save_embeddings = False
    train_args.random_state = seed

    results = train_model(graph, train_args)
    results["processed"] = processed
    results["entities"] = entities
    results["graph"] = graph
    return results


def build_eval_graph(args, variant="full", perturbation="clean"):
    config = build_hetero_config_from_args(args)
    if perturbation == "clean":
        graph, _, _ = build_hetero_graph(config)
    else:
        graph, _, _ = build_processed_inputs(config, perturbation=perturbation)
    graph = copy.deepcopy(graph)
    apply_ablation_to_graph(graph, variant)
    return graph


def run_multiseed(args):
    output_dir = ensure_dir(Path(args.paper_output_dir) / "multiseed")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    baseline_rows = []
    hetero_rows = []

    for seed in seeds:
        set_seed(seed)

        baseline_config = build_baseline_config_from_args(args)
        baseline_graph, _, _ = build_baseline_graph(baseline_config)
        baseline_results = train_baseline(baseline_graph, random_state=seed)
        baseline_rows.append({
            "seed": seed,
            "model": "baseline",
            "test_accuracy": float(baseline_results["test_accuracy"]),
            "test_macro_f1": float(baseline_results["test_macro_f1"]),
            "test_bot_precision": float(baseline_results["test_bot_precision"]),
            "test_bot_recall": float(baseline_results["test_bot_recall"]),
        })

        hetero_results = train_hetero_once(
            args,
            seed=seed,
            variant="full",
            output_dir=output_dir / f"hetero_seed_{seed}",
        )
        hetero_rows.append({
            "seed": seed,
            "model": "hetero_gnn",
            "test_accuracy": float(hetero_results["test_metrics"]["accuracy"]),
            "test_macro_f1": float(hetero_results["test_metrics"]["macro_f1"]),
            "test_bot_precision": float(hetero_results["test_metrics"]["bot_precision"]),
            "test_bot_recall": float(hetero_results["test_metrics"]["bot_recall"]),
        })

    baseline_df = pd.DataFrame(baseline_rows)
    hetero_df = pd.DataFrame(hetero_rows)
    all_df = pd.concat([baseline_df, hetero_df], ignore_index=True)
    metric_cols = ["test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"]

    baseline_summary = summarize_metric_frame(baseline_df, metric_cols)
    hetero_summary = summarize_metric_frame(hetero_df, metric_cols)

    all_df.to_csv(output_dir / "per_seed_results.csv", index=False)
    baseline_summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    hetero_summary.to_csv(output_dir / "hetero_summary.csv", index=False)
    plot_mean_std_comparison(
        [baseline_summary, hetero_summary],
        output_dir / "multiseed_mean_std_comparison.png",
        "Multi-Seed Performance: Baseline vs Hetero-GNN",
    )
    save_json(
        output_dir / "summary.json",
        {
            "seeds": seeds,
            "baseline": baseline_summary.to_dict(orient="records"),
            "hetero_gnn": hetero_summary.to_dict(orient="records"),
        },
    )
    summary_md = f"""
# Multi-Seed Summary

Seeds: {", ".join(str(seed) for seed in seeds)}

## Baseline
{build_markdown_table(baseline_summary, ["metric", "mean", "std", "min", "max"])}

## Hetero-GNN
{build_markdown_table(hetero_summary, ["metric", "mean", "std", "min", "max"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)
    print(f"Saved multi-seed outputs to {output_dir}")


def run_ablations(args):
    output_dir = ensure_dir(Path(args.paper_output_dir) / "ablations")
    variants = [
        "profile_only",
        "profile_plus_text",
        "plus_similar_profile",
        "plus_co_hashtag",
        "plus_co_url",
        "full",
    ]
    rows = []

    for variant in variants:
        results = train_hetero_once(
            args,
            seed=args.random_state,
            variant=variant,
            output_dir=output_dir / variant,
        )
        row = {
            "variant": variant,
            "best_epoch": int(results["best_epoch"]),
            **metrics_row("val", results["val_metrics"]),
            **metrics_row("test", results["test_metrics"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ablation_results.csv", index=False)
    save_json(output_dir / "ablation_results.json", rows)
    plot_grouped_metric_bars(
        df,
        category_col="variant",
        metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
        output_path=output_dir / "ablation_test_metrics.png",
        title="Ablation Study: Test Metrics",
        rotate_labels=True,
    )
    plot_grouped_metric_bars(
        df,
        category_col="variant",
        metric_cols=["val_macro_f1", "test_macro_f1"],
        output_path=output_dir / "ablation_val_vs_test_macro_f1.png",
        title="Ablation Study: Validation vs Test Macro-F1",
        rotate_labels=True,
    )
    summary_md = f"""
# Ablation Summary

{build_markdown_table(df, ["variant", "best_epoch", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)
    print(f"Saved ablation outputs to {output_dir}")


def evaluate_perturbed_graph(model, graph, mask):
    y_true, bot_probs, indices = predict_probabilities(model, graph, mask)
    metrics = compute_metrics_from_probabilities(y_true, bot_probs, threshold=0.5)
    return y_true, bot_probs, indices, metrics


def run_faithfulness(args):
    output_dir = ensure_dir(Path(args.paper_output_dir) / "faithfulness")
    results = train_hetero_once(
        args,
        seed=args.random_state,
        variant="full",
        output_dir=output_dir / "full_model",
    )

    model = results["model"]
    test_mask = results["test_mask"]
    base_graph = copy.deepcopy(results["graph"])
    base_y, base_probs, _, base_metrics = evaluate_perturbed_graph(model, base_graph, test_mask)

    top_k = int(args.faithfulness_top_k)
    feature_names = list(base_graph["user"].feature_names)
    saliency = compute_feature_saliency(model, base_graph, test_mask)
    top_features = [name for name, _ in sorted(zip(feature_names, saliency), key=lambda x: x[1], reverse=True)[:top_k]]

    perturbations = {
        "full_graph": lambda g: g,
        "drop_similar_profile": lambda g: drop_relations(g, {"similar_profile", "rev_similar_profile"}),
        "drop_hashtag_and_url_edges": lambda g: drop_relations(
            g,
            {"co_hashtag", "rev_co_hashtag", "co_url", "rev_co_url", "contains", "rev_contains", "links", "rev_links"},
        ),
        "mask_top_user_features": lambda g: zero_out_user_features(g, top_features),
    }

    rows = []
    for name, fn in perturbations.items():
        graph = copy.deepcopy(base_graph)
        fn(graph)
        y_true, bot_probs, _, metrics = evaluate_perturbed_graph(model, graph, test_mask)
        rows.append({
            "perturbation": name,
            "top_features_masked": ",".join(top_features) if name == "mask_top_user_features" else "",
            **metrics_row("test", metrics),
            "mean_abs_prob_shift": float(np.mean(np.abs(bot_probs - base_probs))),
            "mean_bot_prob": float(np.mean(bot_probs)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "faithfulness_results.csv", index=False)
    plot_grouped_metric_bars(
        df,
        category_col="perturbation",
        metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
        output_path=output_dir / "faithfulness_metric_drop.png",
        title="Faithfulness Check: Metrics After Evidence Removal",
        rotate_labels=True,
    )
    plot_probability_shift(
        df,
        category_col="perturbation",
        output_path=output_dir / "faithfulness_probability_shift.png",
        title="Faithfulness Check: Prediction Shift After Evidence Removal",
    )
    save_json(
        output_dir / "faithfulness_results.json",
        {
            "top_features": top_features,
            "base_metrics": metrics_to_serializable(base_metrics),
            "rows": rows,
        },
    )
    summary_md = f"""
# Faithfulness Summary

Top masked features: {", ".join(top_features)}

{build_markdown_table(df, ["perturbation", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall", "mean_abs_prob_shift"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)
    print(f"Saved faithfulness outputs to {output_dir}")


def metrics_to_serializable(metrics):
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    return payload


def threshold_table(y_true, bot_probabilities, thresholds):
    rows = []
    for threshold in thresholds:
        metrics = compute_metrics_from_probabilities(y_true, bot_probabilities, threshold=threshold)
        rows.append({
            "threshold": float(threshold),
            **metrics_row("metric", metrics),
        })
    return pd.DataFrame(rows)


def plot_threshold_curves(df, output_path, title):
    x = df["threshold"].to_numpy(dtype=float)
    macro_f1 = df["metric_macro_f1"].to_numpy(dtype=float)
    bot_precision = df["metric_bot_precision"].to_numpy(dtype=float)
    bot_recall = df["metric_bot_recall"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, macro_f1, label="Macro-F1", color="#1d4ed8", linewidth=2)
    ax.plot(x, bot_precision, label="Bot Precision", color="#b45309", linewidth=2)
    ax.plot(x, bot_recall, label="Bot Recall", color="#0f766e", linewidth=2)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_threshold(args):
    output_dir = ensure_dir(Path(args.paper_output_dir) / "threshold")
    results = train_hetero_once(
        args,
        seed=args.random_state,
        variant="full",
        output_dir=output_dir / "full_model",
    )
    model = results["model"]
    graph = results["graph"]

    val_y, val_probs, _ = predict_probabilities(model, graph, results["val_mask"])
    test_y, test_probs, _ = predict_probabilities(model, graph, results["test_mask"])
    thresholds = np.linspace(0.05, 0.95, 19)

    val_df = threshold_table(val_y, val_probs, thresholds)
    test_df = threshold_table(test_y, test_probs, thresholds)
    joined = val_df.merge(test_df, on="threshold", suffixes=("_val", "_test"))

    best_idx = int(val_df["metric_macro_f1"].idxmax())
    best_threshold = float(val_df.loc[best_idx, "threshold"])
    tuned_test_metrics = compute_metrics_from_probabilities(test_y, test_probs, threshold=best_threshold)
    default_test_metrics = compute_metrics_from_probabilities(test_y, test_probs, threshold=0.5)

    joined.to_csv(output_dir / "threshold_sweep.csv", index=False)
    save_json(
        output_dir / "threshold_summary.json",
        {
            "best_threshold_by_val_macro_f1": best_threshold,
            "default_test_metrics": metrics_to_serializable(default_test_metrics),
            "tuned_test_metrics": metrics_to_serializable(tuned_test_metrics),
        },
    )
    plot_threshold_curves(val_df, output_dir / "val_threshold_curves.png", "Validation Threshold Sweep")
    plot_threshold_curves(test_df, output_dir / "test_threshold_curves.png", "Test Threshold Sweep")
    summary_md = f"""
# Threshold Sweep Summary

Best threshold by validation Macro-F1: {best_threshold:.2f}

## Default Threshold (0.50)

{build_markdown_table(pd.DataFrame([metrics_row("test", default_test_metrics)]), ["test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}

## Tuned Threshold

{build_markdown_table(pd.DataFrame([metrics_row("test", tuned_test_metrics)]), ["test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)
    print(f"Saved threshold outputs to {output_dir}")


def run_robustness(args):
    output_dir = ensure_dir(Path(args.paper_output_dir) / "robustness")
    clean_results = train_hetero_once(
        args,
        seed=args.random_state,
        variant="full",
        perturbation="clean",
        output_dir=output_dir / "clean_model",
    )

    model = clean_results["model"]
    clean_graph = clean_results["graph"]
    test_mask = clean_results["test_mask"]
    _, clean_probs, _, clean_metrics = evaluate_perturbed_graph(model, clean_graph, test_mask)

    rows = [{
        "perturbation": "clean",
        **metrics_row("test", clean_metrics),
        "mean_abs_prob_shift": 0.0,
    }]

    for perturbation in ["light_rewrite", "hashtag_injection", "url_injection", "combined_attack"]:
        perturbed_graph = build_eval_graph(args, variant="full", perturbation=perturbation)
        _, perturbed_probs, _, metrics = evaluate_perturbed_graph(model, perturbed_graph, test_mask)
        rows.append({
            "perturbation": perturbation,
            **metrics_row("test", metrics),
            "mean_abs_prob_shift": float(np.mean(np.abs(perturbed_probs - clean_probs))),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "robustness_results.csv", index=False)
    save_json(output_dir / "robustness_results.json", rows)
    plot_grouped_metric_bars(
        df,
        category_col="perturbation",
        metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
        output_path=output_dir / "robustness_metrics.png",
        title="Robustness to Profile-Text Perturbations",
        rotate_labels=True,
    )
    plot_probability_shift(
        df,
        category_col="perturbation",
        output_path=output_dir / "robustness_probability_shift.png",
        title="Robustness: Prediction Shift Under Text Perturbations",
    )
    summary_md = f"""
# Robustness Summary

{build_markdown_table(df, ["perturbation", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall", "mean_abs_prob_shift"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)
    print(f"Saved robustness outputs to {output_dir}")


def run_cross_dataset(args):
    output_dir = ensure_dir(Path(args.paper_output_dir) / "cross_dataset")
    results = train_hetero_once(
        args,
        seed=args.random_state,
        variant="full",
        output_dir=output_dir / Path(args.dataset_config).stem,
    )
    payload = {
        "dataset_config": args.dataset_config,
        "best_epoch": int(results["best_epoch"]),
        "val_metrics": metrics_to_serializable(results["val_metrics"]),
        "test_metrics": metrics_to_serializable(results["test_metrics"]),
    }
    save_json(output_dir / f"{Path(args.dataset_config).stem}_summary.json", payload)
    summary_df = pd.DataFrame([{
        "dataset": Path(args.dataset_config).stem,
        "best_epoch": int(results["best_epoch"]),
        "test_accuracy": float(results["test_metrics"]["accuracy"]),
        "test_macro_f1": float(results["test_metrics"]["macro_f1"]),
        "test_bot_precision": float(results["test_metrics"]["bot_precision"]),
        "test_bot_recall": float(results["test_metrics"]["bot_recall"]),
    }])
    plot_grouped_metric_bars(
        summary_df,
        category_col="dataset",
        metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
        output_path=output_dir / f"{Path(args.dataset_config).stem}_metrics.png",
        title=f"Cross-Dataset Check: {Path(args.dataset_config).stem}",
        rotate_labels=False,
    )
    summary_md = f"""
# Cross-Dataset Summary

Dataset config: `{args.dataset_config}`

{build_markdown_table(summary_df, ["dataset", "best_epoch", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}
"""
    write_markdown(output_dir / f"{Path(args.dataset_config).stem}_summary.md", summary_md)
    print(f"Saved cross-dataset outputs to {output_dir}")


def postprocess_multiseed(output_dir):
    output_dir = Path(output_dir)
    baseline_summary = pd.read_csv(output_dir / "baseline_summary.csv")
    hetero_summary = pd.read_csv(output_dir / "hetero_summary.csv")
    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    plot_mean_std_comparison(
        [baseline_summary, hetero_summary],
        output_dir / "multiseed_mean_std_comparison.png",
        "Multi-Seed Performance: Baseline vs Hetero-GNN",
    )
    summary_md = f"""
# Multi-Seed Summary

Seeds: {", ".join(str(seed) for seed in summary_json.get("seeds", []))}

## Baseline
{build_markdown_table(baseline_summary, ["metric", "mean", "std", "min", "max"])}

## Hetero-GNN
{build_markdown_table(hetero_summary, ["metric", "mean", "std", "min", "max"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)


def postprocess_ablations(output_dir):
    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir / "ablation_results.csv")
    plot_grouped_metric_bars(
        df,
        category_col="variant",
        metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
        output_path=output_dir / "ablation_test_metrics.png",
        title="Ablation Study: Test Metrics",
        rotate_labels=True,
    )
    plot_grouped_metric_bars(
        df,
        category_col="variant",
        metric_cols=["val_macro_f1", "test_macro_f1"],
        output_path=output_dir / "ablation_val_vs_test_macro_f1.png",
        title="Ablation Study: Validation vs Test Macro-F1",
        rotate_labels=True,
    )
    summary_md = f"""
# Ablation Summary

{build_markdown_table(df, ["variant", "best_epoch", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)


def postprocess_faithfulness(output_dir):
    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir / "faithfulness_results.csv")
    payload = json.loads((output_dir / "faithfulness_results.json").read_text(encoding="utf-8"))
    top_features = payload.get("top_features", [])
    plot_grouped_metric_bars(
        df,
        category_col="perturbation",
        metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
        output_path=output_dir / "faithfulness_metric_drop.png",
        title="Faithfulness Check: Metrics After Evidence Removal",
        rotate_labels=True,
    )
    plot_probability_shift(
        df,
        category_col="perturbation",
        output_path=output_dir / "faithfulness_probability_shift.png",
        title="Faithfulness Check: Prediction Shift After Evidence Removal",
    )
    summary_md = f"""
# Faithfulness Summary

Top masked features: {", ".join(top_features)}

{build_markdown_table(df, ["perturbation", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall", "mean_abs_prob_shift"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)


def postprocess_threshold(output_dir):
    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir / "threshold_sweep.csv")
    val_df = df[[c for c in df.columns if c == "threshold" or c.endswith("_val")]].copy()
    val_df.columns = [c.replace("_val", "") for c in val_df.columns]
    test_df = df[[c for c in df.columns if c == "threshold" or c.endswith("_test")]].copy()
    test_df.columns = [c.replace("_test", "") for c in test_df.columns]
    payload = json.loads((output_dir / "threshold_summary.json").read_text(encoding="utf-8"))
    plot_threshold_curves(val_df, output_dir / "val_threshold_curves.png", "Validation Threshold Sweep")
    plot_threshold_curves(test_df, output_dir / "test_threshold_curves.png", "Test Threshold Sweep")
    summary_md = f"""
# Threshold Sweep Summary

Best threshold by validation Macro-F1: {payload["best_threshold_by_val_macro_f1"]:.2f}

## Default Threshold (0.50)

{build_markdown_table(pd.DataFrame([metrics_row("test", payload["default_test_metrics"])]), ["test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}

## Tuned Threshold

{build_markdown_table(pd.DataFrame([metrics_row("test", payload["tuned_test_metrics"])]), ["test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)


def postprocess_robustness(output_dir):
    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir / "robustness_results.csv")
    plot_grouped_metric_bars(
        df,
        category_col="perturbation",
        metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
        output_path=output_dir / "robustness_metrics.png",
        title="Robustness to Profile-Text Perturbations",
        rotate_labels=True,
    )
    plot_probability_shift(
        df,
        category_col="perturbation",
        output_path=output_dir / "robustness_probability_shift.png",
        title="Robustness: Prediction Shift Under Text Perturbations",
    )
    summary_md = f"""
# Robustness Summary

{build_markdown_table(df, ["perturbation", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall", "mean_abs_prob_shift"])}
"""
    write_markdown(output_dir / "summary.md", summary_md)


def postprocess_cross_dataset(output_dir):
    output_dir = Path(output_dir)
    for json_path in output_dir.glob("*_summary.json"):
        if json_path.name == "summary.json":
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        dataset = json_path.stem.replace("_summary", "")
        summary_df = pd.DataFrame([{
            "dataset": dataset,
            "best_epoch": int(payload["best_epoch"]),
            "test_accuracy": float(payload["test_metrics"]["accuracy"]),
            "test_macro_f1": float(payload["test_metrics"]["macro_f1"]),
            "test_bot_precision": float(payload["test_metrics"]["bot_precision"]),
            "test_bot_recall": float(payload["test_metrics"]["bot_recall"]),
        }])
        plot_grouped_metric_bars(
            summary_df,
            category_col="dataset",
            metric_cols=["test_macro_f1", "test_bot_precision", "test_bot_recall"],
            output_path=output_dir / f"{dataset}_metrics.png",
            title=f"Cross-Dataset Check: {dataset}",
            rotate_labels=False,
        )
        summary_md = f"""
# Cross-Dataset Summary

Dataset config summary: `{json_path.name}`

{build_markdown_table(summary_df, ["dataset", "best_epoch", "test_accuracy", "test_macro_f1", "test_bot_precision", "test_bot_recall"])}
"""
        write_markdown(output_dir / f"{dataset}_summary.md", summary_md)


def run_postprocess(args):
    command_map = {
        "multiseed": postprocess_multiseed,
        "ablations": postprocess_ablations,
        "faithfulness": postprocess_faithfulness,
        "threshold": postprocess_threshold,
        "robustness": postprocess_robustness,
        "cross_dataset": postprocess_cross_dataset,
    }
    fn = command_map[args.experiment]
    fn(args.output_dir)
    print(f"Postprocessed {args.experiment} outputs in {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper-ready BotXGraph experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser):
        subparser.add_argument("--base-config", default="config/base.yaml")
        subparser.add_argument("--dataset-config", default="config/twitter_human_bots.yaml")
        subparser.add_argument("--csv", help="Path to twitter_human_bots_dataset.csv")
        subparser.add_argument("--download", action="store_true")
        subparser.add_argument("--limit", type=int)
        subparser.add_argument("--paper-output-dir", default="reports/paper_experiments")
        subparser.add_argument("--epochs", type=int, default=120)
        subparser.add_argument("--patience", type=int, default=20)
        subparser.add_argument("--eval-every", type=int, default=2)
        subparser.add_argument("--hidden-dim", type=int, default=64)
        subparser.add_argument("--dropout", type=float, default=0.3)
        subparser.add_argument("--lr", type=float, default=1e-3)
        subparser.add_argument("--weight-decay", type=float, default=1e-4)
        subparser.add_argument("--random-state", type=int, default=42)
        subparser.add_argument("--num-heads", type=int, default=4)
        subparser.add_argument("--num-global-tokens", type=int, default=8)
        subparser.add_argument("--knn-k", type=int, default=20)
        subparser.add_argument("--min-similarity", type=float, default=0.5)
        subparser.add_argument("--min-shared-hashtags", type=int, default=2)
        subparser.add_argument("--min-shared-urls", type=int, default=1)
        subparser.add_argument("--max-users-per-token", type=int, default=200)
        subparser.add_argument("--output", default="reports/hetero_gnn_results.md")
        subparser.add_argument("--plots-dir", default="reports/plots/hetero_gnn")
        subparser.add_argument("--artifacts-dir", default="reports/artifacts/hetero_gnn")
        subparser.add_argument("--skip-plots", action="store_true")
        subparser.add_argument("--save-epoch-artifacts", action="store_true", default=True)
        subparser.add_argument("--no-save-epoch-artifacts", dest="save_epoch_artifacts", action="store_false")
        subparser.add_argument("--save-checkpoints", action="store_true", default=True)
        subparser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
        subparser.add_argument("--save-embeddings", action="store_true", default=True)
        subparser.add_argument("--no-save-embeddings", dest="save_embeddings", action="store_false")

    multiseed = subparsers.add_parser("multiseed")
    add_shared(multiseed)
    multiseed.add_argument("--seeds", default="42,43,44,45,46")

    ablations = subparsers.add_parser("ablations")
    add_shared(ablations)

    faithfulness = subparsers.add_parser("faithfulness")
    add_shared(faithfulness)
    faithfulness.add_argument("--faithfulness-top-k", type=int, default=3)

    threshold = subparsers.add_parser("threshold")
    add_shared(threshold)

    robustness = subparsers.add_parser("robustness")
    add_shared(robustness)

    cross_dataset = subparsers.add_parser("cross-dataset")
    add_shared(cross_dataset)

    postprocess = subparsers.add_parser("postprocess")
    postprocess.add_argument(
        "--experiment",
        required=True,
        choices=["multiseed", "ablations", "faithfulness", "threshold", "robustness", "cross_dataset"],
    )
    postprocess.add_argument("--output-dir", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "multiseed":
        run_multiseed(args)
    elif args.command == "ablations":
        run_ablations(args)
    elif args.command == "faithfulness":
        run_faithfulness(args)
    elif args.command == "threshold":
        run_threshold(args)
    elif args.command == "robustness":
        run_robustness(args)
    elif args.command == "cross-dataset":
        run_cross_dataset(args)
    elif args.command == "postprocess":
        run_postprocess(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
