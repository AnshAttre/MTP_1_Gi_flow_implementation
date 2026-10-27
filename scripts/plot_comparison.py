"""Plot dataset-specific metrics from matched model runs and saved baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out", default="paper/figures/reconstruction_comparison.png")
    return parser.parse_args(argv)


def _read_run(root: Path):
    summary = json.loads((root / "summary.json").read_text())
    args = summary["args"]
    histories = [
        json.loads(path.read_text())
        for path in sorted(root.glob("seed*/history.json"))
    ]
    if not histories:
        raise FileNotFoundError("no seed*/history.json files under %s" % root)
    return args, histories


def _source_name(name: str) -> str:
    return re.sub(r"\[\d+\]$", "", name)


def main(argv=None):
    args = parse_args(argv)
    run_dirs = [Path(value) for value in args.run_dirs]
    runs = [_read_run(root) for root in run_dirs]
    reference_args = runs[0][0]
    for run_args, _ in runs[1:]:
        for key in ("missing", "rho", "split_mode", "split_seed"):
            if run_args.get(key) != reference_args.get(key):
                raise ValueError("model runs differ in comparison setting %r" % key)

    model_entries = []
    for run_args, histories in runs:
        source_label = "+".join(run_args.get("datasets") or [run_args["dataset"]])
        label = "%s (%s)" % (run_args["variant"], source_label)
        per_dataset = {}
        for history in histories:
            test_metrics = history["test"]
            dataset_metrics = test_metrics.get("per_dataset")
            if dataset_metrics is None:
                dataset_names = history.get("dataset_names", [run_args["dataset"]])
                dataset_metrics = {dataset_names[0]: test_metrics}
            for dataset_name, metrics in dataset_metrics.items():
                source = _source_name(dataset_name)
                per_dataset.setdefault(source, {}).setdefault("nmse", []).append(
                    metrics["nmse"]
                )
                per_dataset[source].setdefault("pcc", []).append(metrics["pcc"])
        model_entries.append((label, per_dataset))

    baseline_path = run_dirs[0] / "baselines.json"
    baselines = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    dataset_names = list(dict.fromkeys(
        source
        for _, result in model_entries
        for source in result
    ))
    normalized_baselines = {}
    for name, metrics in baselines.items():
        per_dataset = metrics.get("per_dataset")
        if per_dataset is None:
            source_names = runs[0][1][0].get(
                "dataset_names", [reference_args["dataset"]]
            )
            per_dataset = {source_names[0]: metrics}
        normalized_baselines[name] = {
            _source_name(source): values
            for source, values in per_dataset.items()
        }
    labels = [label for label, _ in model_entries] + list(normalized_baselines)

    metrics_to_plot = (("nmse", "NMSE (lower is better)"), ("pcc", "PCC (higher is better)"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(dataset_names))
    width = 0.8 / max(len(labels), 1)
    for metric_index, (metric, title) in enumerate(metrics_to_plot):
        axis = axes[metric_index]
        entries = model_entries + [(name, None) for name in normalized_baselines]
        for index, (label, root_result) in enumerate(entries):
            offset = (index - (len(labels) - 1) / 2) * width
            if root_result is not None:
                means = [
                    (
                        float(np.mean(root_result[name][metric]))
                        if name in root_result else float("nan")
                    )
                    for name in dataset_names
                ]
                errors = [
                    (
                        float(np.std(root_result[name][metric]))
                        if name in root_result else float("nan")
                    )
                    for name in dataset_names
                ]
                axis.bar(x + offset, means, width, yerr=errors, capsize=3, label=label)
            else:
                per_dataset = normalized_baselines[label]
                values = [
                    per_dataset[name][metric] if name in per_dataset else float("nan")
                    for name in dataset_names
                ]
                axis.bar(x + offset, values, width, label=label)
        axis.set_title(title)
        axis.set_xticks(x, dataset_names, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize="small")

    seed_counts = {len(histories) for _, histories in runs}
    run_summary = (
        "single seed"
        if seed_counts == {1}
        else "mean across %d seeds" % max(seed_counts)
    )
    figure.suptitle(
        "%s missingness, %s split; %s"
        % (reference_args["missing"], reference_args["split_mode"], run_summary)
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print("saved %s" % output)


if __name__ == "__main__":
    main()
