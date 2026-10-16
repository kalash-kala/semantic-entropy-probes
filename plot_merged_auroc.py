#!/usr/bin/env python3
"""Parse log file and create merged plots for layer-wise AUROC across datasets."""

import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))


def parse_log_file(log_path):
    """
    Parse the log file and extract layer-wise AUROC values.

    Returns:
        dict: {(token_pos, metric, dataset): {layer: test_auroc}}
    """
    data = defaultdict(lambda: defaultdict(float))

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_token_pos = None
    current_metric = None
    current_dataset = None
    current_layer = None

    for i, line in enumerate(lines):
        # Extract token position and metric
        match = re.search(r'\[(\w+)\] probe target: (\S+)', line)
        if match:
            current_token_pos = match.group(1)
            current_metric = match.group(2)
            continue

        # Extract dataset
        match = re.search(r'\s+(sciq|svamp|trivia_qa)\s*$', line)
        if match:
            current_dataset = match.group(1)
            continue

        # Extract layer number
        match = re.search(r'layer (\d+)/\d+', line)
        if match:
            current_layer = int(match.group(1))
            # Look at next line for test_auroc
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                auroc_match = re.search(r'test_auroc=([\d.]+)', next_line)
                if auroc_match and current_token_pos and current_metric and current_dataset:
                    test_auroc = float(auroc_match.group(1))
                    key = (current_token_pos, current_metric, current_dataset)
                    data[key][current_layer] = test_auroc

    return data


def plot_merged_auroc(data, output_dir=None):
    """Create merged plots for each token position and metric combination."""
    if output_dir is None:
        output_dir = os.path.join(REPO, "semantic_entropy_probes", "probe_results", "figures", "qwen")

    # Group by token position and metric
    plot_configs = {}
    for (token_pos, metric, dataset), layers_data in data.items():
        key = (token_pos, metric)
        if key not in plot_configs:
            plot_configs[key] = {}
        plot_configs[key][dataset] = layers_data

    datasets = ['sciq', 'svamp', 'trivia_qa']
    colors = {'sciq': '#1f77b4', 'svamp': '#ff7f0e', 'trivia_qa': '#2ca02c'}
    markers = {'sciq': 'o', 'svamp': 's', 'trivia_qa': '^'}

    # Create plots
    for (token_pos, metric), dataset_data in sorted(plot_configs.items()):
        fig, ax = plt.subplots(figsize=(10, 6))

        for dataset in datasets:
            if dataset in dataset_data:
                layers_data = dataset_data[dataset]
                layers = sorted(layers_data.keys())
                aurocs = [layers_data[l] for l in layers]

                ax.plot(layers, aurocs,
                       color=colors[dataset],
                       marker=markers[dataset],
                       label=dataset,
                       linewidth=2,
                       markersize=6,
                       alpha=0.8)

        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test AUROC', fontsize=12, fontweight='bold')
        ax.set_title(f'Layer-wise Test AUROC\n{token_pos.capitalize()} | {metric}',
                    fontsize=13, fontweight='bold')
        ax.legend(title='Dataset', fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])

        # Save figure
        filename = f"{output_dir}/merged_auroc__{token_pos}__{metric}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


if __name__ == "__main__":
    log_path = os.environ.get(
        "SEP_LOG",
        os.path.join(REPO, "semantic_entropy_probes", "probe_run_qwen.log"),
    )

    print("Parsing log file...")
    data = parse_log_file(log_path)

    print(f"Found {len(data)} configurations")
    print("\nGenerating plots...")
    plot_merged_auroc(data)
    print("\nDone!")
