"""
Visualization script for person-space mapping results.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from preference_items import PREFERENCE_ITEMS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


def load_results():
    """Load all results from the results directory."""
    with open("results/analyses.json") as f:
        analyses = json.load(f)
    return analyses


def plot_scree(analyses, output_dir="results/plots"):
    """Plot scree plots showing variance explained by each component."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model_name, analysis) in enumerate(analyses.items()):
        if model_name == "cross_model_comparison":
            continue

        var_exp = analysis["profile_analysis"]["variance_explained"][:20]
        cumvar = analysis["profile_analysis"]["cumulative_variance"][:20]

        ax = axes[min(idx, 1)]
        x = range(1, len(var_exp) + 1)
        ax.bar(x, var_exp, alpha=0.6, label="Individual")
        ax.plot(x, cumvar, 'ro-', label="Cumulative")
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label="80% threshold")
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label="90% threshold")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance Explained")
        ax.set_title(f"Scree Plot: {model_name}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/scree_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved scree_plots.png")


def plot_congruence_distribution(output_dir="results/plots"):
    """Plot distribution of congruence scores across profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    import glob
    result_files = sorted(glob.glob("results/*_results.json"))

    for idx, filepath in enumerate(result_files):
        model_name = filepath.split("/")[-1].replace("_results.json", "")
        with open(filepath) as f:
            results = json.load(f)

        behavioral = [r["behavioral_congruence"] for r in results]
        coherence = [r["coherence_rating"] for r in results]

        ax = axes[min(idx, 1)]
        ax.hist(behavioral, bins=20, alpha=0.5, label="Behavioral", color="steelblue")
        ax.hist(coherence, bins=20, alpha=0.5, label="Coherence", color="coral")
        ax.set_xlabel("Congruence Score")
        ax.set_ylabel("Count")
        ax.set_title(f"Congruence Distribution: {model_name}")
        ax.legend()
        ax.axvline(np.mean(behavioral), color="steelblue", linestyle="--", alpha=0.7)
        ax.axvline(np.mean(coherence), color="coral", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/congruence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved congruence_distribution.png")


def plot_cooccurrence_heatmap(output_dir="results/plots"):
    """Plot heatmap of preference co-occurrence congruence."""
    import glob
    matrix_files = sorted(glob.glob("results/*_cooccur_matrix.npy"))

    for filepath in matrix_files:
        model_name = filepath.split("/")[-1].replace("_cooccur_matrix.npy", "")
        matrix = np.load(filepath)

        # Get domain labels
        domains = [PREFERENCE_ITEMS[i]["domain"] for i in range(60)]
        domain_order = sorted(range(60), key=lambda i: domains[i])
        matrix_sorted = matrix[np.ix_(domain_order, domain_order)]
        domain_labels = [domains[i] for i in domain_order]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(matrix_sorted, cmap="RdYlBu_r", center=0.5,
                    ax=ax, xticklabels=False, yticklabels=False)
        ax.set_title(f"Preference Co-occurrence Congruence: {model_name}")

        # Add domain boundaries
        unique_domains = []
        boundaries = [0]
        for i, d in enumerate(domain_labels):
            if d not in unique_domains:
                unique_domains.append(d)
            if i > 0 and d != domain_labels[i-1]:
                boundaries.append(i)
        boundaries.append(60)

        for b in boundaries[1:-1]:
            ax.axhline(b, color='black', linewidth=0.5)
            ax.axvline(b, color='black', linewidth=0.5)

        # Add domain labels on the side
        for i in range(len(boundaries)-1):
            mid = (boundaries[i] + boundaries[i+1]) / 2
            ax.text(-2, mid, unique_domains[i], ha='right', va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/cooccurrence_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved cooccurrence_{model_name}.png")


def plot_cross_model_comparison(output_dir="results/plots"):
    """Plot cross-model congruence score comparison."""
    import glob
    result_files = sorted(glob.glob("results/*_results.json"))

    if len(result_files) < 2:
        print("Need at least 2 models for cross-model comparison plot")
        return

    model_scores = {}
    for filepath in result_files:
        model_name = filepath.split("/")[-1].replace("_results.json", "")
        with open(filepath) as f:
            results = json.load(f)
        scores = [(r["behavioral_congruence"] + r["coherence_rating"]) / 2 for r in results]
        model_scores[model_name] = scores

    names = list(model_scores.keys())
    scores_a = model_scores[names[0]]
    scores_b = model_scores[names[1]]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(scores_a, scores_b, alpha=0.5, s=30)
    ax.set_xlabel(f"Congruence ({names[0]})")
    ax.set_ylabel(f"Congruence ({names[1]})")
    r, p = pearsonr(scores_a, scores_b)
    ax.set_title(f"Cross-Model Congruence (r={r:.3f}, p={p:.2e})")

    # Add diagonal
    lims = [min(min(scores_a), min(scores_b)), max(max(scores_a), max(scores_b))]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cross_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved cross_model_comparison.png")


def plot_component_loadings(analyses, output_dir="results/plots"):
    """Plot the loadings of top components on preference items."""
    for model_name, analysis in analyses.items():
        if model_name == "cross_model_comparison":
            continue

        interpretations = analysis.get("interpretations", [])
        if not interpretations:
            continue

        fig, axes = plt.subplots(len(interpretations), 1, figsize=(12, 4 * len(interpretations)))
        if len(interpretations) == 1:
            axes = [axes]

        for idx, interp in enumerate(interpretations):
            ax = axes[idx]

            # Collect all items with their loadings
            all_items = []
            for item_str in interp["positive_pole"]:
                name, loading = item_str.rsplit("(", 1)
                loading = float(loading.rstrip(")"))
                all_items.append((name.strip(), loading))

            for item_str in interp["negative_pole"]:
                name, loading = item_str.rsplit("(", 1)
                loading = float(loading.rstrip(")"))
                all_items.append((name.strip(), loading))

            # Sort by loading
            all_items.sort(key=lambda x: x[1])

            if all_items:
                names = [x[0][:40] for x in all_items]
                values = [x[1] for x in all_items]
                colors = ['coral' if v < 0 else 'steelblue' for v in values]
                ax.barh(names, values, color=colors)

            ax.set_title(f"Component {interp['component']}")
            ax.axvline(0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/components_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved components_{model_name}.png")


def plot_domain_congruence(output_dir="results/plots"):
    """Plot average congruence by domain composition of profiles."""
    import glob
    result_files = sorted(glob.glob("results/*_results.json"))

    fig, axes = plt.subplots(1, len(result_files), figsize=(7 * len(result_files), 5))
    if len(result_files) == 1:
        axes = [axes]

    for idx, filepath in enumerate(result_files):
        model_name = filepath.split("/")[-1].replace("_results.json", "")
        with open(filepath) as f:
            results = json.load(f)

        # Group by dominant domain
        domain_scores = {}
        for r in results:
            domains = r.get("domains", [])
            combined = (r["behavioral_congruence"] + r["coherence_rating"]) / 2
            for d in domains:
                if d not in domain_scores:
                    domain_scores[d] = []
                domain_scores[d].append(combined)

        ax = axes[idx]
        domain_names = sorted(domain_scores.keys())
        means = [np.mean(domain_scores[d]) for d in domain_names]
        stds = [np.std(domain_scores[d]) for d in domain_names]
        ax.bar(domain_names, means, yerr=stds, capsize=3, alpha=0.7, color="steelblue")
        ax.set_ylabel("Mean Congruence")
        ax.set_title(f"Congruence by Domain: {model_name}")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/domain_congruence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved domain_congruence.png")


def main():
    """Generate all visualizations."""
    analyses = load_results()

    plot_scree(analyses)
    plot_congruence_distribution()
    plot_cooccurrence_heatmap()
    plot_cross_model_comparison()
    plot_component_loadings(analyses)
    plot_domain_congruence()

    print("\nAll plots saved to results/plots/")


if __name__ == "__main__":
    main()
