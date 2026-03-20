"""
Deep statistical analysis of person-space mapping results.
"""

import json
import os
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from preference_items import PREFERENCE_ITEMS


def load_all():
    with open("results/analyses.json") as f:
        analyses = json.load(f)
    model_results = {}
    for name in ["gpt-4.1-nano", "gpt-4.1-mini"]:
        with open(f"results/{name}_results.json") as f:
            model_results[name] = json.load(f)
    return analyses, model_results


def analyze_congruence_variance(model_results):
    """Test H1: Is there significant variance in congruence scores?"""
    print("\n" + "="*60)
    print("H1: VARIANCE IN CONGRUENCE SCORES")
    print("="*60)

    for model_name, results in model_results.items():
        behavioral = [r["behavioral_congruence"] for r in results]
        coherence = [r["coherence_rating"] for r in results]
        combined = [(b + c) / 2 for b, c in zip(behavioral, coherence)]

        print(f"\n{model_name}:")
        print(f"  Behavioral congruence: {np.mean(behavioral):.4f} ± {np.std(behavioral):.4f}")
        print(f"    Range: [{np.min(behavioral):.4f}, {np.max(behavioral):.4f}]")
        print(f"  Coherence rating: {np.mean(coherence):.4f} ± {np.std(coherence):.4f}")
        print(f"    Range: [{np.min(coherence):.4f}, {np.max(coherence):.4f}]")
        print(f"  Combined: {np.mean(combined):.4f} ± {np.std(combined):.4f}")

        # Test if variance is significantly above what we'd expect from random noise
        # Under null: all profiles equally congruent, variance comes only from measurement noise
        # Permutation test: shuffle scores across profiles, compare variance
        observed_var = np.var(combined)
        n_perm = 1000
        perm_vars = []
        rng = np.random.RandomState(42)
        for _ in range(n_perm):
            shuffled = rng.permutation(combined)
            perm_vars.append(np.var(shuffled))
        # This is a baseline - but since we're shuffling the same values, variance is the same
        # Instead, test if the distribution differs from uniform:
        # Kolmogorov-Smirnov test against uniform distribution on [min, max]
        ks_stat, ks_p = stats.kstest(combined, 'uniform', args=(min(combined), max(combined)-min(combined)))
        print(f"  KS test (vs uniform): stat={ks_stat:.4f}, p={ks_p:.4e}")

        # Coefficient of variation
        cv = np.std(combined) / np.mean(combined)
        print(f"  Coefficient of variation: {cv:.4f}")

        # Skewness and kurtosis
        print(f"  Skewness: {stats.skew(combined):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(combined):.4f}")

        # Are behavioral and coherence scores correlated?
        r, p = pearsonr(behavioral, coherence)
        print(f"  Behavioral-Coherence correlation: r={r:.4f}, p={p:.4e}")


def analyze_low_rank_significance(model_results):
    """Test H2: Is the low-rank structure significant?"""
    print("\n" + "="*60)
    print("H2: SIGNIFICANCE OF LOW-RANK STRUCTURE")
    print("="*60)

    for model_name in model_results:
        matrix = np.load(f"results/{model_name}_cooccur_matrix.npy")

        # SVD of actual matrix
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, s_actual, _ = np.linalg.svd(centered, full_matrices=False)
        var_actual = s_actual**2 / np.sum(s_actual**2)

        # Generate random matrices with same marginal statistics
        n_perm = 100
        random_rank_80 = []
        random_participation = []
        rng = np.random.RandomState(42)

        for _ in range(n_perm):
            # Random matrix with same mean and variance per element
            rand_matrix = rng.normal(matrix.mean(), matrix.std(), matrix.shape)
            rand_centered = rand_matrix - rand_matrix.mean(axis=0, keepdims=True)
            _, s_rand, _ = np.linalg.svd(rand_centered, full_matrices=False)
            var_rand = s_rand**2 / np.sum(s_rand**2)
            cumvar_rand = np.cumsum(var_rand)
            random_rank_80.append(np.searchsorted(cumvar_rand, 0.80) + 1)
            random_participation.append((np.sum(s_rand**2))**2 / np.sum(s_rand**4))

        cumvar_actual = np.cumsum(var_actual)
        actual_rank_80 = np.searchsorted(cumvar_actual, 0.80) + 1
        actual_participation = (np.sum(s_actual**2))**2 / np.sum(s_actual**4)

        # p-value: fraction of random matrices with lower rank
        p_rank = np.mean(np.array(random_rank_80) <= actual_rank_80)
        p_part = np.mean(np.array(random_participation) <= actual_participation)

        print(f"\n{model_name}:")
        print(f"  Actual rank-80%: {actual_rank_80}, Random mean: {np.mean(random_rank_80):.1f} ± {np.std(random_rank_80):.1f}")
        print(f"  p-value (rank): {p_rank:.4f}")
        print(f"  Actual participation ratio: {actual_participation:.1f}, Random mean: {np.mean(random_participation):.1f} ± {np.std(random_participation):.1f}")
        print(f"  p-value (participation): {p_part:.4f}")


def analyze_cross_model_structure(model_results):
    """Test H4: Cross-model structural similarity."""
    print("\n" + "="*60)
    print("H4: CROSS-MODEL STRUCTURAL SIMILARITY")
    print("="*60)

    models = list(model_results.keys())
    if len(models) < 2:
        print("Need 2+ models")
        return

    # Compare congruence scores directly
    scores_a = [(r["behavioral_congruence"] + r["coherence_rating"])/2 for r in model_results[models[0]]]
    scores_b = [(r["behavioral_congruence"] + r["coherence_rating"])/2 for r in model_results[models[1]]]

    r_pearson, p_pearson = pearsonr(scores_a, scores_b)
    r_spearman, p_spearman = spearmanr(scores_a, scores_b)

    print(f"\nCongruence score correlation ({models[0]} vs {models[1]}):")
    print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.4e}")
    print(f"  Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.4e}")

    # Compare co-occurrence matrices via Mantel test
    mat_a = np.load(f"results/{models[0]}_cooccur_matrix.npy")
    mat_b = np.load(f"results/{models[1]}_cooccur_matrix.npy")

    # Flatten upper triangle for comparison
    idx = np.triu_indices(60, k=1)
    vec_a = mat_a[idx]
    vec_b = mat_b[idx]

    r_mantel, p_mantel = pearsonr(vec_a, vec_b)
    print(f"\nCo-occurrence matrix correlation (Mantel-like):")
    print(f"  Pearson r = {r_mantel:.4f}, p = {p_mantel:.4e}")

    # Permutation test for Mantel
    n_perm = 1000
    rng = np.random.RandomState(42)
    perm_rs = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(60)
        mat_b_perm = mat_b[np.ix_(perm_idx, perm_idx)]
        vec_b_perm = mat_b_perm[idx]
        perm_rs.append(pearsonr(vec_a, vec_b_perm)[0])

    p_mantel_perm = np.mean(np.array(perm_rs) >= r_mantel)
    print(f"  Permutation p-value: {p_mantel_perm:.4f}")

    # Compare SVD structure via Procrustes
    from scipy.spatial import procrustes

    _, s_a, Vt_a = np.linalg.svd(mat_a - mat_a.mean(0), full_matrices=False)
    _, s_b, Vt_b = np.linalg.svd(mat_b - mat_b.mean(0), full_matrices=False)

    # Compare top-k components
    for k in [3, 5, 10]:
        Va = Vt_a[:k].T  # 60 x k
        Vb = Vt_b[:k].T  # 60 x k

        try:
            _, _, disparity = procrustes(Va, Vb)
            print(f"  Procrustes disparity (top-{k} components): {disparity:.4f}")
        except Exception as e:
            print(f"  Procrustes (top-{k}) failed: {e}")

    # Canonical Correlation Analysis of top components
    Va_5 = Vt_a[:5].T
    Vb_5 = Vt_b[:5].T
    # Cross-correlation matrix
    cross_corr = np.abs(Va_5.T @ Vb_5)
    print(f"\n  Cross-correlation (top-5 components, absolute values):")
    for i in range(5):
        print(f"    Component {i+1}: max cross-corr = {cross_corr[i].max():.3f} "
              f"(with component {cross_corr[i].argmax()+1})")


def analyze_profile_characteristics(model_results):
    """Analyze which profile characteristics predict congruence."""
    print("\n" + "="*60)
    print("PROFILE CHARACTERISTICS vs CONGRUENCE")
    print("="*60)

    for model_name, results in model_results.items():
        print(f"\n{model_name}:")

        # Feature: number of unique domains
        n_domains = [len(r["domains"]) for r in results]
        combined = [(r["behavioral_congruence"] + r["coherence_rating"])/2 for r in results]

        r, p = pearsonr(n_domains, combined)
        print(f"  Domain diversity vs congruence: r={r:.4f}, p={p:.4e}")

        # Feature: fraction of "like" vs "dislike" in profile
        like_fracs = []
        for r in results:
            profile = r["profile"]
            n_likes = sum(1 for v in profile.values() if v == "like")
            like_fracs.append(n_likes / len(profile))

        r, p = pearsonr(like_fracs, combined)
        print(f"  Like-fraction vs congruence: r={r:.4f}, p={p:.4e}")

        # Top 10 highest and lowest congruence profiles
        sorted_idx = np.argsort(combined)
        print(f"\n  Top 5 most congruent profiles:")
        for i in sorted_idx[-5:][::-1]:
            profile_desc = []
            for k, v in results[i]["profile"].items():
                item = PREFERENCE_ITEMS[int(k)]
                profile_desc.append(item[v][:30])
            print(f"    Score: {combined[i]:.3f} — {'; '.join(profile_desc[:4])}...")

        print(f"\n  Top 5 least congruent profiles:")
        for i in sorted_idx[:5]:
            profile_desc = []
            for k, v in results[i]["profile"].items():
                item = PREFERENCE_ITEMS[int(k)]
                profile_desc.append(item[v][:30])
            print(f"    Score: {combined[i]:.3f} — {'; '.join(profile_desc[:4])}...")


def plot_singular_value_comparison(output_dir="results/plots"):
    """Compare singular value spectra across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, color in [("gpt-4.1-nano", "steelblue"), ("gpt-4.1-mini", "coral")]:
        matrix = np.load(f"results/{model_name}_cooccur_matrix.npy")
        centered = matrix - matrix.mean(0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var_exp = s**2 / np.sum(s**2)
        cumvar = np.cumsum(var_exp)

        axes[0].plot(range(1, len(var_exp)+1), var_exp, 'o-', label=model_name, color=color, markersize=3)
        axes[1].plot(range(1, len(cumvar)+1), cumvar, 'o-', label=model_name, color=color, markersize=3)

    # Add random baseline
    rng = np.random.RandomState(42)
    rand_vars = []
    for _ in range(50):
        rand_m = rng.normal(0.5, 0.1, (60, 60))
        rand_c = rand_m - rand_m.mean(0)
        _, s_r, _ = np.linalg.svd(rand_c, full_matrices=False)
        rand_vars.append(s_r**2 / np.sum(s_r**2))
    rand_mean = np.mean(rand_vars, axis=0)
    rand_cumvar = np.cumsum(rand_mean)
    axes[0].plot(range(1, len(rand_mean)+1), rand_mean, 'k--', label="Random", alpha=0.5, markersize=2)
    axes[1].plot(range(1, len(rand_cumvar)+1), rand_cumvar, 'k--', label="Random", alpha=0.5, markersize=2)

    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title("Singular Value Spectrum (Co-occurrence)")
    axes[0].legend()
    axes[0].set_xlim(0, 30)

    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Cumulative Variance")
    axes[1].set_title("Cumulative Variance Explained")
    axes[1].legend()
    axes[1].axhline(0.8, color='gray', linestyle='--', alpha=0.3)
    axes[1].axhline(0.9, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sv_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved sv_comparison.png")


def plot_behavioral_vs_coherence(model_results, output_dir="results/plots"):
    """Scatter plot of behavioral congruence vs coherence rating."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (model_name, results) in enumerate(model_results.items()):
        behavioral = [r["behavioral_congruence"] for r in results]
        coherence = [r["coherence_rating"] for r in results]

        ax = axes[idx]
        ax.scatter(behavioral, coherence, alpha=0.5, s=30)
        r, p = pearsonr(behavioral, coherence)
        ax.set_xlabel("Behavioral Congruence")
        ax.set_ylabel("Coherence Rating")
        ax.set_title(f"{model_name} (r={r:.3f})")

        # Fit line
        z = np.polyfit(behavioral, coherence, 1)
        x_line = np.linspace(min(behavioral), max(behavioral), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/behavioral_vs_coherence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved behavioral_vs_coherence.png")


def main():
    analyses, model_results = load_all()

    analyze_congruence_variance(model_results)
    analyze_low_rank_significance(model_results)
    analyze_cross_model_structure(model_results)
    analyze_profile_characteristics(model_results)

    plot_singular_value_comparison()
    plot_behavioral_vs_coherence(model_results)

    print("\n\nAnalysis complete!")


if __name__ == "__main__":
    main()
