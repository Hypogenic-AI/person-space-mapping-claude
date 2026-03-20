"""
Main experiment: Mapping Person Space Through Simple Likes and Dislikes

This script:
1. Generates persona profiles from preference items
2. Presents them to LLMs and measures congruence
3. Analyzes dimensionality of the resulting preference space
4. Compares structure across LLMs
"""

import json
import os
import random
import time
import sys
import asyncio
import numpy as np
from datetime import datetime
from openai import OpenAI

from preference_items import PREFERENCE_ITEMS, PROBE_QUESTIONS, NUM_ITEMS

# Set working directory to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
NUM_PROFILES = 120        # Number of persona profiles to test
PREFS_PER_PROFILE = 8     # Preferences per profile
NUM_PROBES = 5            # Follow-up questions per profile
CONGRUENCE_BATCH_SIZE = 5 # Profiles per batch for efficiency

# Models to test
MODELS = {
    "gpt-4.1-nano": "gpt-4.1-nano",      # Fast, cheap model for primary experiment
    "gpt-4.1-mini": "gpt-4.1-mini",      # Stronger model for comparison
}


def generate_persona_profiles(n_profiles, prefs_per_profile, seed=SEED):
    """Generate random persona profiles as subsets of preference items.

    Returns list of profiles, each a dict mapping item_id -> polarity ('like' or 'dislike').
    Includes some stereotype-consistent and some mixed profiles.
    """
    rng = random.Random(seed)
    profiles = []
    item_ids = list(range(NUM_ITEMS))

    for i in range(n_profiles):
        selected = rng.sample(item_ids, prefs_per_profile)
        # Each preference is randomly like or dislike
        profile = {}
        for item_id in selected:
            profile[item_id] = rng.choice(["like", "dislike"])
        profiles.append(profile)

    return profiles


def profile_to_description(profile):
    """Convert a profile dict to a natural language persona description."""
    traits = []
    for item_id, polarity in sorted(profile.items()):
        item = PREFERENCE_ITEMS[item_id]
        traits.append(item[polarity])
    return "; ".join(traits)


def build_system_prompt(profile):
    """Build a system prompt that assigns a persona profile to the LLM."""
    desc = profile_to_description(profile)
    return (
        f"You are a person with the following traits and preferences: {desc}. "
        f"Respond naturally and consistently as this person. Stay in character. "
        f"When answering questions, let these preferences genuinely inform your responses."
    )


def build_scoring_prompt(profile, probe_question, response):
    """Build a prompt to score how congruent a response is with the assigned profile."""
    desc = profile_to_description(profile)
    return (
        f"A person was assigned the following persona traits:\n"
        f"{desc}\n\n"
        f"They were asked: \"{probe_question}\"\n\n"
        f"They responded: \"{response}\"\n\n"
        f"On a scale of 0 to 10, how well does this response reflect the assigned "
        f"preferences? Consider:\n"
        f"- Does the response align with the stated likes/dislikes?\n"
        f"- Does the response naturally extend these preferences (volunteering related preferences)?\n"
        f"- Does the response feel like it comes from a coherent person with these traits?\n\n"
        f"Respond with ONLY a number from 0-10, nothing else."
    )


def build_incongruence_detection_prompt(profile):
    """Ask the model directly whether the preference set feels coherent."""
    desc = profile_to_description(profile)
    return (
        f"Consider a person with the following set of preferences:\n"
        f"{desc}\n\n"
        f"On a scale of 0 to 10, how coherent and natural does this combination of "
        f"preferences feel? A score of 10 means these preferences fit together perfectly "
        f"and you can easily imagine a real person with all of them. A score of 0 means "
        f"these preferences are deeply contradictory or extremely unlikely to co-occur.\n\n"
        f"Think briefly about why, then respond with your score.\n"
        f"Format: SCORE: [number]\nREASON: [brief explanation]"
    )


class ExperimentRunner:
    """Runs the persona congruence experiment."""

    def __init__(self, model_name, model_id):
        self.model_name = model_name
        self.model_id = model_id
        self.client = OpenAI()
        self.call_count = 0
        self.total_tokens = 0

    def _call_api(self, messages, max_tokens=300, temperature=0.7):
        """Make an API call with retry logic."""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.call_count += 1
                self.total_tokens += response.usage.total_tokens
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"  API error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  API error after 3 attempts: {e}")
                    return None

    def measure_behavioral_congruence(self, profile, probe_questions):
        """Measure how congruently an LLM adopts a persona profile.

        Returns congruence score (0-1) and individual probe scores.
        """
        system_prompt = build_system_prompt(profile)
        scores = []
        responses = []

        for question in probe_questions:
            # Get response in character
            response = self._call_api(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=200,
                temperature=0.7,
            )
            if response is None:
                continue

            # Score the response for congruence
            scoring_prompt = build_scoring_prompt(profile, question, response)
            score_str = self._call_api(
                [{"role": "user", "content": scoring_prompt}],
                max_tokens=10,
                temperature=0.0,
            )

            try:
                score = float(score_str.strip().split()[0]) / 10.0
                score = max(0.0, min(1.0, score))
            except (ValueError, IndexError, AttributeError):
                score = 0.5  # default if parsing fails

            scores.append(score)
            responses.append({"question": question, "response": response, "score": score})

        avg_score = np.mean(scores) if scores else 0.5
        return avg_score, responses

    def measure_coherence_rating(self, profile):
        """Ask the model to rate how coherent a preference set feels."""
        prompt = build_incongruence_detection_prompt(profile)
        response = self._call_api(
            [{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )

        if response is None:
            return 5.0, "API error"

        # Parse score
        try:
            for line in response.split("\n"):
                if "SCORE:" in line.upper():
                    score_str = line.split(":")[-1].strip()
                    # Extract first number
                    num = ""
                    for ch in score_str:
                        if ch.isdigit() or ch == ".":
                            num += ch
                        elif num:
                            break
                    score = float(num)
                    break
            else:
                # Try to find any number
                import re
                nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
                score = float(nums[0]) if nums else 5.0
        except (ValueError, IndexError):
            score = 5.0

        # Extract reason
        reason = ""
        for line in response.split("\n"):
            if "REASON:" in line.upper():
                reason = line.split(":", 1)[-1].strip()
                break
        if not reason:
            reason = response

        return score, reason

    def run_experiment(self, profiles, probe_questions):
        """Run the full experiment for one model."""
        results = []
        n = len(profiles)

        print(f"\n{'='*60}")
        print(f"Running experiment with {self.model_name} ({n} profiles)")
        print(f"{'='*60}")

        for i, profile in enumerate(profiles):
            if i % 10 == 0:
                print(f"  Progress: {i}/{n} profiles ({self.call_count} API calls, {self.total_tokens} tokens)")

            # Measure behavioral congruence
            behav_score, responses = self.measure_behavioral_congruence(profile, probe_questions)

            # Measure coherence rating
            coherence_score, reason = self.measure_coherence_rating(profile)

            results.append({
                "profile_idx": i,
                "profile": {str(k): v for k, v in profile.items()},
                "behavioral_congruence": behav_score,
                "coherence_rating": coherence_score / 10.0,  # normalize to 0-1
                "coherence_reason": reason,
                "probe_responses": responses,
                "domains": list(set(PREFERENCE_ITEMS[k]["domain"] for k in profile.keys())),
            })

        print(f"  Done! Total API calls: {self.call_count}, Total tokens: {self.total_tokens}")
        return results


def build_preference_cooccurrence_matrix(profiles, results):
    """Build a matrix showing how preference pairs affect congruence.

    For each pair (i, j), compute the average congruence when both preferences
    appear in the same profile, vs. when they don't.
    """
    n_items = NUM_ITEMS
    # Count co-occurrences and sum congruence
    cooccur_sum = np.zeros((n_items, n_items))
    cooccur_count = np.zeros((n_items, n_items))

    for profile, result in zip(profiles, results):
        score = (result["behavioral_congruence"] + result["coherence_rating"]) / 2
        items = list(profile.keys())
        for a in items:
            for b in items:
                cooccur_sum[a, b] += score
                cooccur_count[a, b] += 1

    # Average congruence when co-occurring
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccur_avg = np.where(cooccur_count > 0, cooccur_sum / cooccur_count, 0.5)

    return cooccur_avg, cooccur_count


def build_profile_preference_matrix(profiles, results):
    """Build a matrix: rows=profiles, cols=preferences, values=congruence-weighted presence.

    Each cell is: congruence_score if preference is in profile (with sign for like/dislike), 0 otherwise.
    """
    n_profiles = len(profiles)
    n_items = NUM_ITEMS

    # Encode as +score for like, -score for dislike, 0 for absent
    matrix = np.zeros((n_profiles, n_items))
    scores = np.zeros(n_profiles)

    for i, (profile, result) in enumerate(zip(profiles, results)):
        score = (result["behavioral_congruence"] + result["coherence_rating"]) / 2
        scores[i] = score
        for item_id, polarity in profile.items():
            sign = 1 if polarity == "like" else -1
            matrix[i, item_id] = sign * score

    return matrix, scores


def analyze_dimensionality(matrix, name="matrix"):
    """Perform SVD and analyze effective dimensionality."""
    # Center the matrix
    centered = matrix - matrix.mean(axis=0, keepdims=True)

    # SVD
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)

    # Variance explained
    var_explained = s**2 / np.sum(s**2)
    cumvar = np.cumsum(var_explained)

    # Effective rank metrics
    rank_80 = np.searchsorted(cumvar, 0.80) + 1
    rank_90 = np.searchsorted(cumvar, 0.90) + 1
    rank_95 = np.searchsorted(cumvar, 0.95) + 1

    # Participation ratio (effective dimensionality)
    participation_ratio = (np.sum(s**2))**2 / np.sum(s**4) if np.sum(s**4) > 0 else 0

    print(f"\n--- Dimensionality Analysis: {name} ---")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Top 10 singular values: {s[:10].round(3)}")
    print(f"Variance explained (top 10): {var_explained[:10].round(4)}")
    print(f"Cumulative variance (top 10): {cumvar[:10].round(4)}")
    print(f"Rank for 80% variance: {rank_80}")
    print(f"Rank for 90% variance: {rank_90}")
    print(f"Rank for 95% variance: {rank_95}")
    print(f"Participation ratio: {participation_ratio:.2f}")

    return {
        "singular_values": s.tolist(),
        "variance_explained": var_explained.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "rank_80": int(rank_80),
        "rank_90": int(rank_90),
        "rank_95": int(rank_95),
        "participation_ratio": float(participation_ratio),
        "Vt": Vt,
        "U": U,
    }


def interpret_components(Vt, n_components=5, top_k=8):
    """Interpret top PCA components by their loadings on preference items."""
    interpretations = []

    for comp_idx in range(min(n_components, Vt.shape[0])):
        loadings = Vt[comp_idx]
        # Top positive and negative loadings
        sorted_idx = np.argsort(loadings)
        top_positive = sorted_idx[-top_k:][::-1]
        top_negative = sorted_idx[:top_k]

        pos_items = []
        for idx in top_positive:
            if abs(loadings[idx]) > 0.05:
                item = PREFERENCE_ITEMS[idx]
                pos_items.append(f"{item['like']} ({loadings[idx]:.3f})")

        neg_items = []
        for idx in top_negative:
            if abs(loadings[idx]) > 0.05:
                item = PREFERENCE_ITEMS[idx]
                neg_items.append(f"{item['like']} ({loadings[idx]:.3f})")

        interp = {
            "component": comp_idx + 1,
            "positive_pole": pos_items,
            "negative_pole": neg_items,
        }
        interpretations.append(interp)

        print(f"\nComponent {comp_idx + 1}:")
        print(f"  Positive pole: {', '.join(pos_items[:4])}")
        print(f"  Negative pole: {', '.join(neg_items[:4])}")

    return interpretations


def compare_models(results_a, results_b, profiles):
    """Compare person-space structure between two models."""
    scores_a = np.array([(r["behavioral_congruence"] + r["coherence_rating"])/2 for r in results_a])
    scores_b = np.array([(r["behavioral_congruence"] + r["coherence_rating"])/2 for r in results_b])

    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(scores_a, scores_b)
    spearman_r, spearman_p = spearmanr(scores_a, scores_b)

    print(f"\n--- Cross-Model Comparison ---")
    print(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"Spearman rho: {spearman_r:.4f} (p={spearman_p:.4e})")
    print(f"Mean congruence A: {scores_a.mean():.4f} ± {scores_a.std():.4f}")
    print(f"Mean congruence B: {scores_b.mean():.4f} ± {scores_b.std():.4f}")

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "mean_congruence_a": float(scores_a.mean()),
        "std_congruence_a": float(scores_a.std()),
        "mean_congruence_b": float(scores_b.mean()),
        "std_congruence_b": float(scores_b.std()),
    }


def main():
    """Run the complete experiment."""
    print(f"Starting experiment at {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    # Step 1: Generate profiles
    print("\n--- Step 1: Generating persona profiles ---")
    profiles = generate_persona_profiles(NUM_PROFILES, PREFS_PER_PROFILE)
    print(f"Generated {len(profiles)} profiles with {PREFS_PER_PROFILE} preferences each")

    # Show a few examples
    for i in range(3):
        print(f"\nProfile {i}: {profile_to_description(profiles[i])}")

    all_results = {}
    all_analyses = {}

    # Step 2: Run experiment for each model
    for model_name, model_id in MODELS.items():
        runner = ExperimentRunner(model_name, model_id)
        results = runner.run_experiment(profiles, PROBE_QUESTIONS)
        all_results[model_name] = results

        # Save raw results
        with open(f"results/{model_name}_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Step 3: Build matrices and analyze
        matrix, scores = build_profile_preference_matrix(profiles, results)
        cooccur, counts = build_preference_cooccurrence_matrix(profiles, results)

        # Dimensionality analysis on profile-preference matrix
        analysis = analyze_dimensionality(matrix, name=f"{model_name} profile-preference")

        # Dimensionality analysis on co-occurrence matrix
        cooccur_analysis = analyze_dimensionality(cooccur, name=f"{model_name} co-occurrence")

        # Interpret components
        print(f"\n--- Interpreting components for {model_name} ---")
        interpretations = interpret_components(analysis["Vt"], n_components=5)

        all_analyses[model_name] = {
            "profile_analysis": {k: v for k, v in analysis.items() if k not in ("Vt", "U")},
            "cooccur_analysis": {k: v for k, v in cooccur_analysis.items() if k not in ("Vt", "U")},
            "interpretations": interpretations,
            "congruence_stats": {
                "mean_behavioral": float(np.mean([r["behavioral_congruence"] for r in results])),
                "std_behavioral": float(np.std([r["behavioral_congruence"] for r in results])),
                "mean_coherence": float(np.mean([r["coherence_rating"] for r in results])),
                "std_coherence": float(np.std([r["coherence_rating"] for r in results])),
                "mean_combined": float(scores.mean()),
                "std_combined": float(scores.std()),
            },
            "api_stats": {
                "total_calls": runner.call_count,
                "total_tokens": runner.total_tokens,
            },
        }

        # Save matrices
        np.save(f"results/{model_name}_profile_matrix.npy", matrix)
        np.save(f"results/{model_name}_cooccur_matrix.npy", cooccur)

    # Step 4: Cross-model comparison
    model_names = list(all_results.keys())
    if len(model_names) >= 2:
        comparison = compare_models(
            all_results[model_names[0]],
            all_results[model_names[1]],
            profiles
        )
        all_analyses["cross_model_comparison"] = comparison

    # Save all analyses
    with open("results/analyses.json", "w") as f:
        json.dump(all_analyses, f, indent=2, default=str)

    # Save config
    config = {
        "seed": SEED,
        "num_profiles": NUM_PROFILES,
        "prefs_per_profile": PREFS_PER_PROFILE,
        "num_probes": NUM_PROBES,
        "models": MODELS,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    with open("results/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete at {datetime.now().isoformat()}")
    print(f"Results saved to results/")
    print(f"{'='*60}")

    return all_results, all_analyses, profiles


if __name__ == "__main__":
    main()
