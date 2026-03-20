# Mapping Person Space Through Simple Likes and Dislikes

Investigating whether the space of LLM personas is low-rank by probing which combinations of everyday preferences LLMs adopt congruently, and whether this structure is consistent across models.

## Key Findings

- **Person space is low-rank**: The preference co-occurrence matrix has ~21 effective dimensions out of 60 (65% reduction), significantly lower than random baselines (p < 0.001)
- **Cross-model consistency is extremely high**: GPT-4.1-nano and GPT-4.1-mini share nearly identical person-space structure (co-occurrence correlation r = 0.96, top-5 PCA components match at r > 0.90)
- **Not all preference combinations are equal**: Congruence scores vary significantly across profiles. Stereotypically consistent profiles (e.g., "quiet, cooperative") are adopted more naturally than contradictory ones
- **Interpretable dimensions emerge**: Top components separate "outgoing intellectual vs. homebody", "creative/artistic vs. traditional", and "privacy/minimalism vs. tech enthusiasm"
- **Positivity bias**: GPT-4.1-mini adopts "like" preferences more easily than "dislike" preferences (r = 0.31)

## Project Structure

```
├── REPORT.md              # Full research report with results
├── planning.md            # Research plan and motivation
├── src/
│   ├── preference_items.py  # 60 preference items across 10 domains
│   ├── experiment.py        # Main experiment runner
│   ├── analysis.py          # Statistical analysis
│   └── visualize.py         # Visualization generation
├── results/
│   ├── analyses.json        # All analysis results
│   ├── config.json          # Experiment configuration
│   ├── *_results.json       # Raw per-model results
│   ├── *_cooccur_matrix.npy # Co-occurrence matrices
│   └── plots/               # All visualizations
├── literature_review.md     # Background literature
├── resources.md             # Resource catalog
├── papers/                  # Downloaded papers
├── datasets/                # Pre-gathered datasets
└── code/                    # Reference implementations
```

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy scipy scikit-learn matplotlib seaborn pandas

# Run experiment (requires OPENAI_API_KEY)
cd src && python experiment.py

# Generate visualizations
python visualize.py

# Run statistical analysis
python analysis.py
```

**Requirements**: Python 3.10+, OpenAI API key. Total cost: ~$5-10 for both models. Runtime: ~60 minutes.

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, and analysis.
