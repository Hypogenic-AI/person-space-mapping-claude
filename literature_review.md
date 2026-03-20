# Literature Review: Mapping Person Space Through Simple Likes and Dislikes

## Research Area Overview

This research investigates whether the space of personas displayed by LLMs is low-rank and can be mapped by analyzing preference patterns (likes/dislikes). The literature converges from three directions: (1) **steering vector / representation engineering** showing personality traits exist as linear directions in LLM activation space, (2) **personalized alignment** showing user preferences have low-rank structure amenable to few-shot adaptation, and (3) **persona evaluation** showing LLMs can simulate personality traits but with important caveats about self-report vs. behavioral validity.

---

## Key Papers

### 1. LoRe: Personalizing LLMs via Low-Rank Reward Modeling (Bose et al., 2025)
- **arXiv:** 2504.14439
- **Key Contribution:** Formalizes the assumption that the N×M user-preference matrix is low-rank (rank B << min(N,M)) and decomposes it as P = WR, where R is a B×M reward basis and W is user-specific simplex weights.
- **Methodology:** Joint training of shared reward basis R_φ and per-user weight vectors {w_i}. New users adapted via few-shot optimization over B-dimensional weight vector with frozen basis.
- **Datasets:** PersonalLLM (synthetic, 10K prompts × 10 reward models), Reddit TLDR (real, 40 workers), PRISM (real, 1,500 participants from 75 countries).
- **Results:** 93.8% accuracy on diverse preferences (vs 86.4% monolithic baseline). On PRISM with ~4 examples/user, LoRe is the only method maintaining performance on unseen users (71.0% vs. PAL's 59.0% unseen).
- **Key Insight for Our Research:** Each user is a B-dimensional point on the simplex — this IS person space. Few-shot identification works with as few as 3-4 preference comparisons.

### 2. Persona Vectors: Monitoring and Controlling Character Traits (Chen et al., 2025)
- **arXiv:** 2507.21509
- **Key Contribution:** Identifies single linear directions in activation space for personality traits (evil, sycophancy, hallucination). Demonstrates monitoring and preventative steering.
- **Methodology:** Contrastive difference-in-means on residual stream activations. Automated pipeline generates prompts, extracts vectors, validates via steering and projection.
- **Key Findings:** Projection onto persona vectors correlates r=0.75-0.83 with trait expression. Traits are correlated, not orthogonal — negative traits shift together. Cross-trait correlations (0.34-0.86) lower than same-trait (0.76-0.97).
- **Open Questions (from paper):** "How high-dimensional is persona space? Does there exist a natural 'persona basis'?" — directly our research question.

### 3. Do Personality Traits Interfere? Geometric Limitations of Steering (Bhandari et al., 2026)
- **arXiv:** 2602.15847
- **Key Contribution:** Shows Big Five steering vectors are NOT orthogonal. Even hard orthonormalization doesn't eliminate behavioral cross-trait interference.
- **Methodology:** Six conditioning schemes (C0-C5) applied to trait vectors in LLaMA-3-8B and Ministral-8B. Evaluation via GPT-4o-mini judge on BFI questionnaires.
- **Key Findings:** A dominant "social desirability" axis exists — increasing any positive trait increases others and decreases Neuroticism. Geometric orthogonality ≠ behavioral independence (nonlinear mechanisms involved). C4 (soft projection, β=0.5) offers best trade-off.
- **Relevance:** Person space may have fewer effective dimensions than Big Five suggests. The entanglement pattern implies a lower-dimensional latent structure.

### 4. PERSONA: Dynamic and Compositional Inference-Time Personality Control (Feng et al., 2026)
- **arXiv:** 2602.15669
- **Key Contribution:** Training-free personality control via activation vector algebra. Shows vectors support scalar multiplication (linear intensity control, r>0.9) and addition (multi-trait composition).
- **Methodology:** Contrastive activation extraction for 10 Big Five poles. PERSONA-FLOW predicts steering coefficients per turn.
- **Key Findings:** Opposing traits have strong negative cosine similarity (-0.85). Cross-dimensional correlations small but non-zero. Compositionality works — vector addition produces predictable multi-trait profiles. Matches SFT upper bound (9.60 vs 9.61 on PersonalityBench).
- **Relevance:** Person space has algebraic structure — if likes/dislikes map onto activation directions, they too should compose.

### 5. Personas within Parameters: Low-Rank Adapters for User Behaviors (Thakur et al., 2024)
- **arXiv:** 2509.09689
- **Key Contribution:** Uses LoRA adapters to capture user persona clusters from likes/dislikes data.
- **Methodology:** GPT-4o distills user-item interactions into textual profiles. KMeans++ clusters users into K=4 personas. Per-persona LoRA (rank 256) fine-tuned on Phi-3-Mini.
- **Datasets:** MovieLens-1M (200 users, 100-200 interactions each).
- **Key Findings:** Persona-level LoRA beats full LLaMA-3-8B baseline. Clusters are semantically interpretable (genre preferences). Low-rank adapters suffice to capture user behavioral variation.
- **Relevance:** Direct evidence that likes/dislikes produce meaningful, low-rank person clusters.

### 6. The Personality Illusion (Han et al., 2025)
- **arXiv:** 2509.03730
- **Key Contribution:** Self-reported personality traits do NOT reliably predict LLM behavior (~24% of trait-task associations significant). Persona injection steers self-reports but not actual behavior.
- **Relevance:** Critical caution — mapping person space requires behavioral validation, not just questionnaire-based measurement.

### 7. PersonalLLM: Tailoring LLMs to Individual Preferences (Zollo et al., 2024)
- **arXiv:** 2409.20296
- **Key Contribution:** Benchmark with 10K prompts × 8 responses × 10 reward models simulating diverse users. Persona prompting produces preferences only half as diverse as real user base.
- **Relevance:** Establishes that high-level persona descriptions are insufficient for preference diversity — finer-grained mapping (likes/dislikes) is needed.

### 8. Prompts to Proxies: Preference Reconstruction (Wang et al., 2026)
- **arXiv:** 2509.11311
- **Key Contribution:** Formalizes preference reconstruction as representation learning. L1-regularized regression selects compact LLM ensemble matching target population.
- **Relevance:** Population preference distributions can be spanned by finite LLM agent basis — another validation of low-rank preference space.

### 9. AlignX: Scaling Personalized Preference (Li et al., 2025)
- **arXiv:** 2503.15463
- **Key Contribution:** 90-dimensional preference space grounded in psychology. AlignX dataset with 1.3M examples. 17% accuracy gain over baselines.
- **Relevance:** Most concrete implementation of "person space" with explicit dimensional structure.

### 10. Few-shot Steerable Alignment (Kobalczyk et al., 2024)
- **arXiv:** 2412.13998
- **Key Contribution:** Neural Processes for amortized inference of user-specific latent variables from few-shot preferences. NP-DPO enables inference-time adaptation.
- **Relevance:** Shows user position in preference space can be inferred from minimal data.

### 11. Personality Sliders (Hoppe et al., 2026)
- **arXiv:** 2603.03326
- **Key Contribution:** Sequential Adaptive Steering orthogonalizes steering vectors by training each on residual shifted by prior interventions. Achieves Pareto dominance on trait accuracy vs perplexity.

### 12. Provable Multi-Party RLHF (Zhong et al., 2024)
- **arXiv:** 2403.05006
- **Key Contribution:** Proves single reward function provably fails for heterogeneous preferences. Shared representation + individual rewards structure.

---

## Common Methodologies

1. **Contrastive Activation Extraction:** Used in Persona Vectors, PERSONA, CAA, Personality Sliders — generate responses under positive/negative prompts, compute mean difference in residual stream activations. (Papers 2, 4, 11)
2. **Low-Rank Preference Decomposition:** User-preference matrix factored as P=WR with low-rank B. (Papers 1, 5, 8)
3. **Big Five Personality Framework:** OCEAN traits as standard basis for personality evaluation. (Papers 3, 4, 6, 11)
4. **Few-Shot Adaptation:** Learning user-specific parameters from minimal preference data. (Papers 1, 10)

## Standard Baselines
- **Monolithic Reward Model (BT):** Single Bradley-Terry model, no personalization
- **VPL (Variational Preference Learning):** High-dimensional latent user codes
- **PAL (Pluralistic Alignment):** Prototype ideal points with distance-based reward
- **Contrastive Activation Addition (CAA):** Standard steering vector baseline

## Evaluation Metrics
- **Preference prediction accuracy:** Pairwise comparison accuracy on held-out data
- **BFI-44 scores:** Big Five Inventory questionnaire for personality measurement
- **Trait expression score (0-100):** LLM-judge rated trait intensity
- **MMLU accuracy:** General capability preservation
- **Cosine similarity:** Geometric relationship between steering vectors

## Datasets in the Literature
- **PRISM:** 1,500 participants, 75 countries, 8,011 conversations, 21 LLMs (Papers 1, 9)
- **PersonalLLM:** 10K prompts × 8 responses × 10 reward models (Papers 1, 7)
- **AlignX:** 1.3M examples, 90-dimensional preference space (Paper 9)
- **MovieLens-1M:** Classic user-item preference data (Paper 5)
- **Anthropic Persona Dataset:** Persona elicitation benchmarks (Perez et al., 2022)

## Gaps and Opportunities

1. **No study directly maps "person space" through simple likes/dislikes of everyday items.** Most work uses either psychometric instruments (Big Five) or LLM response preferences. Simple preferences (favorite foods, hobbies, music) as a mapping signal are unexplored.
2. **The effective dimensionality of person space is unknown.** Papers find entangled but manipulable directions, but systematic dimensionality analysis hasn't been done.
3. **Cross-model consistency of person space is untested.** Do the same preference patterns feel incongruent across different LLMs? Are the same persona dimensions discoverable?
4. **The relationship between preference-based and activation-based person space is unclear.** LoRe learns person space from preference data; PERSONA finds it in activations. Are these the same space?

## Recommendations for Experiment

- **Primary datasets:** PRISM (real user preferences with demographics) and PersonalLLM (controlled diverse preferences)
- **Recommended baselines:** LoRe (low-rank reward), monolithic BT, persona prompting
- **Recommended metrics:** Preference prediction accuracy, effective dimensionality (eigenvalue analysis), cross-model correlation of person-space structure
- **Key tools:** repeng/CAA for steering vector extraction, PRISM codebase for data loading
- **Methodological approach:** (1) Collect LLM responses to like/dislike probes, (2) extract activation representations, (3) analyze dimensionality via PCA/SVD, (4) test whether low-rank structure predicts held-out preferences, (5) compare across LLMs
