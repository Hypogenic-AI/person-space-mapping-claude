# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Mapping Person Space Through Simple Likes and Dislikes." The hypothesis is that LLM persona space is low-rank and can be mapped through preference patterns.

---

## Papers
Total papers downloaded: 24

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | LoRe: Low-Rank Reward Modeling | Bose et al. | 2025 | `2504.14439_lore_low_rank_reward_modeling.pdf` | Low-rank preference decomposition, P=WR |
| 2 | Persona Vectors | Chen et al. | 2025 | `2507.21509_persona_vectors_monitoring_controlling.pdf` | Linear persona directions, monitoring |
| 3 | Personality Traits Interfere (Geometric) | Bhandari et al. | 2026 | `2602.15847_personality_traits_interfere_geometric.pdf` | Non-orthogonality of Big Five vectors |
| 4 | PERSONA (Compositional) | Feng et al. | 2026 | `2602.15669_persona_compositional_activation_vectors.pdf` | Vector algebra for personality |
| 5 | Personas within Parameters | Thakur et al. | 2024 | `2509.09689_personas_within_parameters_low_rank.pdf` | LoRA persona clusters from likes/dislikes |
| 6 | The Personality Illusion | Han et al. | 2025 | `2509.03730_personality_illusion_self_reports_vs_behavior.pdf` | Self-report ≠ behavior |
| 7 | PersonalLLM | Zollo et al. | 2024 | `2409.20296_personalllm_tailoring_individual_preferences.pdf` | Preference diversity benchmark |
| 8 | Prompts to Proxies | Wang et al. | 2026 | `2509.11311_prompts_to_proxies_preference_reconstruction.pdf` | Preference reconstruction theory |
| 9 | AlignX | Li et al. | 2025 | `2503.15463_alignx_scaling_personalized_preference.pdf` | 90-dim preference space |
| 10 | PICLe | Choi & Li | 2024 | `2405.02501_picle_persona_in_context_learning.pdf` | Bayesian persona elicitation |
| 11 | Persona-Plug | Liu et al. | 2024 | `2409.11901_persona_plug_personalized_llms.pdf` | Lightweight user embedding |
| 12 | Few-shot Steerable Alignment | Kobalczyk et al. | 2024 | `2412.13998_few_shot_steerable_alignment.pdf` | Neural Processes for preferences |
| 13 | Personality Sliders | Hoppe et al. | 2026 | `2603.03326_personality_sliders_inference_time.pdf` | Sequential adaptive steering |
| 14 | BiPO Steering | Cao et al. | 2024 | `2406.00045_personalized_steering_bipo.pdf` | Bi-directional preference optimization |
| 15 | SteerX | Zhao et al. | 2025 | `2510.22256_steerx_disentangled_steering.pdf` | Causal disentanglement |
| 16 | Multi-Party RLHF | Zhong et al. | 2024 | `2403.05006_multi_party_rlhf_diverse_feedback.pdf` | Theory of diverse preferences |
| 17 | Finding Agreement | Bakker et al. | 2022 | `2211.15006_finding_agreement_diverse_preferences.pdf` | Individual preference modeling |
| 18 | Tutor Persona Steering | Lee et al. | 2026 | `2602.07639_tutor_persona_steering_vectors.pdf` | Persona vectors from dialogue |
| 19 | PersonaLLM (traits) | Jiang et al. | 2023 | `2305.02547_personallm_personality_traits.pdf` | Big Five in LLMs |
| 20 | LLMs Simulate Big Five | Sorokovikova et al. | 2024 | `2402.01765_llms_simulate_big_five.pdf` | Personality simulation stability |
| 21 | Neuron-based Personality | Deng et al. | 2024 | `2410.12327_neuron_based_personality_induction.pdf` | Mechanistic trait control |
| 22 | Two Tales of Persona | Tseng et al. | 2024 | `2406.01171_two_tales_persona_survey.pdf` | Survey paper |
| 23 | Beyond Discrete Personas | Pal et al. | 2024 | `2412.11250_beyond_discrete_personas.pdf` | Continuous personality |
| 24 | Aligning to Thousands | Lee et al. | 2024 | `2405.17977_aligning_thousands_preferences.pdf` | System message generalization |

See `papers/README.md` for detailed descriptions.

---

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| PRISM Alignment | HuggingFace | 77.8K rows, 133MB | Individual preference analysis | `datasets/prism/` | PRIMARY — 1,500 users, 75 countries |
| PersonalLLM | HuggingFace | 10.4K prompts, 50MB | Preference prediction | `datasets/personalllm/` | Controlled benchmark, 10 reward models |
| Anthropic Global Opinions | HuggingFace | 2.5K questions, 1MB | Opinion reference | `datasets/global_opinions/` | Country-level distributions |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| repeng | github.com/vgel/repeng | Steering vector library | `code/repeng/` | 698 stars, actively maintained |
| CAA | github.com/nrimsky/CAA | Contrastive activation addition | `code/CAA/` | 216 stars, ACL 2024 |
| PRISM Alignment | github.com/HannahKirk/prism-alignment | Dataset analysis code | `code/prism-alignment/` | 90 stars |
| Pluralistic Alignment | github.com/RamyaLab/pluralistic-alignment | Personalized reward models | `code/pluralistic-alignment/` | PAL baseline |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used arXiv API with 4 targeted queries covering: LLM persona preferences, personality space, steering vectors, low-rank behavior
2. Supplemented with second round of 4 queries on: Big Five + LLM, dimensional personality space, RLHF preference diversity, steering + persona
3. Searched HuggingFace for datasets with individual user preferences
4. Searched GitHub for implementation repositories

### Selection Criteria
- Papers directly addressing low-rank structure in preference/persona space (highest priority)
- Papers on steering vectors for personality control
- Papers with behavioral validation (not just self-report)
- Datasets with per-user IDs enabling individual preference analysis
- Code repositories with practical steering vector infrastructure

### Challenges Encountered
- Paper-finder service was unavailable; conducted manual search via arXiv API
- "Person space" / "personality space" queries returned mostly physical/proxemic space papers from robotics
- Many preference alignment papers aggregate preferences, losing individual-level variation

### Gaps and Workarounds
- No dataset of simple everyday likes/dislikes (favorite foods, hobbies, music) paired with LLM persona evaluation — this would need to be created as part of the experiment
- No existing study directly tests cross-LLM consistency of person-space structure
- PACIFIC dataset (personality-labeled preferences, arXiv:2602.07181) could not be confirmed as publicly available yet

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **PRISM** for real human preference data with demographics (validation of person-space structure)
- **PersonalLLM** for controlled methodology development (synthetic users with known ground truth)

### 2. Baseline Methods
- **LoRe** (low-rank reward modeling) — strongest personalization baseline
- **Monolithic BT** — single reward model baseline
- **Persona prompting** — simple persona description baseline
- **PAL** — alternative personalization approach

### 3. Evaluation Metrics
- **Preference prediction accuracy** on held-out data
- **Effective dimensionality** via eigenvalue analysis of preference/activation matrices
- **Cross-model cosine similarity** of discovered person-space dimensions
- **Congruence score** — how well LLMs adopt assigned preference profiles (measured behaviorally)

### 4. Code to Adapt/Reuse
- **repeng** for extracting preference/persona steering vectors
- **CAA** for contrastive activation methodology
- **PRISM analysis code** for data loading and preprocessing
- **pluralistic-alignment** for PAL baseline comparison

### 5. Proposed Experimental Pipeline
1. Design a set of simple like/dislike probes (e.g., "Do you prefer X or Y?") spanning diverse domains
2. Present probes to multiple LLMs and collect response patterns
3. Apply SVD/PCA to the preference response matrix to determine effective dimensionality
4. Extract activation-space representations using repeng/CAA methodology
5. Test whether low-rank preference structure predicts held-out preference responses
6. Compare person-space structure across different LLMs
7. Validate against PRISM real-user preference data
