# Research Plan: Mapping Person Space Through Simple Likes and Dislikes

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used in personalized applications where they must adopt different personas. Understanding the underlying structure of "person space" — the set of coherent persona configurations an LLM can adopt — is critical for personalization, alignment, and safety. If this space is low-rank, it means LLM personas are constrained to a small number of underlying dimensions, which has profound implications for how we think about AI personality and preference alignment.

### Gap in Existing Work
Based on the literature review, existing work has:
- Shown that steering vectors for personality traits are entangled and non-orthogonal (Bhandari et al., 2026)
- Demonstrated low-rank structure in preference reward models (Bose et al., 2025 - LoRe)
- Found that persona vectors exist as linear directions in activation space (Chen et al., 2025)

**But no study has**: (1) Directly probed LLMs with simple everyday likes/dislikes to map persona space, (2) measured which preference combinations feel "congruent" vs. "incongruent" to LLMs, (3) compared this structure across different LLM families, or (4) estimated the effective dimensionality of person space from behavioral (not activation-level) data.

### Our Novel Contribution
We propose a purely behavioral approach: present LLMs with persona profiles defined by sets of likes/dislikes, measure how consistently and naturally they adopt each profile, and use the resulting congruence matrix to map the effective dimensionality of person space. This is complementary to activation-based approaches (which require model internals) and works across any LLM via API.

### Experiment Justification
- **Experiment 1 (Preference Congruence Probing):** Needed to establish which preference combinations LLMs find natural vs. incongruent. This is the core data collection.
- **Experiment 2 (Dimensionality Analysis):** Needed to test the low-rank hypothesis by applying SVD/PCA to the congruence matrix.
- **Experiment 3 (Cross-Model Comparison):** Needed to test whether person space structure is universal across LLM families.

## Research Question
Can the space of LLM personas be mapped by probing which combinations of everyday likes and dislikes LLMs adopt congruently, and is this space low-rank and consistent across different LLMs?

## Hypothesis Decomposition

**H1:** LLMs will adopt some preference combinations more congruently than others (not all combinations are equally natural).

**H2:** The preference congruence matrix has low effective rank (few principal components explain most variance).

**H3:** The discovered person-space dimensions correspond to interpretable persona types (e.g., "intellectual vs. practical", "adventurous vs. homebody").

**H4:** Different LLMs (GPT-4.1, Claude, Gemini) share similar person-space structure (high correlation between their congruence matrices).

## Proposed Methodology

### Approach
1. **Design preference items**: Create ~60 preference items across 10 domains (food, music, hobbies, social, aesthetics, values, entertainment, travel, work style, lifestyle).
2. **Create persona profiles**: Generate ~100 persona profiles as random subsets of 8-10 preferences each.
3. **Measure congruence**: For each profile, ask the LLM to adopt it, then probe with follow-up questions to measure behavioral consistency. Congruence = how consistently the LLM maintains and extends the assigned preferences.
4. **Build preference co-occurrence matrix**: Which preferences tend to be adopted congruently together?
5. **Dimensionality analysis**: SVD/PCA on the congruence matrix.
6. **Cross-model comparison**: Repeat with multiple LLMs, compare structure.

### Experimental Steps

**Step 1: Preference Item Design**
- Create 60 preference items as binary like/dislike across 10 domains
- Each item is a concrete, everyday preference (e.g., "likes spicy food", "prefers quiet evenings")

**Step 2: Persona Profile Generation**
- Generate 100 random profiles of 8 preferences each
- Include some "stereotype-consistent" profiles and some "mixed" profiles

**Step 3: Congruence Measurement**
For each (LLM, profile) pair:
1. System prompt assigns the persona with the 8 preferences
2. Ask 5 follow-up questions that probe preference consistency
3. Score each response for alignment with assigned preferences (0-1)
4. Average score = congruence score for that profile

**Step 4: Build Co-occurrence Matrix**
- For each pair of preferences (i, j), compute: how much does including both in a profile affect congruence?
- This gives a 60×60 preference interaction matrix

**Step 5: Dimensionality Analysis**
- SVD of the interaction matrix
- Scree plot to determine effective rank
- Interpret top components

**Step 6: Cross-Model Comparison**
- Repeat Steps 3-5 for GPT-4.1 and at least one other model
- Compute Procrustes alignment between person spaces
- Report structural similarity

### Baselines
- **Random baseline**: Expected congruence if preferences were independent
- **Stereotype baseline**: Congruence for hand-crafted "coherent" vs. "incoherent" profiles

### Evaluation Metrics
- **Congruence score**: Average consistency of LLM responses with assigned preferences (0-1)
- **Effective dimensionality**: Number of SVD components needed to explain 80% and 90% of variance
- **Cross-model correlation**: Pearson/Spearman correlation between congruence matrices from different LLMs
- **Interpretability**: Whether top PCA components have clear semantic meaning

### Statistical Analysis Plan
- Permutation tests for significance of low-rank structure (compare to random preference matrices)
- Bootstrap confidence intervals for effective dimensionality
- Mantel test for cross-model matrix correlation
- Significance level: α = 0.05

## Expected Outcomes
- **Supporting H1**: Significant variance in congruence scores across profiles (some much higher/lower than mean)
- **Supporting H2**: Effective rank of 5-10 dimensions (out of 60) explaining >80% variance
- **Supporting H3**: Top components have interpretable labels
- **Supporting H4**: Cross-model correlation r > 0.5

## Timeline and Milestones
1. Environment setup + preference design: 15 min
2. API infrastructure + congruence measurement code: 30 min
3. Run primary experiments (GPT-4.1): 45 min
4. Run comparison experiments (second model): 30 min
5. Analysis + visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
- **API rate limits**: Mitigate with exponential backoff and batching
- **Congruence measurement noise**: Use multiple probe questions per profile, average scores
- **Cost**: ~100 profiles × 5 questions × 2 models = ~1000 API calls. At ~$0.01/call ≈ $10
- **Judge reliability**: Use structured scoring rubric, validate with examples

## Success Criteria
1. Clear evidence of non-uniform congruence (variance significantly above random)
2. Identifiable low-rank structure (scree plot shows elbow)
3. At least 2 interpretable dimensions
4. Cross-model comparison completed with quantitative similarity metric
