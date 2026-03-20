# Cloned Repositories

## Repo 1: repeng (Representation Engineering Control Vectors)
- **URL**: https://github.com/vgel/repeng
- **Stars**: 698
- **Location**: `code/repeng/`
- **Purpose**: Practical library for creating and applying RepE control vectors to transformer models. Can extract steering vectors from contrastive prompts.
- **Key files**: `repeng/` (library), examples/notebooks
- **How to use**: `pip install repeng` or use from source. Create control vectors by specifying positive/negative prompt pairs, then apply to generation.
- **Relevance**: Core tool for extracting persona/preference steering vectors from contrastive like/dislike prompts.

## Repo 2: CAA (Contrastive Activation Addition)
- **URL**: https://github.com/nrimsky/CAA
- **Stars**: 216
- **Location**: `code/CAA/`
- **Purpose**: Implementation of steering Llama 2 with contrastive activation addition (ACL 2024). Extracts behavior-specific steering vectors from paired prompts.
- **Key files**: Jupyter notebooks demonstrating the full pipeline
- **Relevance**: Reference implementation of the contrastive method for extracting behavioral dimensions — directly applicable to extracting preference dimensions.

## Repo 3: PRISM Alignment
- **URL**: https://github.com/HannahKirk/prism-alignment
- **Stars**: 90
- **Location**: `code/prism-alignment/`
- **Purpose**: Code and analysis for the PRISM Alignment Project. Data loading, analysis scripts, and visualization.
- **Key files**: Analysis notebooks, data processing scripts
- **Relevance**: Data loading infrastructure for the PRISM dataset (our primary dataset). Understanding their analysis approach.

## Repo 4: Pluralistic Alignment (PAL)
- **URL**: https://github.com/RamyaLab/pluralistic-alignment
- **Stars**: 15
- **Location**: `code/pluralistic-alignment/`
- **Purpose**: PAL: Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment. Implements personalized reward models.
- **Key files**: Model implementations, training scripts
- **Relevance**: Baseline implementation for personalized reward modeling — comparison method for our low-rank person-space approach.

## Additional Relevant Repos (Not Cloned)

These are available for cloning if needed during experiments:

- **andyzoujm/representation-engineering** (965 stars): Foundational RepE framework
- **hjian42/PersonaLLM** (68 stars): Big Five personality expression in LLMs
- **kaustpradalab/LLM-Persona-Steering** (16 stars): Personality steering via latent features
- **steering-vectors/steering-vectors** (140 stars): General steering vectors library
- **IBM/activation-steering** (151 stars): Production-grade activation steering
- **LaMP-Benchmark/LaMP** (189 stars): Personalization benchmark
