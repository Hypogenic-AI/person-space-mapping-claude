# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: PRISM Alignment (PRIMARY)

### Overview
- **Source**: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
- **Paper**: Kirk et al., "The PRISM Alignment Project" (NeurIPS 2024, arXiv:2404.16019)
- **Size**: 77,882 total rows (1,500 survey, 8,011 conversations, 68,371 utterances), ~133MB
- **Format**: HuggingFace Dataset (3 configs: survey, conversations, utterances)
- **Task**: Individual preference analysis, person-space mapping
- **License**: CC-BY-4.0

### Why This Dataset
Most directly relevant dataset available. Contains per-user preference data with rich demographic profiles linked to multi-dimensional LLM interaction ratings. 1,500 participants from 75 countries interacting with 21 LLMs. Enables correlating stated personality/preferences with revealed behavioral preferences.

### Key Fields
- **Survey**: user_id, demographics (age, gender, education, religion, ethnicity, country), stated preferences, LM familiarity
- **Conversations**: conversation_id, user_id, conversation_type, opening_prompt, performance_attributes, choice_attributes
- **Utterances**: utterance_id, user_id, conversation_id, turn, scores (1-100), 7 fine-grained attributes (values, fluency, factuality, safety, diversity, creativity, helpfulness)

### Download Instructions

```python
from datasets import load_dataset

# Download all splits
for config in ['survey', 'conversations', 'utterances']:
    ds = load_dataset("HannahRoseKirk/prism-alignment", config)
    ds.save_to_disk(f"datasets/prism/{config}")
```

### Loading the Dataset

```python
from datasets import load_from_disk

survey = load_from_disk("datasets/prism/survey")['train']
conversations = load_from_disk("datasets/prism/conversations")['train']
utterances = load_from_disk("datasets/prism/utterances")['train']
```

---

## Dataset 2: PersonalLLM Benchmark

### Overview
- **Source**: https://huggingface.co/datasets/namkoong-lab/PersonalLLM
- **Paper**: Zollo et al., "PersonalLLM" (arXiv:2409.20296)
- **Size**: 10,402 prompts (9,402 train, 1,000 test), ~50MB
- **Format**: HuggingFace Dataset
- **Task**: Personalized preference prediction
- **Splits**: train (9,402), test (1,000)

### Why This Dataset
Controlled benchmark with 10K prompts × 8 LLM responses × 10 reward model scores. Enables studying preference diversity and low-rank structure in a clean setting. Synthetic users generated via Dirichlet-weighted combinations of reward models.

### Key Fields
- prompt, prompt_id, subset (15 categories)
- response_1 through response_8 (from GPT-4o, Claude 3 Opus, Gemini 1.5 Pro, etc.)
- response_1_model through response_8_model
- score_1_1 through score_8_10 (8 responses × 10 reward models = 80 scores)

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("namkoong-lab/PersonalLLM")
ds.save_to_disk("datasets/personalllm/data")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/personalllm/data")
train = ds['train']
test = ds['test']
```

---

## Dataset 3: Anthropic LLM Global Opinions

### Overview
- **Source**: https://huggingface.co/datasets/Anthropic/llm_global_opinions
- **Size**: 2,556 survey questions, ~1MB
- **Format**: HuggingFace Dataset
- **Task**: Cross-cultural opinion analysis reference
- **License**: Apache 2.0

### Why This Dataset
Reference dataset for understanding which global opinions LLMs reflect. Contains country-level response distributions from World Values Survey and Pew Global Attitudes Survey. Useful for calibrating person-space against known cultural variation.

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("Anthropic/llm_global_opinions")
ds.save_to_disk("datasets/global_opinions/data")
```

### Loading the Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/global_opinions/data")['train']
```

---

## Notes

- All datasets are downloaded locally in `datasets/` but excluded from git via `.gitignore`
- Sample data for reference is in `*/samples/` directories (these ARE committed)
- The PRISM dataset is the primary recommendation for experiments
- PersonalLLM provides a controlled environment for methodology development
