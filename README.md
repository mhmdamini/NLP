
---

# Morphological MCQ Generation for Grades 3–5

This repository supports the automatic generation and evaluation of morphologically focused multiple-choice questions (MCQs) for students in grades 3–5. It is based on the research presented in the paper:

> **Prompting Strategies for Language Model-Based Item Generation in K–12 Education: Bridging the Gap Between Small and Large Language Models**  
> Mohammad Amini, Babak Ahmadi, et al., University of Florida

---

## 🚀 Project Overview

This project aims to reduce manual effort in test creation by leveraging language models to automatically generate and evaluate MCQs targeting **prefixes, suffixes, root words, word transformations, and affix-based definitions**. It focuses on:

- Fine-tuning a mid-sized model (Gemma 2B) for morphological item generation
- Comparing model output to GPT-3.5 (zero-shot and structured prompts)
- Supporting **13 morphologically informed question types**
- Evaluating output using both automated NLP metrics and human-aligned scoring

> ⚠️ **Note**: The original dataset (WordChomp) used for training and testing is **not publicly available** due to licensing restrictions.

---

## 📂 Repository Structure

```
├── gpt.py                               # Prompting strategies using GPT-3.5
├── gemma.py                             # Prompting strategies using fine-tuned Gemma model
├── fine_tuning_gemma.py                 # Training pipeline for Gemma model using LoRA adapters
├── auto_eval.py                         # Automatic evaluation: grammar, complexity, readability, and fluency
├── parser_for_human_expert_evals.py     # Parses human expert docx annotations into CSV format
├──IPNYB/                                #  Jupyter notebooks files
├──results/                              # Output files from evaluation scripts in Excel format
└── expert-based_evaluations_by_gpt41.py  # GPT-4.1 based structured scoring of questions
```

---

## 🔧 Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

Required packages include:
- `transformers`, `torch`, `spacy`, `language-tool-python`
- `openai`, `nltk`, `pandas`, `tqdm`, `openpyxl`
- `peft` (for LoRA-based fine-tuning)
- `huggingface_hub` (for downloading configs)

---

## 🧠 Prompting Strategies

This repo supports multiple LLM prompting styles, including:

- **Zero-shot / Few-shot**
- **Chain-of-thought (CoT)**
- **CoT + Sequential Reasoning**
- **CoT + Role-Conditioning**
- **CoT + Multi-step Real-time prompting (RL)**

These strategies are abstracted in `gpt.py` and `gemma.py`.

---

## 📈 Evaluation Metrics

Implemented in `auto_eval.py`:

- **Grammar**: LanguageTool error density
- **Complexity**: Syntax tree depth, word variety
- **Readability**: Flesch Reading Ease, Gunning Fog
- **Fluency**: GPT-2 perplexity

Each item receives a composite quality score based on these metrics.

---


## 📄 File Descriptions

### `gpt.py`
**Purpose:**  
Implements prompting strategies using **OpenAI’s GPT-3.5 or GPT-4** models for generating morphologically-aware multiple-choice questions (MCQs).

**Key Features:**
- Supports **13 morphological question types**, such as identifying prefixes, suffixes, root words, and transformations.
- Implements five prompting strategies:
  - Zero-shot
  - Few-shot (with real examples)
  - Chain-of-thought (CoT)
  - CoT + Sequential Reasoning
  - CoT + Multi-role (Teacher → Student → Psychometrician)
- Handles word difficulty, task difficulty, and reuse prevention (e.g., "forbidden word list").

### `gemma.py`
**Purpose:**  
Provides an inference interface for a **fine-tuned Gemma model**, replicating the logic in `gpt.py` but using a local (LoRA-adapted) model rather than the OpenAI API.

**Key Features:**
- Uses Hugging Face's `transformers` and `peft` for model loading and prompt generation.
- All five prompting strategies from `gpt.py` are replicated.
- Adds a `generate_gpt()` wrapper that interfaces with the local Gemma model instead of OpenAI's API.
- Requires a valid `HF_TOKEN` and access to a pre-fine-tuned model.

### `fine_tuning_gemma.py`
**Purpose:**  
Handles **data preparation, augmentation, training, and saving** for fine-tuning the Gemma 2B model on your morphological dataset.

**Key Features:**
- Custom `MorphologyDataset` class for formatting question data (both MCQ and open-ended).
- Augments input data with phrasing variations to improve generalization.
- Uses Hugging Face's `Trainer` and `LoRA` adapters for memory-efficient training.
- Includes GPU monitoring, stratified data splitting, and model-saving logic.

### `auto_eval.py`
**Purpose:**  
Performs **automatic evaluation** of generated questions using standard NLP metrics. This helps identify the quality of LLM outputs across multiple dimensions.

**Evaluated Dimensions:**
- **Grammar Quality** – via LanguageTool (error density)
- **Syntactic Complexity** – using spaCy parse trees and POS distributions
- **Readability** – via Flesch Reading Ease and Gunning Fog Index
- **Fluency** – using GPT-2 perplexity as a proxy

**Pipeline:**
- Input: Excel file of generated questions
- Output: Annotated Excel file with quality scores and detailed error messages

---

### 📄 `expert-based_evaluations_by_gpt41.py`

**Purpose:**  
This script simulates expert evaluation of generated multiple-choice questions by leveraging **GPT-4.1** in a structured, rubric-based fashion.

**What it does:**
- Takes a CSV of auto-generated questions (e.g., from GPT-3.5 or Gemma).
- For each question, GPT-4.1 is prompted to evaluate the item across **five specific educational criteria**, modeled after human review:
  1. **Clarity of Instruction**
  2. **Accuracy of the Correct Answer**
  3. **Quality of Distractors**
  4. **Appropriateness of Word Difficulty**
  5. **Alignment with Task Difficulty**
- GPT responds in **structured JSON format**, including both binary scores (0 or 1) and explanations.
- The scores and rationale are saved to a CSV for analysis.

**How it's used:**
- Replace `ITEM_FILE` with the path to your generated questions.
- Ensure corresponding human evaluation examples exist (loaded from `HUMAN_FILE`) to prime GPT.
- Run the async `main()` loop to score all items in parallel.

**Benefits:**
- Automates expert-level review at scale.
- Ensures interpretability via rationale for each score.

---

### 📄 `parser_for_human_expert_evals.py`

**Purpose:**  
Parses human evaluations stored in a `.docx` file and converts them into a structured CSV format usable by other tools (e.g., as priming examples in `expert-based_evaluations_by_gpt41.py`).

**What it does:**
- Reads `Human Evaluation Metrics.docx`, where each section includes:
  - A sample question (with choices, correct answer)
  - Morphological difficulty tags
  - Human-assigned scores (0/1) for each of the 5 evaluation metrics
  - Justifying explanations for each score
- Extracts each item using regular expressions.
- Converts the result into `mcq_human_evals.csv`.

**Why it's important:**
- Provides a high-quality, human-labeled set of examples that teach GPT how to evaluate items realistically.
- Forms the foundation for **few-shot priming** inside the GPT-4.1-based rubric evaluator.

---


## 🔁 Model Training

Use `fine_tuning_gemma.py` to:

- Preprocess data and augment question phrasing
- Fine-tune the **Gemma 2B model** using LoRA
- Save adapters and tokenizer for inference

---

## ✏️ Example Use

Generate a prefix-based question (type QT1) using Gemma:
```python
from gemma import generate_prefix_prompt, load_gemma_model

model_path = "path/to/your/fine-tuned/model"
model, tokenizer = load_gemma_model(model_path)

prompt, word = generate_prefix_prompt(
    word_difficulty=3,
    task_difficulty=2,
    data=your_df,
    prompting='chain_of_thought_plus_sequential_rl',
    words=[],
    model=model,
    tokenizer=tokenizer
)
print(prompt)
```

---

## 📜 Citation

If you use this codebase in your research, please cite the paper:
```bibtex
@inproceedings{amini2025morphmcq,
  title={Prompting Strategies for Language Model-Based Item Generation in K–12 Education},
  author={Mohammad Amini and Babak Ahmadi and others},
  year={2025},
  organization={University of Florida}
}
```
---

## 📬 Contact

For questions, feedback, or collaboration inquiries, please contact:

- **Mohammad Amini** – m.amini@ufl.edu  
- **Babak Ahmadi** – babak.ahmadi@ufl.edu  
- Department of Industrial & Systems Engineering, University of Florida

---

💡 If it’s more convenient for you to work in notebooks, we’ve also provided interactive `.ipynb` versions of key scripts in the `IPNYB` folder.
