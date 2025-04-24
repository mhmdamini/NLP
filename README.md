Here's a draft `README.md` tailored for your GitHub repository, based on the uploaded files and project description from your paper:

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
├── gpt.py                   # Prompting strategies using GPT-3.5
├── gemma.py                # Prompting strategies using fine-tuned Gemma model
├── fine_tuning_gemma.py    # Training pipeline for Gemma model using LoRA adapters
├── auto_veal.py            # Automatic evaluation: grammar, complexity, readability, and fluency
├── Final_Auto_Eval.ipynb   # Notebook for evaluating generated items (GPT-based)
├── Untitled1.ipynb         # Additional experimental notebook (in development)
├── Untitled.ipynb          # Additional experimental notebook (in development)
└── CAI6307_Team_10_EduGen_Innovators.pdf  # Project paper
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

Implemented in `auto_veal.py`:

- **Grammar**: LanguageTool error density
- **Complexity**: Syntax tree depth, word variety
- **Readability**: Flesch Reading Ease, Gunning Fog
- **Fluency**: GPT-2 perplexity

Each item receives a composite quality score based on these metrics.

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
