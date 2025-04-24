#########################################################################
## Parse the human expert evaluation into a csv file:

"""
Parse “Human Evaluation Metrics.docx” and write mcq_human_evals.csv.

• One row per annotated sample.
• Captures question data, human scores, AND the explanatory
  sentences for each of the five evaluation dimensions.
"""

import re, csv, sys
from pathlib import Path
from collections import defaultdict
from docx import Document         


DOCX_FILE = "Human Evaluation Metrics.docx"
OUT_FILE  = "mcq_human_evals.csv"


# ---------------------------------------------------------
# 1. Pull non‑empty lines out of the .docx
# ---------------------------------------------------------
try:
    paragraphs = Document(DOCX_FILE).paragraphs
except Exception as e:
    sys.exit(f"Could not open {DOCX_FILE}: {e}")

lines = [p.text.strip() for p in paragraphs if p.text.strip()]

# ---------------------------------------------------------
# 2. regex patterns
# ---------------------------------------------------------
QT_HEADER   = re.compile(r"=+Question Type (\d+)=+")
SAMPLE_HDR  = re.compile(r"Sample #(\d+) for QT\d+", re.I)

QUESTION_RE = re.compile(r"^(?:Question:)?\s*(.*?\?)\s*$", re.I)
CHOICE_RE   = re.compile(r"^Choice\s*\d+\s*[:)]\s*(.+)$", re.I)
CORRECT_RE  = re.compile(r"^Correct[_ ]answer\s*[:)]\s*([A-D])\)\s*(.+)$", re.I)
WORD_RE     = re.compile(r"word[_ ]difficulty\s*[:)]\s*(\d+)", re.I)
TASK_RE     = re.compile(r"task[_ ]difficulty\s*[:)]\s*([EMH])", re.I)

EVAL_START  = re.compile(r"Evaluation for .*?Total Score:\s*(\d+)/5", re.I)
METRIC_RE   = re.compile(
    r"(Clarity of Instruction|Accuracy of Correct Answer|Quality of Distractors|"
    r"Word Difficulty|Task Difficulty)\s*\((\d)\)\s*:\s*(.+)$",
    re.I,
)

metric_cols = {
    "clarity of instruction":   ("eval_instruction_score",  "eval_instruction_exp"),
    "accuracy of correct answer":("eval_accuracy_score",    "eval_accuracy_exp"),
    "quality of distractors":   ("eval_distractors_score",  "eval_distractors_exp"),
    "word difficulty":          ("eval_word_diff_score",    "eval_word_diff_exp"),
    "task difficulty":          ("eval_task_diff_score",    "eval_task_diff_exp"),
}

# ---------------------------------------------------------
# 3. streaming parse
# ---------------------------------------------------------
records   = []
ctx       = defaultdict(lambda: None)

def flush():
    """push current ctx into records & reset"""
    if ctx.get("question"):
        # ensure every metric column exists
        for score_col, exp_col in metric_cols.values():
            ctx.setdefault(score_col, 0)
            ctx.setdefault(exp_col, "")
        records.append(ctx.copy())
        ctx.clear()

current_qt = None

for ln in lines:
    # ---- section headers ----
    if m := QT_HEADER.match(ln):
        flush()
        current_qt = int(m.group(1))
        continue
    if m := SAMPLE_HDR.match(ln):
        flush()
        ctx["question_type"] = current_qt
        ctx["sample_number"] = int(m.group(1))
        continue

    # ---- question / choices ----
    if m := QUESTION_RE.match(ln):
        ctx["question"] = m.group(1).strip()
        continue
    if m := CHOICE_RE.match(ln):
        key = f"choice_{len([k for k in ctx if k.startswith('choice_')]) + 1}"
        ctx[key] = m.group(1).strip()
        continue
    if m := CORRECT_RE.match(ln):
        ctx["correct_answer_letter"] = m.group(1)
        ctx["correct_answer_text"]   = m.group(2).strip()
        continue

    # ---- misc attributes ----
    if m := WORD_RE.search(ln):
        ctx["word_difficulty"] = int(m.group(1))
    if m := TASK_RE.search(ln):
        ctx["task_difficulty"] = m.group(1).upper()

    # ---- evaluation block ----
    if m := EVAL_START.match(ln):
        ctx["eval_total_score"] = int(m.group(1))
        # seed blank metric fields
        for score_col, exp_col in metric_cols.values():
            ctx[score_col] = 0
            ctx[exp_col]   = ""
        continue

    # ---- individual metric lines ----
    if m := METRIC_RE.match(ln):
        label      = m.group(1).lower()
        score      = int(m.group(2))
        explanation= m.group(3).strip()
        score_col, exp_col = metric_cols[label]
        ctx[score_col] = score
        ctx[exp_col]   = explanation
        continue

# final sample
flush()

# ---------------------------------------------------------
# 4. normalise choice columns
# ---------------------------------------------------------
max_choices = max(int(k.split("_")[1])
                  for r in records for k in r if k.startswith("choice_"))
for r in records:
    for i in range(1, max_choices + 1):
        r.setdefault(f"choice_{i}", "")

# ---------------------------------------------------------
# 5. write csv
# ---------------------------------------------------------
base_cols = ["question_type", "sample_number", "question"] + \
            [f"choice_{i}" for i in range(1, max_choices + 1)] + \
            ["correct_answer_letter", "correct_answer_text",
             "word_difficulty", "task_difficulty", "eval_total_score"]

metric_cols_flat = []
for score_col, exp_col in metric_cols.values():
    metric_cols_flat.extend([score_col, exp_col])

fieldnames = base_cols + metric_cols_flat

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    csv.DictWriter(f, fieldnames=fieldnames).writerows(records)

print(f"Wrote {len(records)} rows → {OUT_FILE}")

