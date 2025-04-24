# %% FILTERS I care about
#     – leave key absent or empty list to keep *all* rows for that column
# FILTERS = {
#     "prompting_strategy": ["chain_of_thought_plus_role_chain"],
#     "question_type":      [2],
#     "word_difficulty":    ["1"],
#     "task_difficulty":    ["E"],          # 'E' | 'M' | 'H'
# }
FILTERS = {}

# ═══════════════════════════════════════════════════════════════
#  Imports / config
# ═══════════════════════════════════════════════════════════════
import os, json, math, asyncio, pandas as pd, openai
from tqdm.notebook import tqdm
# from dotenv import load_dotenv

HUMAN_FILE = "mcq_human_evals.csv"

## A) if gpt-3.5's outputs need to be evaluated:
ITEM_FILE  = "Final_generated_questions_GPT35_clean_UPDATED_new.csv"
OUT_FILE   = "gpt_41_mini_evaluations_of_gpt_35_UPDATED_new.csv"

## B) if gemma's outputs need to be evaluated:
# ITEM_FILE  = "Final_generated_questions_Gemma.csv"
# OUT_FILE   = "gpt_41_mini_evaluations_of_gemma.csv"


MODEL       = "gpt-4.1-mini"
TEMPERATURE = 0.0
CONCURRENT  = 10          # parallel requests

items_df = pd.read_csv(ITEM_FILE,  dtype=str)
human_df = pd.read_csv(HUMAN_FILE, dtype=str)

human_df["question_type"] = human_df["question_type"].astype(int)
items_df["question_type"] = items_df["question_type"].astype(int)

for col, allowed in FILTERS.items():
    if allowed:                                   # skip empty lists / None
        items_df = items_df[items_df[col].isin([str(x) for x in allowed])]

print("Items after filtering:", len(items_df))

def fmt_human_example(row) -> str:
    """Return one nicely formatted human‑rated example, skipping NaN/empty choices."""
    choice_lines = []
    for i in range(1, 5):
        val = row.get(f"choice_{i}", "")
        # skip if val is NaN or empty
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        txt = str(val).strip()
        if not txt or txt.lower() == "nan":
            continue
        choice_lines.append(f"{chr(64+i)}) {txt}")

    choices = "\n".join(choice_lines)

    return (
        "### Human‑rated Example\n"
        f"Question: {row['question']}\n{choices}\n"
        f"Correct Answer: {row['correct_answer_text']}\n"
        f"word_difficulty={row['word_difficulty']} "
        f"task_difficulty={row['task_difficulty']}\n"
        "Scores:\n"
        f"- Instruction Clarity: {row['eval_instruction_score']} ({row['eval_instruction_exp']})\n"
        f"- Accuracy of Correct Answer: {row['eval_accuracy_score']} ({row['eval_accuracy_exp']})\n"
        f"- Quality of Distractors: {row['eval_distractors_score']} ({row['eval_distractors_exp']})\n"
        f"- Word Difficulty Appropriateness: {row['eval_word_diff_score']} ({row['eval_word_diff_exp']})\n"
        f"- Task Difficulty Alignment: {row['eval_task_diff_score']} ({row['eval_task_diff_exp']})\n"
        "### End Example\n"
    )

EXAMPLE_BLOCKS = {
    q: "\n".join(fmt_human_example(r) for _, r in g.iterrows())
    for q, g in human_df.groupby("question_type")
}

RUBRIC = """
Rate on five binary metrics (1 = meets, 0 = does not) and reply JSON only:
{
 "instr_score":0/1, "instr_exp":"...",
 "acc_score":0/1,   "acc_exp":"...",
 "dist_score":0/1,  "dist_exp":"...",
 "word_score":0/1,  "word_exp":"...",
 "task_score":0/1,  "task_exp":"...",
 "total_score":0-5
}
"""

def append_row(row_dict: dict, path: str):
    """Append one row to CSV; create file+header on first call."""
    first_write = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=row_dict.keys())
        if first_write:
            writer.writeheader()
        writer.writerow(row_dict)


sem = asyncio.Semaphore(CONCURRENT)

async def rate_item(idx, row):
    choice_a = str(row.get("choice_a", "") or "").strip()
    choice_b = str(row.get("choice_b", "") or "").strip()
    choice_c = str(row.get("choice_c", "") or "").strip()

    user_prompt = (
        f"{EXAMPLE_BLOCKS[row['question_type']]}\n"
        "---------------------------------\n"
        "### Item to evaluate\n"
        f"Question: {row['question']}\n"
        f"A) {choice_a}\n"
        f"B) {choice_b}\n"
        f"C) {choice_c}\n"
        f"Correct Answer: {row['correct_answer']}\n"
        f"word_difficulty={row['word_difficulty']} "
        f"task_difficulty={row['task_difficulty']}\n\n"
        f"{RUBRIC}"
    )
    messages = [
        {"role": "system",
         "content": "You are a meticulous K‑12 morphology test reviewer."},
        {"role": "user", "content": user_prompt}
    ]

    async with sem:
        try:
            rsp  = await openai.ChatCompletion.acreate(
                model=MODEL, temperature=TEMPERATURE, messages=messages
            )
            data = json.loads(rsp.choices[0].message["content"])
        except Exception as e:
            print(f"[warn] row {idx} failed: {e}")
            data = {k: None for k in
                    ["instr_score","instr_exp","acc_score","acc_exp",
                     "dist_score","dist_exp","word_score","word_exp",
                     "task_score","task_exp","total_score"]}

    merged = row.to_dict()
    merged.update(data)
    append_row(merged, OUT_FILE)      
    return merged

async def main():
    tasks = [asyncio.create_task(rate_item(i, r))
             for i, r in items_df.iterrows()]

    # live progress bar as rows finish (order is non‑deterministic)
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f

    print(f"✓ Completed – streamed to {OUT_FILE}")

await main()