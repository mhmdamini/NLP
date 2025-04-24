HF_TOKEN = "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"
import os
import pandas as pd
import torch
import gc
import numpy as np
import json
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import pandas as pd
import numpy as np
import random
import openai
import os
import re


###############################################################################
# Load pretrained model first word utility
###############################################################################

def load_gemma_model(model_path, base_model_name="last_gemma_morphology_20250404_014910/final_model", use_lora=True):
    """Load the trained model and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float32
    )

    if use_lora:
        print("Loading LoRA adapters...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


###############################################################################
# 1. Configure OpenAI
###############################################################################
# Replace this with a safer method (e.g., environment variable) in production
# openai.api_key = "YOUR_OPENAI_API_KEY"

def test_api_access():
    """
    Attempts to list OpenAI models to confirm that the API key is valid.
    Prints a success or failure message, along with a list of available models if successful.
    """
    try:
        models = openai.Model.list()
        # print("Access to OpenAI API successful! Available models:")
        # for model in models['data']:
        #     print(f" - {model['id']}")
    except Exception as e:
        print("Failed to access the OpenAI API:")
        print(e)

###############################################################################
# 2. Core LLM function (using OpenAI GPT models)
###############################################################################

def load_gemma_model(model_path, base_model_name="gemma_morphology/final_model", use_lora=True):
    """Load the trained model and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float32
    )

    if use_lora:
        print("Loading LoRA adapters...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer

def generate_gpt(prompt, model, tokenizer, max_length=512):
    """
    Generate text using the pre-trained Gemma model
    """
    try:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generation parameters
        generation_config = {
            'max_new_tokens': 200,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.92,
            'top_k': 50,
            'repetition_penalty': 1.2,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )

        # Decode and return
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the generated text
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    except Exception as e:
        print(f"Error in generate_gemma: {e}")
        return None
    #blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/MC_data_MMA.csv
    
model_path="/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/last_gemma_morphology_20250403_172610/final_model"
#blue/babajani_directory/babak.ahmadi/NLP_Dorr/Project/MA/gemma_morphology_20250403_164502/final_model"
    #model_path="/blue/cai6307/EduGen/gemma_few_shot_20250327_162345/final_model"
    #model_path = "gemma_morphology_20250402_025445/final_model"  # Update this path
model, tokenizer = load_gemma_model(model_path)

def generate_gpat(prompt, model_name="gpt-3.5-turbo"):
    """
    Sends a prompt to the OpenAI GPT (ChatCompletion) and returns the response text.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"An error occurred with the GPT call: {e}")
        return None

###############################################################################
# 3. Extract first word utility
###############################################################################
def extract_word_from_response(response_text):
    """
    Extracts and returns the first word-like token from the response_text.
    Returns None if no valid word is found or if response_text is None.
    """
    if response_text is None:
        return None
    cleaned_text = re.sub(r"[^\w'-]", " ", response_text)
    word = re.search(r'\b\w+\b', cleaned_text)
    return word.group(0) if word else None

###############################################################################
# 4. Few-shot example formatter
###############################################################################
def prompt_few_shot(data, Question_Type, num_examples):
    """
    Filters the dataset by Question_Type, samples up to num_examples,
    and formats them as example references to be appended to the main prompt.
    """
    filtered_df = data[data.iloc[:, 0] == Question_Type]
    if filtered_df.empty:
        return ""
    else:
        examples = filtered_df.sample(n=min(num_examples, len(filtered_df)))
        formatted_examples = "There are few examples, please do not use them on the generated questions. \n"
        for _, row in examples.iterrows():
            correct_choice = row[f"Choice_{row['Correct_Answer']}"]
            formatted_examples += f"For Example:\n"
            formatted_examples += f"Question: {row['Instruction']}\n"
            formatted_examples += f"A) {row['Choice_1']}\n"
            formatted_examples += f"B) {row['Choice_2']}\n"
            formatted_examples += f"C) {row['Choice_3']}\n"
            formatted_examples += f"Correct Answer: {correct_choice}\n"
            formatted_examples += (
                f"Explanation: Task difficulty of this question is {row['Task_Difficulty']}, "
                f"and word difficulty of this question is {row['Word_Difficulty']}\n\n"
                f"This is few_shot examples, generate different questions from these examples\n\n"
            )
        return formatted_examples

###############################################################################
# 5. Chain-of-thought (single-prompt)
###############################################################################
def prompt_chain_of_thought():
    """
    Appends a general chain-of-thought instruction.
    No CSV examples are used—just an instruction telling the model
    to 'think aloud' before finalizing the question.
    """
    # Ensuring we keep the final question in a parseable format
    chain_instruction = f"""
--- Chain of Thought ---
Please think aloud, and provide your reasoning before providing the final 3-choice question.
Include your reasoning in the final output as well.

Finally, PRESENT the final question in this format:
Question: [your question]
A) [option A]
B) [option B]
C) [option C]
Correct Answer: [the correct choice]
"""
    return chain_instruction

###############################################################################
# 6A. "Fake" single-prompt chain_of_thought_plus_sequential
###############################################################################
def prompt_chain_of_thought_plus_sequential(question_type, word_difficulty, task_difficulty):
    """
    A multi-step chain-of-thought approach. The final prompt instructs the model
    to create the MCQ in 3 steps (selecting words, drafting a question, adding distractors),
    each time showing its chain-of-thought reasoning.
    """
    multi_step_instructions = f"""
--- Chain of Thought + Sequential Steps ---

We want a 3-choice question for question_type={question_type}.
Word difficulty = {word_difficulty}, Task difficulty = {task_difficulty}.

Please follow these steps in your final output (all in one go):

Step 1: List three suitable Grade 3-5 words that illustrate the morphological concept
         (prefix, suffix, root, etc. depending on question_type).
         Show reasoning why each word is appropriate.
         Then select exactly ONE of them.

Step 2: Using the single selected word, generate a DRAFT 3-choice question.
        Provide chain-of-thought: i.e., explain your reasoning for how the question is framed.

Step 3: Add one correct answer choice and two distractors.
        Provide chain-of-thought for how each distractor might trick the student,
        and confirm which is correct.

Finally, PRESENT the final question in this format:
Question: [your question]
A) [option A]
B) [option B]
C) [option C]
Correct Answer: [the correct choice]

Be explicit with your chain-of-thought reasoning for each step,
but ensure the final output ends with the standard question format shown above.
    """
    return multi_step_instructions


def parse_chosen_word(response_text):
    """
    Try to parse the chosen word from the multi-step response text.
    We look for specific cues like 'Final word:' or 'Chosen word:' or
    text like 'I would choose "XYZ".'
    If none of these are found, we fall back to a naive approach.
    """
    if not response_text:
        return None

    # 0) Look for something like: Final word choice: XYZ
    match = re.search(r'(?i)final word choice:\s*([A-Za-z\'\-]+)', response_text)
    if match:
        return match.group(1)    
        
    # 1) Look for something like: Final word: XYZ
    match = re.search(r'(?i)final word:\s*([A-Za-z\'\-]+)', response_text)
    if match:
        return match.group(1)

    # 2) Look for something like: Chosen word: XYZ
    match = re.search(r'(?i)chosen word:\s*([A-Za-z\'\-]+)', response_text)
    if match:
        return match.group(1)

    # 3) Look for: I would choose "XYZ"
    match = re.search(r'(?i)i would choose\s+"([^"]+)"', response_text)
    if match:
        return match.group(1)

    # 4) Look for: I choose "XYZ"
    match = re.search(r'(?i)i choose\s+"([^"]+)"', response_text)
    if match:
        return match.group(1)

    # 5) As a last resort, do a naive approach: first alphabetic word.
    cleaned_text = re.sub(r"[^\w'-]", " ", response_text)
    first_word = re.search(r'\b[a-zA-Z\'\-]+\b', cleaned_text)
    return first_word.group(0) if first_word else None


###############################################################################
# 6B. REAL multi-step approach: chain_of_thought_plus_sequential_rl
###############################################################################
def prompt_chain_of_thought_plus_sequential_rl(question_type, word_difficulty, task_difficulty,model, tokenizer,forbidden_list=None, **kwargs):
    """
    TRUE multi-step approach that calls GPT multiple times.

    Step 0 (New): We prepend an instruction prompt based on question_type.
    Step 1: We ask GPT for 3 suitable words + chain-of-thought, then parse out
            the chosen word.
    Step 2: We feed that chosen word to GPT, ask for a draft 3-choice question
            (with chain-of-thought).
    Step 3: We ask GPT to add distractors & finalize the parseable question.

    Returns (final_text, the_word).
    """
        
    word_list = kwargs.get("word_list", None)
    if question_type in [4, 5] and word_list is None:
        raise ValueError("word_list is required for question_type 4 and 5")


    if forbidden_list is None:
        forbidden_list = []
        
    # ---------------------------
    # Define all question prompts
    # ---------------------------
    prompt_instruction_qt1 = (
        f"We want to generate a 3-choice question for a student learning about prefixes. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to identify the prefix in a chosen word and provide "
        f"two incorrect choices along with the correct answer."
    )
    prompt_instruction_qt2 = (
        f"We want to generate a 3-choice question for a student learning about suffixes. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to identify the suffix in the chosen word "
        f"and provide two incorrect choices along with the correct answer. "
    )
    prompt_instruction_qt3 = (
        f"We want to generate a 3-choice question for a student learning about root words. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to identify the root word in the chosen word "
        f"and provide two incorrect choices along with the correct answer. "
    )
    prompt_instruction_qt4 = (
        f"We want to generate a 3-choice question for a student learning about morphemes. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to identify the word that does NOT share "
        f"the same prefix as the others from the given words {word_list}. "
    )
    prompt_instruction_qt5 = (
        f"We want to generate a 3-choice question for a student learning about morphemes. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to identify the word that does NOT share "
        f"the same suffix as the others from the given words {word_list}. "
    )
    prompt_instruction_qt6 = (
        f"We want to generate a 3-choice question for a student learning about word transformations. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to transform the chosen word to a new meaning, "
        f"with two incorrect choices and one correct answer. "
    )
    prompt_instruction_qt7 = (
        f"We want to generate a 3-choice question for a student learning about affixed words. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to select the correct meaning of the chosen word  "
        f"from three answer choices. "
    )
    # prompt_restriction used in qt8 is assumed defined elsewhere or can be set to an empty string if not needed.
    prompt_instruction_qt8 = (
        f"We want to generate a 3-choice question for a student learning about spelling based on morpheme meaning"
        f"{{prompt_restriction}} "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should include a word with a suffix, provide two misspelled variations "
        f"and one correct spelling."
    )
    prompt_instruction_qt9 = (
        f"We want to generate a 3-choice question for a student learning to break affixed words into parts. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to break the chosen word into its correct parts "
        f"(prefix, root, suffix) and provide two incorrect choices along with the correct answer."
    )
    prompt_instruction_qt10 = (
        f"We want to generate a 3-choice question for a student learning about prefixes. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to select the correct definition of the prefix "
        f"in the chosen word from three answer choices."
    )
    prompt_instruction_qt11 = (
        f"We want to generate a 3-choice question for a student learning about root words in affixed words. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to select the correct definition of the root word "
        f"in the chosen word from three answer choices."
    )
    prompt_instruction_qt12 = (
        f"We want to generate a 3-choice question for a student learning about suffixes. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to select the correct definition or function of the suffix "
        f"in the chosen word from three answer choices."
    )
    prompt_instruction_qt13 = (
        f"We want to generate a 3-choice question for a student learning about morphologically complex words. "
        f"The word difficulty must be {word_difficulty} and task difficulty must be {task_difficulty}. "
        f"Be informed that ultimately, the question should ask the student to select the correct definition of the chosen word "
        f"based on its morphemes."
    )

    # -------------------------------------------------------
    # Map question_type to the corresponding prompt instruction
    # -------------------------------------------------------
    instructions_map = {
        1: prompt_instruction_qt1,
        2: prompt_instruction_qt2,
        3: prompt_instruction_qt3,
        4: prompt_instruction_qt4,
        5: prompt_instruction_qt5,
        6: prompt_instruction_qt6,
        7: prompt_instruction_qt7,
        8: prompt_instruction_qt8,
        9: prompt_instruction_qt9,
        10: prompt_instruction_qt10,
        11: prompt_instruction_qt11,
        12: prompt_instruction_qt12,
        13: prompt_instruction_qt13
    }

    # ---------------------------
    # Retrieve the question prompt
    # ---------------------------
    question_prompt_instruction = instructions_map.get(
        question_type,
        f"[No prompt defined for question_type={question_type}]"
    )

    
    
    
    # ---------------------------
    # Step 1: Pick Words
    # ---------------------------
    # We prepend the relevant instruction prompt here
    if question_type not in [4, 5]:
        step1_prompt = f"""
    {question_prompt_instruction}

    Step 1 (Pick Words):
    Question Type = {question_type}
    Word Difficulty = {word_difficulty}, Task Difficulty = {task_difficulty}

    Please list three suitable Grade 3-5 words that fit the morphological concept
    (prefix, suffix, root, etc.) for this Question Type: {question_type}. Have in mind that this word is going to be used for this instruction {question_prompt_instruction}. 
    Do NOT generate the whole question yet. In this step just generate and choose appropriate word considering what the question is about. Explain your chain-of-thought for each choice (why is it appropriate?).
    Then select exactly ONE of the three as the final word, and provide reasoning for it (think aloud).

    You should only consider word that does NOT share the same prefix as the others from the given words {word_list}. Have in mind that this word is going to be used for this instruction {question_prompt_instruction}. 
    Do NOT generate the whole question yet. In this step just generate and choose appropriate word considering what the question is about. Explain your chain-of-thought for each choice (why is it appropriate?).
    Then select exactly ONE of the three as the final word, and provide reasoning for it (think aloud).
    """
        step1_result = generate_gpt(step1_prompt, model, tokenizer)
        print(f"==> Step 1 result: {step1_result}\n")
        # Just do a naive parse: find "Chosen Word:" or something
        # If there's no consistent structure, we could guess or rely on a simpler approach
        chosen_word = parse_chosen_word(step1_result)

        # If we can't parse the chosen word, fallback
        if not chosen_word:
            chosen_word = "mysteryWord"

        # If the chosen word is already forbidden, optionally attempt a few more tries:
        attempts = 0
        while chosen_word and chosen_word.lower() in [fw.lower() for fw in forbidden_list]:
            attempts += 1
            if attempts > 3:
                # If GPT keeps repeating words, just force a placeholder.
                # print(f"Chosen word '{chosen_word}' was in forbidden_list. Change this word.")
                temp_prompt = f"Chosen word '{chosen_word}' was in forbidden_list: {forbidden_list}. Change this word such that it is not in the forbidden list."
                step1_result = generate_gpt(temp_prompt, model, tokenizer)
                chosen_word = parse_chosen_word(step1_result)
                break
            print(f"Chosen word '{chosen_word}' was in forbidden_list; re-asking GPT for a new word.")
            step1_result = generate_gpt(step1_prompt, model, tokenizer)
            chosen_word = parse_chosen_word(step1_result)
            if not chosen_word:
                chosen_word = "mysteryWord"

    else:
        step1_prompt = f"""
    {question_prompt_instruction}

    Step 1 (Pick Words):
    Question Type = {question_type}
    Word Difficulty = {word_difficulty}, Task Difficulty = {task_difficulty}

    Have in mind that everything should be suitable for Grade 3-5 words that fit the morphological concept
    (prefix, suffix, root, etc.) for this Question Type: {question_type}. 
    You should only consider words that one of them does NOT share the same prefix as the others from the given words {word_list}. 
    Have in mind that this word is going to be used for this instruction {question_prompt_instruction}. 
    Do NOT generate the whole question yet. In this step just consider choosing the appropriate word considering what the question is about. 
    Explain your chain-of-thought for each choice (why is it appropriate?). 
    SKIP THIS STEP, AND MOVE FORWARD WITH STEP 2.
    """
        step1_result = generate_gpt(step1_prompt, model, tokenizer)
        print(f"==> Step 1 result: {step1_result}\n")
        # Just do a naive parse: find "Chosen Word:" or something
        # If there's no consistent structure, we could guess or rely on a simpler approach
        chosen_word = parse_chosen_word(step1_result)

        # If we can't parse the chosen word, fallback
        if not chosen_word:
            chosen_word = "mysteryWord"

        # If the chosen word is already forbidden, optionally attempt a few more tries:
        attempts = 0
        while chosen_word and chosen_word.lower() in [fw.lower() for fw in forbidden_list]:
            attempts += 1
            if attempts > 3:
                # If GPT keeps repeating words, just force a placeholder.
                # print(f"Chosen word '{chosen_word}' was in forbidden_list. Change this word.")
                temp_prompt = f"Chosen word '{chosen_word}' was in forbidden_list: {forbidden_list}. Change this word such that it is not in the forbidden list."
                step1_result = generate_gpt(temp_prompt, model, tokenizer)
                chosen_word = parse_chosen_word(step1_result)
                break
            print(f"Chosen word '{chosen_word}' was in forbidden_list; re-asking GPT for a new word.")
            step1_result = generate_gpt(step1_prompt, model, tokenizer)
            chosen_word = parse_chosen_word(step1_result)
            if not chosen_word:
                chosen_word = "mysteryWord"
    # ---------------------------
    # Step 2: Draft question
    # ---------------------------
    step2_prompt = f"""
Step 2 (Draft Question):
We have chosen the word: {chosen_word}.

Now draft a 3-choice question based on the following instruction: {question_prompt_instruction}. Provide a chain-of-thought explaining
how you formed the question, and provide the correct answer.

Do NOT finalize the answer choices yet. Give the question text
and placeholders for A/B/C. For instance:
"Question: ... A) ... B) ... C) ..."

"""
    step2_result = generate_gpt(step2_prompt, model, tokenizer)
    # We'll accept step2_result as a partial question
    print(f"==> Step 2 result: {step2_result}\n")
    
    # ---------------------------
    # Step 3: Add distractors & finalize
    # ---------------------------
    step3_prompt = f"""
Step 3 (Add Choices & Finalize):
Based on the draft question and the correct answer you provided in the previous step:

{step2_result}

Update the TWO distractors with reasoning, considering the specified task difficulty of: {task_difficulty}. Provide chain-of-thought
about how each distractor might trick the student, and confirm the correct answer.

Finally, shuffle the answer choices and present the final question in a parseable format:
Question: ...
A) ...
B) ...
C) ...
Correct Answer: ...
Do NOT provide the reasonings in the above parseable format. Write them here, after everything is done.
"""
    step3_result = generate_gpt(step3_prompt, model, tokenizer)
    print(f"==> Step 3 result: {step3_result}\n")
    # The final text to parse is step3_result
    final_text = step3_result
    return final_text, chosen_word


###############################################################################
# 7. NEW Strategy: chain_of_thought_plus_role_chain
###############################################################################
def prompt_chain_of_thought_plus_role_chain(question_type, word_difficulty, task_difficulty):
    

    role_instructions = f"""
--- 3-Role Reasoning for a Grade 3-5 3-choice question ---

We want a 3-choice question for:
  question_type = {question_type}
  word_difficulty = {word_difficulty}
  task_difficulty = {task_difficulty}

You should act in all of the following three roles, one by one, and think aloud in each of them (provide reasonings):
==========
Roles & Instructions:
==========

(1) Teacher Role
  - Act as a Grade 3–5 teacher.
  - Propose a question that is suitable for a grade 3-5 student, focusing on the morphological concept 
    (prefix, suffix, root, etc.) relevant to question_type={question_type}.
  - Provide your chain-of-thought on how the question was formed (provide reasoning, or think aloud), 
    ensuring it's neither too trivial nor too advanced for grades 3–5.
  - Then pass along your question and partial choices to the next role.

(2) Student Role
  - Act as a Grade 3–5 student.
  - Read what the Teacher proposed. 
  - Comment if the question is confusing, or if any distractor is obviously incorrect.
  - Provide your chain-of-thought as a student (provide reasoning, or think aloud).
  - Then pass along your outputs to the next role.

(3) Technometrician Role
  - Act as a test-design specialist focusing on morphological objectives (prefix, suffix, root, etc.) 
    and checking alignment with word_difficulty={word_difficulty} & task_difficulty={task_difficulty}.
  - Evaluate the question from both Teacher and Student roles:
    - Are we accurately testing the morphological skill for question_type={question_type}?
    - Are the difficulty levels appropriate?
  - Provide final refinements if needed. Be strict if you can make the question or the distractors more aligned with what is asked.
  - Then present the finalized question in a parseable format, exactly as follows:

      Final Question: [refined question text here]
      A) [choice A]
      B) [choice B]
      C) [choice C]
      Correct Answer: [the correct choice]

    """

    return role_instructions


###############################################################################
# 8. Prompt Generators for Each Question Type
###############################################################################
def generate_prefix_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 1
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has a prefix{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    # -- 1) GPT call to pick a word
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    # -- 2) The main MCQ prompt (Zero-shot style)
    prompt = (
        f"Generate a 3-choice question for a student learning about prefixes. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to identify the prefix in '{word}' and provide "
        f"two incorrect choices along with the correct answer. Please specify the correct answer."
    )
        
    # -- 3) Append optional prompting strategy
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 1, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(1, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        # Call the RL version, which returns final_text and chosen_word
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            1, word_difficulty, task_difficulty,model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(1, word_difficulty, task_difficulty)
        return prompt, word
    else:
        # default zero-shot
        return prompt, word


def generate_suffix_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 2
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has a suffix{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about suffixes. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to identify the suffix in '{word}' and provide two "
        f"incorrect choices along with the correct answer. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 2, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(2, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            2, word_difficulty, task_difficulty,model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(2, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_root_word_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 3
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has a prefix or suffix{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about root words. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to identify the root word in '{word}' and provide two "
        f"incorrect choices along with the correct answer. Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 3, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(3, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            3, word_difficulty, task_difficulty, model, tokenizer,forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(3, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_common_prefix_prompt(word_list, word_difficulty, task_difficulty, data, prompting, model, tokenizer):  # question_type = 4
    prompt = (
        f"Generate a 3-choice question for a student learning about morphemes. "
        f"The words given are {word_list}, with word difficulty {word_difficulty} and "
        f"task difficulty {task_difficulty}. The question should ask the student to identify the word "
        f"that does NOT share the same prefix as the others. Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 4, 8)
        return prompt, word_list
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(4, word_difficulty, task_difficulty)
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            4, word_difficulty, task_difficulty,model, tokenizer, forbidden_list=[""], word_list=word_list
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(4, word_difficulty, task_difficulty)
        return prompt, word_list
    else:
        return prompt, word_list


def generate_common_suffix_prompt(word_list, word_difficulty, task_difficulty, data, prompting, model, tokenizer):  # question_type = 5
    prompt = (
        f"Generate a 3-choice question for a student learning about morphemes. "
        f"The words given are {word_list}, with word difficulty {word_difficulty} and "
        f"task difficulty {task_difficulty}. The question should ask the student to identify the word "
        f"that does NOT share the same suffix as the others. Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 5, 8)
        return prompt, word_list
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(5, word_difficulty, task_difficulty)
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            5, word_difficulty, task_difficulty,model, tokenizer, forbidden_list=[""], word_list=word_list
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(5, word_difficulty, task_difficulty)
        return prompt, word_list
    else:
        return prompt, word_list


def generate_word_transformation_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 6
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has variations with different meanings{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about word transformations. "
        f"The word is '{word}', word difficulty {word_difficulty}, and task difficulty {task_difficulty}. "
        f"The question should ask the student to transform '{word}' to a new meaning, with two incorrect "
        f"choices and one correct answer. Clearly specify correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 6, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(6, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            6, word_difficulty, task_difficulty, model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(6, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_word_meaning_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 7
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has a different meaning with a same prefix or suffix{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about affixed words. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct meaning of '{word}' from three answer choices. "
        f"Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 7, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(7, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            7, word_difficulty, task_difficulty, model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(7, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_spelling_prompt(word_difficulty, task_difficulty, data, prompting, prompts, model, tokenizer):  # question_type = 8
    if prompts:
        prompt_restriction = (
            f" The question and the word should not be similar to any of the previously generated questions: "
            f"{', '.join(prompts)}."
        )
    else:
        prompt_restriction = ""

    prompt = (
        f"Generate a 3-choice question for a student learning about spelling based on morpheme meaning"
        f"{prompt_restriction} The question should include a word with a suffix."
        f"Provide two misspelled variations and one correct spelling. The question should have word difficulty "
        f"{word_difficulty} and task difficulty {task_difficulty}. Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 8, 8)
        return prompt, prompt
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, prompt
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(8, word_difficulty, task_difficulty)
        return prompt, prompt
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            8, word_difficulty, task_difficulty, model, tokenizer, forbidden_list=prompts
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(8, word_difficulty, task_difficulty)
        return prompt, prompt
    else:
        return prompt, prompt


def generate_affixed_word_breakdown_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 9
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has at least three parts and at most four parts{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning to break affixed words into parts. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to break '{word}' into its correct parts (prefix, root, suffix) "
        f"and provide two incorrect choices along with the correct answer. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 9, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(9, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            9, word_difficulty, task_difficulty, model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(9, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_prefix_definition_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 10
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has a prefix{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about prefixes. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct definition of the prefix in '{word}' from "
        f"three answer choices. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 10, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(10, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            10, word_difficulty, task_difficulty, model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(10, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_root_word_definition_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 11
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has a prefix or suffix{word_exclusion} "
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about root words in affixed words. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct definition of the root word in '{word}' from "
        f"three answer choices. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 11, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(11, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            11, word_difficulty, task_difficulty,model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(11, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_suffix_definition_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 12
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""

    gen_prompt = (
        f"Please generate an English word that has a suffix{word_exclusion}"
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about suffixes. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct definition or function of the suffix in '{word}' "
        f"from three answer choices. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 12, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(12, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            12, word_difficulty, task_difficulty,  model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(12, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_morphologically_complex_word_definition_prompt(word_difficulty, task_difficulty, data, prompting, words, model, tokenizer):  # question_type = 13
    if words:
        word_exclusion = f" and it must NOT be any of these words (case insensitive): {', '.join(words)}."
    else:
        word_exclusion = ""
    gen_prompt = (
        f"Please generate an English word whose morpheme has a distinct meaning, can take a suffix or prefix,{word_exclusion}"
        f"Its level of difficulty for grade 3-5 is {word_difficulty} out of 5.\n"
        f"Return your response in this exact format:\n"
        f"WORD: [your word]\n"
        f"EXPLANATION: [brief explanation why this word is appropriate]"
    )
    result = generate_gpt(gen_prompt, model, tokenizer)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about morphologically complex words. "
        f"The word is '{word}'. With word difficulty {word_difficulty} and task difficulty {task_difficulty}, "
        f"the question should ask the student to select the correct definition of '{word}' based on its morphemes. "
        f"Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 13, 8)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(13, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            13, word_difficulty, task_difficulty, model, tokenizer, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(13, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word

###############################################################################
# 9. Dispatcher for Zero/Few/Chain-of-thought/Chain-of-thought-plus-sequential/
#    Chain-of-thought-plus-role-chain
###############################################################################
def load_lists(file):
    import json
    with open(file,'r') as f:
        list1=json.load(f)
    return list1
def prompt_generator(question_type, word_difficulty, task_difficulty, data, prompting, forbidden_list,model, tokenizer):
    list1=load_lists('/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/JsonLists/list1.json')
    list2=load_lists('/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/JsonLists/list2.json')
    if question_type == 1:
        return generate_prefix_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list,model, tokenizer)
    elif question_type == 2:
        return generate_suffix_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list, model, tokenizer)
    elif question_type == 3:
        return generate_root_word_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list, model, tokenizer)
    elif question_type == 4:
        allowed = [sublist for sublist in list1 if sublist not in forbidden_list]
        random_word_list = random.choice(allowed)
        return generate_common_prefix_prompt(random_word_list, word_difficulty, task_difficulty, data, prompting, model, tokenizer)
    elif question_type == 5:
        allowed = [sublist for sublist in list2 if sublist not in forbidden_list]
        random_word_list = random.choice(allowed)
        return generate_common_suffix_prompt(random_word_list, word_difficulty, task_difficulty, data, prompting, model, tokenizer)
    elif question_type == 6:
        return generate_word_transformation_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list, model, tokenizer)
    elif question_type == 7:
        return generate_word_meaning_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list, model, tokenizer)
    elif question_type == 8:
        return generate_spelling_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list, model, tokenizer)
    elif question_type == 9:
        return generate_affixed_word_breakdown_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list, model, tokenizer)
    elif question_type == 10:
        return generate_prefix_definition_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list, model, tokenizer)
    elif question_type == 11:
        return generate_root_word_definition_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list, model, tokenizer)
    elif question_type == 12:
        return generate_suffix_definition_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list, model, tokenizer)
    elif question_type == 13:
        return generate_morphologically_complex_word_definition_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list, model, tokenizer)
    else:
        # Catch-all for unspecified question_type
        return generate_definition_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list, model, tokenizer)

###############################################################################
# 10. A more comprehensive question parser
###############################################################################
def parse_question_output(text):
    """
    A comprehensive parser for different strategies.

    Returns a dict with:
      {
        'question': str or None,
        'choice_a': str or None,
        'choice_b': str or None,
        'choice_c': str or None,
        'correct_answer': str or None,
        'chain_of_thought': str or None,
        'teacher_reasoning': str or None,
        'student_reasoning': str or None,
        'psychometrician_reasoning': str or None,
        'step_1': str or None,
        'step_2': str or None,
        'step_3': str or None
      }
    """
    if not text:
        return None

    parsed = {
        'question': None,
        'choice_a': None,
        'choice_b': None,
        'choice_c': None,
        'correct_answer': None,
        'chain_of_thought': None,
        'teacher_reasoning': None,
        'student_reasoning': None,
        'psychometrician_reasoning': None,
        'step_1': None,
        'step_2': None,
        'step_3': None
    }

    # 1) Extract teacher/student/psychometrician if present
    teacher_match = re.search(
        r"(?:Teacher\s*:\s*)(.*?)(?=\n\s*(?:Student\s*:|Psychometrician\s*:|Final Question\s*:|Question\s*:|$))",
        text, re.DOTALL
    )
    if teacher_match:
        parsed['teacher_reasoning'] = teacher_match.group(1).strip()

    student_match = re.search(
        r"(?:Student\s*:\s*)(.*?)(?=\n\s*(?:Teacher\s*:|Psychometrician\s*:|Final Question\s*:|Question\s*:|$))",
        text, re.DOTALL
    )
    if student_match:
        parsed['student_reasoning'] = student_match.group(1).strip()

    psych_match = re.search(
        r"(?:Psychometrician\s*:\s*|Technometrician\s*:\s*)(.*?)(?=\n\s*(?:Teacher\s*:|Student\s*:|Final Question\s*:|Question\s*:|$))",
        text, re.DOTALL
    )
    if psych_match:
        parsed['psychometrician_reasoning'] = psych_match.group(1).strip()

    # 2) Extract chain-of-thought if "Chain of Thought" block
    cot_match = re.search(
        r"--- Chain of Thought\s*---(.*?)(?=\n---|\nQuestion|\nStep|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if cot_match:
        parsed['chain_of_thought'] = cot_match.group(1).strip()

    # 3) Extract step_1 / step_2 / step_3 if present
    step1 = re.search(r"(?:Step\s*1\s*\(.*?\)\s*:|Step\s*1\s*:)(.*?)(?=Step\s*2|$)", text, re.DOTALL)
    if step1:
        parsed['step_1'] = step1.group(1).strip()

    step2 = re.search(r"(?:Step\s*2\s*\(.*?\)\s*:|Step\s*2\s*:)(.*?)(?=Step\s*3|$)", text, re.DOTALL)
    if step2:
        parsed['step_2'] = step2.group(1).strip()

    step3 = re.search(r"(?:Step\s*3\s*\(.*?\)\s*:|Step\s*3\s*:)(.*?)(?=(Step\s*4|Question\s*:|Final Question\s*:|$))", 
                      text, re.DOTALL)
    if step3:
        parsed['step_3'] = step3.group(1).strip()

    # 4) Extract final question (look for "Final Question:" or "Question:")
    final_q_match = re.search(r"Final Question\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if final_q_match:
        # We'll parse out until we find A) or end
        remainder = final_q_match.group(1)
        # The question portion ends where "A)" might begin
        splitted = re.split(r"\nA\)|\nA\) ", remainder, 1)
        parsed['question'] = splitted[0].strip()
    else:
        # fallback to "Question:"
        q_match = re.search(r"Question\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if q_match:
            remainder = q_match.group(1)
            splitted = re.split(r"\nA\)|\nA\) ", remainder, 1)
            parsed['question'] = splitted[0].strip() if splitted else remainder.strip()

    # 5) Extract choices A), B), C)
    # We do multi-line capture    
    a_match = re.search(r"A\)\s*(.*?)(?=\n[B-Z]\)|\nCorrect Answer:|\Z)", text, re.DOTALL)
    b_match = re.search(r"B\)\s*(.*?)(?=\n[C-Z]\)|\nCorrect Answer:|\Z)", text, re.DOTALL)
    c_match = re.search(r"C\)\s*(.*?)(?=\n[D-Z]\)|\nCorrect Answer:|\Z)", text, re.DOTALL)
    
    if a_match:
        parsed['choice_a'] = a_match.group(1).strip()
    # b_match = re.search(r"B\)\s*(.*?)(?=\n[C-Z]\)|\Z)", text, re.DOTALL)
    if b_match:
        parsed['choice_b'] = b_match.group(1).strip()
    # c_match = re.search(r"C\)\s*(.*?)(?=\n[D-Z]\)|\Z)", text, re.DOTALL)
    if c_match:
        parsed['choice_c'] = c_match.group(1).strip()

    # 6) Extract "Correct Answer:"
    correct_match = re.search(r"Correct\s*Answer\s*:\s*(.*)", text, re.IGNORECASE)
    if correct_match:
        ans = correct_match.group(1).strip()
        # Remove leading "A) ", "B) ", or "C) " if present:
        ans_no_label = re.sub(r'^[ABC]\)\s*', '', ans, flags=re.IGNORECASE).strip()
        parsed['correct_answer'] = ans_no_label

    return parsed
if __name__ == "__main__":
    #model, tokenizer= train_gemma()
    #model_path = "/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/last_gemma_morphology_20250404_014910/final_model"
    model_path = "/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/gemma_morphology/final_model"
    model, tokenizer = load_gemma_model(model_path)
    data_file = 'MC_data_MA2.csv'
    
    strategies = [#'chain_of_thought_plus_role_chain','chain_of_thought_plus_sequential_rl', 'chain_of_thought',
        #'chain_of_thought_plus_sequential',
                  'few_shot'#,'zero_shot'
    ]#,    'chain_of_thought_plus_role_chain'
     #   'chain_of_thought_plus_sequential_rl',
     #   'chain_of_thought',
     #   'chain_of_thought_plus_sequential',
     #   'few_shot',
     #   'zero_shot']
    #strategies = ['chain_of_thought_plus_sequential_rl']
        # 'chain_of_thought_plus_role_chain'
    # ]

    data = pd.read_csv(data_file, encoding='utf-8')
    NUM_QUESTIONS = 6
    word_difficulties = [3,4,5] #data["Word_Difficulty"].unique()
    task_difficulties = ['Hard']#, 'Hard', 'Medium']
    Question_Type_array = np.array(data["Question_Type"].unique())
    questio_id = np.sort([int(float(x)) for x in Question_Type_array])

    for strategy in strategies:
        questions_data = []
        prev_prompts = {q_type: [] for q_type in questio_id}

        for word_difficulty in word_difficulties:
            for task_difficulty in task_difficulties:
                for question_type in questio_id:
                    for i in range(NUM_QUESTIONS):
                        print(f"\n=== Question #{i+1} === Question Type #{question_type} === {strategy} ===word difficulty:{word_difficulty}======task difficulty: {task_difficulty}")
                        prompt_or_final_text, new_word = prompt_generator(
                            question_type,
                            word_difficulty,
                            task_difficulty,
                            data,
                            strategy,
                            prev_prompts[question_type],
                            model, tokenizer
                        )
                        generated_text = generate_gpt(prompt_or_final_text, model, tokenizer)

                        if generated_text:
                            parsed = parse_question_output(generated_text)
                            if parsed:
                                parsed['question_type'] = question_type
                                parsed['word_difficulty'] = word_difficulty
                                parsed['task_difficulty'] = task_difficulty
                                parsed['whole_text']=generated_text
                                questions_data.append(parsed)

                        if new_word:
                            prev_prompts[question_type].append(new_word)

                        #if generated_text:
                        #    print("\nGenerated Text from GEMMA Model:")
                        #    #print(generated_text)
                        #else:
                            #print("No text returned by the model.")

        # Save after finishing one strategy
        df = pd.DataFrame(questions_data)
        filename = f'Final_Generated_Questions_Gemma_Part10_{strategy}.csv'
        df.to_csv(filename, index=False)
        print(f"\nSaved questions for strategy '{strategy}' to {filename}")