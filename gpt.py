import pandas as pd
import numpy as np
import random
import openai
import os
import re
###############################################################################
#0. API Key
###############################################################################
openai.api_key = "sk-proj-PSXJ5xydTMPUZcHkLIuq5oGnh" #This is a fake api_key and it should be replace with real api_key

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
def generate_gpt(prompt, model_name="gpt-3.5-turbo"):
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
        formatted_examples = ""
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
def prompt_chain_of_thought_plus_sequential_rl(question_type, word_difficulty, task_difficulty, forbidden_list=None, **kwargs):
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
    """
        step1_result = generate_gpt(step1_prompt)
        print(f"==> Step 1 result: {step1_result}\n")
        # Just do a naive parse: find "Chosen Word:" or something
        # If there's no consistent structure, we could guess or rely on a simpler approach
        chosen_word = parse_chosen_word(step1_result)

        # If we can't parse the chosen word, fallback
        if not chosen_word:
            chosen_word = "mysteryWord"

                
        # Single-check if chosen_word is in forbidden_list
        if chosen_word.lower() in [fw.lower() for fw in forbidden_list]:
            # Force a single re-ask with a "temp_prompt"
            conflict_prompt = (
                f"You gave the word '{chosen_word}', but it's in the forbidden list: {forbidden_list}.\n"
                f"Please choose a different word that meets the previously-defined criteria, and also is NOT in that list."
            )
            step1_result_2 = generate_gpt(conflict_prompt)
            chosen_word = parse_chosen_word(step1_result_2)
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
        step1_result = generate_gpt(step1_prompt)
        print(f"==> Step 1 result: {step1_result}\n")
        # Just do a naive parse: find "Chosen Word:" or something
        # If there's no consistent structure, we could guess or rely on a simpler approach
        chosen_word = parse_chosen_word(step1_result)

        # If we can't parse the chosen word, fallback
        if not chosen_word:
            chosen_word = "mysteryWord"

        # # Single-check if chosen_word is in forbidden_list
        # if chosen_word.lower() in [fw.lower() for fw in forbidden_list]:
        #     # Force a single re-ask with a "temp_prompt"
        #     conflict_prompt = (
        #         f"You gave the word '{chosen_word}', but it's in the forbidden list: {forbidden_list}.\n"
        #         f"Please choose a different word that meets the previously-defined criteria, and also is NOT in that list."
        #     )
        #     step1_result_2 = generate_gpt(conflict_prompt)
        #     chosen_word = parse_chosen_word(step1_result_2)
        #     if not chosen_word:
        #         chosen_word = "mysteryWord"
                
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
    step2_result = generate_gpt(step2_prompt)
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
    step3_result = generate_gpt(step3_prompt)
    print(f"==> Step 3 result: {step3_result}\n")
    # The final text to parse is step3_result
    final_text = step3_result
    print(f"====== HERE IS THE FORBIDDEN_LIST: {forbidden_list}\n")
    return final_text, chosen_word


###############################################################################
# 7. NEW Strategy: chain_of_thought_plus_role_chain
###############################################################################
def prompt_chain_of_thought_plus_role_chain(question_type, word_difficulty, task_difficulty):
    """
    A multi-role chain-of-thought approach. The model is asked to produce an MCQ 
    (with exactly 3 choices) for a grade 3-5 audience, but does so by iterating 
    through three different 'roles' in its reasoning:

    Role 1) Teacher:
       - Think about how to instruct a Grade 3-5 student. 
       - Provide a suitable question and partial answer choices in plain, accessible language.
       - Make sure the question is neither too trivial nor too advanced for grades 3-5.
       - Provide your chain-of-thought on how you arrived at this version.

    Role 2) Student:
       - Evaluate the Teacher's question from a student’s perspective.
       - Check if any distractor is obviously or trivially wrong, or if the question is unclear.
       - Suggest if the question might be confusing or if it needs improvement.
       - Provide your chain-of-thought on how you analyzed the question.

    Role 3) Technometrician (or Psychometrician):
       - Verify that the question meets the morphological objective implied by question_type
         (e.g., identifying prefix/suffix, ensuring correct root word, etc.).
       - Check that the difficulty aligns with the stated word/task difficulty.
       - Offer final refinements if needed, then present the final version of the question 
         in a parseable format:

         Final Question: ...
         A) ...
         B) ...
         C) ...
         Correct Answer: ...

    The model should output the chain-of-thought for each role, followed by the final question.
    The final question must always end in the parseable format shown above. 
    """

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
def generate_prefix_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 1
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
    result = generate_gpt(gen_prompt)
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
        prompt += prompt_few_shot(data, 1, 6)
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
            1, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(1, word_difficulty, task_difficulty)
        return prompt, word
    else:
        # default zero-shot
        return prompt, word


def generate_suffix_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 2
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about suffixes. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to identify the suffix in '{word}' and provide two "
        f"incorrect choices along with the correct answer. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 2, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(2, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            2, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(2, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_root_word_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 3
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about root words. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to identify the root word in '{word}' and provide two "
        f"incorrect choices along with the correct answer. Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 3, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(3, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            3, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(3, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_common_prefix_prompt(word_list, word_difficulty, task_difficulty, data, prompting):  # question_type = 4
    prompt = (
        f"Generate a 3-choice question for a student learning about morphemes. "
        f"The words given are {word_list}, with word difficulty {word_difficulty} and "
        f"task difficulty {task_difficulty}. The question should ask the student to identify the word "
        f"that does NOT share the same prefix as the others. Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 4, 6)
        return prompt, word_list
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(4, word_difficulty, task_difficulty)
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            4, word_difficulty, task_difficulty, forbidden_list=[], word_list=word_list
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(4, word_difficulty, task_difficulty)
        return prompt, word_list
    else:
        return prompt, word_list


def generate_common_suffix_prompt(word_list, word_difficulty, task_difficulty, data, prompting):  # question_type = 5
    prompt = (
        f"Generate a 3-choice question for a student learning about morphemes. "
        f"The words given are {word_list}, with word difficulty {word_difficulty} and "
        f"task difficulty {task_difficulty}. The question should ask the student to identify the word "
        f"that does NOT share the same suffix as the others. Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 5, 6)
        return prompt, word_list
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(5, word_difficulty, task_difficulty)
        return prompt, word_list
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            5, word_difficulty, task_difficulty, forbidden_list=[], word_list=word_list
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(5, word_difficulty, task_difficulty)
        return prompt, word_list
    else:
        return prompt, word_list


def generate_word_transformation_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 6
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about word transformations. "
        f"The word is '{word}', word difficulty {word_difficulty}, and task difficulty {task_difficulty}. "
        f"The question should ask the student to transform '{word}' to a new meaning, with two incorrect "
        f"choices and one correct answer. Clearly specify correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 6, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(6, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            6, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(6, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_word_meaning_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 7
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about affixed words. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct meaning of '{word}' from three answer choices. "
        f"Clearly specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 7, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(7, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            7, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(7, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_spelling_prompt(word_difficulty, task_difficulty, data, prompting, prompts):  # question_type = 8
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
        prompt += prompt_few_shot(data, 8, 6)
        return prompt, prompt
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, prompt
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(8, word_difficulty, task_difficulty)
        return prompt, prompt
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            8, word_difficulty, task_difficulty, forbidden_list=prompts
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(8, word_difficulty, task_difficulty)
        return prompt, prompt
    else:
        return prompt, prompt


def generate_affixed_word_breakdown_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 9
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning to break affixed words into parts. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to break '{word}' into its correct parts (prefix, root, suffix) "
        f"and provide two incorrect choices along with the correct answer. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 9, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(9, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            9, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(9, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_prefix_definition_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 10
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about prefixes. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct definition of the prefix in '{word}' from "
        f"three answer choices. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 10, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(10, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            10, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(10, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_root_word_definition_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 11
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about root words in affixed words. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct definition of the root word in '{word}' from "
        f"three answer choices. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 11, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(11, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            11, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(11, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_suffix_definition_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 12
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about suffixes. "
        f"The word is '{word}', with word difficulty {word_difficulty} and task difficulty {task_difficulty}. "
        f"The question should ask the student to select the correct definition or function of the suffix in '{word}' "
        f"from three answer choices. Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 12, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(12, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            12, word_difficulty, task_difficulty, forbidden_list=words
        )
        return final_text, chosen_word
    elif prompting == 'chain_of_thought_plus_role_chain':
        prompt += prompt_chain_of_thought_plus_role_chain(12, word_difficulty, task_difficulty)
        return prompt, word
    else:
        return prompt, word


def generate_morphologically_complex_word_definition_prompt(word_difficulty, task_difficulty, data, prompting, words):  # question_type = 13
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
    result = generate_gpt(gen_prompt)
    word_match = re.search(r"WORD:\s*(\w+)", result or "")
    word = word_match.group(1) if word_match else None

    prompt = (
        f"Generate a 3-choice question for a student learning about morphologically complex words. "
        f"The word is '{word}'. With word difficulty {word_difficulty} and task difficulty {task_difficulty}, "
        f"the question should ask the student to select the correct definition of '{word}' based on its morphemes. "
        f"Please specify the correct answer."
    )
    if prompting == 'few_shot':
        prompt += prompt_few_shot(data, 13, 6)
        return prompt, word
    elif prompting == 'chain_of_thought':
        prompt += prompt_chain_of_thought()
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential':
        prompt += prompt_chain_of_thought_plus_sequential(13, word_difficulty, task_difficulty)
        return prompt, word
    elif prompting == 'chain_of_thought_plus_sequential_rl':
        final_text, chosen_word = prompt_chain_of_thought_plus_sequential_rl(
            13, word_difficulty, task_difficulty, forbidden_list=words
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

def prompt_generator(question_type, word_difficulty, task_difficulty, data, prompting, forbidden_list):
    list1=load_lists('/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/JsonLists/list1.json')
    list2=load_lists('/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/JsonLists/list2.json')

    if question_type == 1:
        return generate_prefix_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list)
    elif question_type == 2:
        return generate_suffix_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list)
    elif question_type == 3:
        return generate_root_word_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list)
    elif question_type == 4:
        allowed = [sublist for sublist in list1 if sublist not in forbidden_list]
        random_word_list = random.choice(allowed)
        return generate_common_prefix_prompt(random_word_list, word_difficulty, task_difficulty, data, prompting)
    elif question_type == 5:
        allowed = [sublist for sublist in list2 if sublist not in forbidden_list]
        random_word_list = random.choice(allowed)
        return generate_common_suffix_prompt(random_word_list, word_difficulty, task_difficulty, data, prompting)
    elif question_type == 6:
        return generate_word_transformation_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list)
    elif question_type == 7:
        return generate_word_meaning_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list)
    elif question_type == 8:
        return generate_spelling_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list)
    elif question_type == 9:
        return generate_affixed_word_breakdown_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list)
    elif question_type == 10:
        return generate_prefix_definition_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list)
    elif question_type == 11:
        return generate_root_word_definition_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list)
    elif question_type == 12:
        return generate_suffix_definition_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list)
    elif question_type == 13:
        return generate_morphologically_complex_word_definition_prompt(word_difficulty, task_difficulty, data, prompting,forbidden_list)
    else:
        # Catch-all for unspecified question_type
        return generate_definition_prompt(word_difficulty, task_difficulty, data, prompting, forbidden_list)

###############################################################################
# 10. A more comprehensive question parser
###############################################################################
import re

def parse_question_output(text):
    """
    A comprehensive parser for all the prompting strategies we discussed:
      - zero_shot
      - few_shot
      - chain_of_thought
      - chain_of_thought_plus_sequential
      - chain_of_thought_plus_sequential_rl
      - chain_of_thought_plus_role_chain

    This function carefully extracts:
      - teacher_reasoning, student_reasoning, psychometrician_reasoning
      - chain_of_thought
      - up to 3 steps (step_1, step_2, step_3)
      - question
      - choice_a, choice_b, choice_c
      - correct_answer

    Key Logic / Fallbacks:
      1) We match teacher/student/psychometrician with lines like:
         'Teacher:', '(1) Teacher Role:', 'Student:', 'Psychometrician:', 'Technometrician:', etc.
      2) Chain-of-thought can appear as '--- Chain of Thought ---', 'Chain-of-Thought:', etc.
      3) Steps appear as 'Step 1:', 'Step 1 (Pick Words):', etc., up to Step 3.
      4) Final question is searched in this order:
           - 'Final Question:', 'Updated Question:', 'Final MCQ:', 'My Final Question is:'
           - fallback: 'Question:'
           - if that fails, we look for a line ending with '?'
      5) Choices are labeled 'A)', 'A.', 'A:', 'B)', 'C)', etc. We use a multiline regex to capture
         until we see the next choice label, 'Correct Answer:', or the end of the text (\\Z).
      6) Correct Answer is first looked for with lines like 'Correct Answer:', 'Answer:', or
         'The correct answer is:'. If not found, we see if one of the choices ends with '(Correct answer)'
         (case-insensitive). If so, we remove that parenthetical from the choice and set that as correct_answer.

    It returns a dictionary with these keys:
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

    # 0) If there's no text or it's empty, return a blank parse.
    if not text or not text.strip():
        return {
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

    # Normalize line endings to "\n"
    text = text.replace('\r\n', '\n')

    # Our parsed results dictionary
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

    ################################################################
    # 1) Extract Teacher / Student / Psychometrician reasonings
    ################################################################
    teacher_match = re.search(
        r"(?i)(?:\(?\d\)?\s*Teacher\s*Role\s*:|Teacher\s*:)(.*?)(?=\n\s*(?:\(?\d\)?\s*Student\s*Role\s*:|Student\s*:|\(?\d\)?\s*Psychometrician\s*Role\s*:|Psychometrician\s*:|Technometrician\s*:|\(?\d\)?\s*Technometrician\s*Role\s*:|Final\s*Question\s*:|Updated\s*Question\s*:|Question\s*:|$))",
        text, re.DOTALL
    )
    if teacher_match:
        parsed['teacher_reasoning'] = teacher_match.group(1).strip()

    student_match = re.search(
        r"(?i)(?:\(?\d\)?\s*Student\s*Role\s*:|Student\s*:)(.*?)(?=\n\s*(?:\(?\d\)?\s*Teacher\s*Role\s*:|Teacher\s*:|\(?\d\)?\s*Psychometrician\s*Role\s*:|Psychometrician\s*:|Technometrician\s*:|\(?\d\)?\s*Technometrician\s*Role\s*:|Final\s*Question\s*:|Updated\s*Question\s*:|Question\s*:|$))",
        text, re.DOTALL
    )
    if student_match:
        parsed['student_reasoning'] = student_match.group(1).strip()

    psych_match = re.search(
        r"(?i)(?:\(?\d\)?\s*Psychometrician\s*Role\s*:|Psychometrician\s*:|\(?\d\)?\s*Technometrician\s*Role\s*:|Technometrician\s*:)(.*?)(?=\n\s*(?:\(?\d\)?\s*Teacher\s*Role\s*:|Teacher\s*:|\(?\d\)?\s*Student\s*Role\s*:|Student\s*:|Final\s*Question\s*:|Updated\s*Question\s*:|Question\s*:|$))",
        text, re.DOTALL
    )
    if psych_match:
        parsed['psychometrician_reasoning'] = psych_match.group(1).strip()

    ################################################################
    # 2) Extract chain-of-thought
    ################################################################
    # Could appear as '--- Chain of Thought ---' or 'Chain-of-Thought:'
    cot_match = re.search(
        r"(?i)(?:---\s*Chain[\s\-_]*of[\s\-_]*Thought\s*---|Chain[\s\-_]*of[\s\-_]*Thought\s*:)(.*?)(?=\n---|\nQuestion|\nStep|\Z)",
        text, re.DOTALL
    )
    if cot_match:
        parsed['chain_of_thought'] = cot_match.group(1).strip()

    ################################################################
    # 3) Extract up to 3 Steps: Step 1, Step 2, Step 3
    ################################################################
    # Usually from chain_of_thought_plus_sequential
    for n in range(1, 4):
        step_regex = (
            rf"(?is)(?:Step\s*{n}\s*(?:\(.*?\))?\s*[:\.]\s*)(.*?)(?="
            rf"^\s*Step\s*{n+1}\s*|^\s*Question\s*:|^\s*Final\s*Question\s*:|^\s*Updated\s*Question\s*:|\Z)"
        )
        match = re.search(step_regex, text, re.MULTILINE)
        if match:
            parsed[f'step_{n}'] = match.group(1).strip()
            
    # 3.1) Extract Step 1, Step 2, Step 3 (if present)
    #    For example: "Step 1 (Pick Words):\n....\nStep 2 (Draft Question): ..."
    #    We'll use a simple pattern that looks for lines starting with "Step X" up to the next "Step" or the end.
    step_regex = re.compile(
        r"(?ims)(Step\s*1\s*\(?.*?\)?:\s*)(.*?)(?=\n\s*Step\s*2\s*\(?|$)"
    )
    match = step_regex.search(text)
    if match:
        parsed['step_1'] = match.group(2).strip()

    step_regex = re.compile(
        r"(?ims)(Step\s*2\s*\(?.*?\)?:\s*)(.*?)(?=\n\s*Step\s*3\s*\(?|$)"
    )
    match = step_regex.search(text)
    if match:
        parsed['step_2'] = match.group(2).strip()

    step_regex = re.compile(
        r"(?ims)(Step\s*3\s*\(?.*?\)?:\s*)(.*)"
    )
    match = step_regex.search(text)
    if match:
        parsed['step_3'] = match.group(2).strip()

    ################################################################
    # 4) Extract the final question from known headings
    ################################################################
    # "Final Question:", "Updated Question:", "Final MCQ:",
    # "My Final Question is:", fallback "Question:"
    question_pattern = re.compile(
        r"(?is)(?:"
        r"(Final\s+Question\s*:\s*)"
        r"|(Updated\s+Question\s*:\s*)"
        r"|(Final\s*MCQ\s*:\s*)"
        r"|(My\s+Final\s+Question\s+is\s*:\s*)"
        r"|(Question\s*:\s*)"
        r")(.+)",
        re.IGNORECASE
    )
    question_match = question_pattern.search(text)
    if question_match:
        remainder = question_match.group(len(question_match.groups()))
        if remainder:
            splitted = re.split(
                r"\n\s*(?:A\)|A\.|A:|Correct\s*Answer\s*:)",
                remainder,
                maxsplit=1,
                flags=re.IGNORECASE
            )
            parsed['question'] = splitted[0].strip() if splitted else remainder.strip()

    ################################################################
    # 5) Fallback: "Question:" if we didn't parse it above
    ################################################################
    if not parsed['question']:
        q_fallback = re.search(r"(?i)Question\s*:\s*(.*)", text, re.DOTALL)
        if q_fallback:
            remainder = q_fallback.group(1)
            splitted = re.split(
                r"\n\s*(?:A\)|A\.|A:|Correct\s*Answer\s*:)",
                remainder,
                maxsplit=1,
                flags=re.IGNORECASE
            )
            parsed['question'] = splitted[0].strip() if splitted else remainder.strip()

    ################################################################
    # 6) If STILL no question => find the first line ending with '?'
    ################################################################
    if not parsed['question']:
        question_line_match = re.search(r"^(.*\?)\s*$", text, re.MULTILINE)
        if question_line_match:
            parsed['question'] = question_line_match.group(1).strip()

    ################################################################
    # 7) Extract choices A), B), C)
    ################################################################
    # We'll allow the pattern to continue until the next choice, a "Correct Answer:", 
    # a blank line, or the end of text (\Z). That way we capture lines like
    #   C) Un- (Correct answer)
    # even if there's no trailing newline.
    def capture_choice(label, full_text):
        pattern = (
            rf"(?ims)^[ \t]*{label}\s*[\)\.:]\s*(.*?)(?="
            rf"^[ \t]*[{chr(ord(label)+1)}]\s*[\)\.:]|"
            rf"^[ \t]*Correct\s*Answer\s*:|"
            rf"^$|"
            rf"\Z)"
        )
        m = re.search(pattern, full_text, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return None

    parsed['choice_a'] = capture_choice('A', text)
    parsed['choice_b'] = capture_choice('B', text)
    parsed['choice_c'] = capture_choice('C', text)

    ################################################################
    # 8) Check for "Correct Answer:" or "Answer:" or 
    #    "The correct answer is:" lines
    ################################################################
    ans_match = re.search(
        r"(?i)(?:Correct\s*Answer\s*:\s*|Answer\s*:\s*|The\s*correct\s*answer\s*is\s*)(.*)",
        text
    )
    if ans_match:
        ans_raw = ans_match.group(1).strip()
        # If the answer text starts with something like "C) " or "C. " or "C:", strip that off
        ans_no_label = re.sub(r'^[ABCabc]\s*[\)\.:]\s*', '', ans_raw).strip()
        parsed['correct_answer'] = ans_no_label

    ################################################################
    # 9) If STILL no correct_answer => check if any choice ends with 
    #    parentheses containing "correct", e.g. (Correct answer)
    ################################################################
    if not parsed['correct_answer']:
        for label in ['a', 'b', 'c']:
            choice_key = f'choice_{label}'
            val = parsed[choice_key]
            if not val:
                continue
            paren_match = re.search(r'\(\s*(.*?)\s*\)\s*$', val)
            if paren_match:
                # e.g. "Un- (Correct answer)"
                paren_text = paren_match.group(1)
                if re.search(r'correct', paren_text, re.IGNORECASE):
                    # remove that parenthetical from the choice
                    val_no_paren = re.sub(r'\(\s*.*?\s*\)\s*$', '', val).strip()
                    parsed[choice_key] = val_no_paren
                    parsed['correct_answer'] = val_no_paren
                    break

    return parsed


    
if __name__ == "__main__":
    test_api_access()
    #model, tokenizer= train_gemma()
    #model_path = "/blue/babajani.a/babak.ahmadi/NLP_Dorr/Project/MA/last_gemma_morphology_20250404_014910/final_model"
    # common_words = [
    # "the", "be", "to", "of", "and", "in", "that", "have", "I", 
    # "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    # "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    # "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    # "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    # "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    # "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    # "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    # "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    # "even", "new", "want", "because", "any", "these", "give", "day", "most", "us","none" ,"an","like","with"]
    
    
    data_file = 'MC_data_MA2.csv'
    
    strategies = [
        'chain_of_thought_plus_sequential_rl',
        'chain_of_thought_plus_role_chain',
        'chain_of_thought',
        'chain_of_thought_plus_sequential',
        'few_shot',
        'zero_shot'
    ]

    data = pd.read_csv(data_file, encoding='utf-8')
    NUM_QUESTIONS = 3
    word_difficulties = [1,2,3,4,5] #data["Word_Difficulty"].unique()
    task_difficulties = ['E', 'H', 'M']
    # Question_Type_array = np.array(data["Question_Type"].unique())
    # questio_id = [int(float(x)) for x in Question_Type_array]
    questio_id = list(range(1,14))
    for strategy in strategies:
        questions_data = []
        # for question_type in questio_id:
        #     if question_type not in [4,5]:
        #         prev_prompts = {q_type: [common_words] for q_type in questio_id}
        #     else:
        prev_prompts = {q_type: [] for q_type in questio_id}

        for word_difficulty in word_difficulties:
            for task_difficulty in task_difficulties:
                for question_type in questio_id:
                    for i in range(NUM_QUESTIONS):
                        print(f"\n=== Question #{i+1} === Question Type #{question_type} === {strategy} ===")
                        prompt_or_final_text, new_word = prompt_generator(
                            question_type,
                            word_difficulty,
                            task_difficulty,
                            data,
                            strategy,
                            prev_prompts[question_type]
                        )
                        # 2) 
                        if strategy == "chain_of_thought_plus_sequential_rl":
                            # The function already did multi-step GPT calls. 
                            # So use the combined text returned from prompt_generator:
                            generated_text = prompt_or_final_text
                            parse_text = generated_text
                        else:
                            # single-step strategy
                            generated_text = generate_gpt(prompt_or_final_text)
                            parse_text = generated_text
                        #parsed = parse_question_output(parse_text or "")
                        # 3) Parse the relevant text
                        if generated_text:
                            parsed = parse_question_output(generated_text)
                            if parsed:
                                parsed['generated_text'] = generated_text
                                parsed['question_type'] = question_type
                                parsed['word_difficulty'] = word_difficulty
                                parsed['task_difficulty'] = task_difficulty
                                parsed['prompting_strategy'] = strategy
                                questions_data.append(parsed)

                        if new_word:
                            prev_prompts[question_type].append(new_word)

                        if generated_text:
                            print("\nGenerated Text from GPT-3.5 Model:")
                            print(generated_text)
                        else:
                            print("No text returned by the model.")

        # Save after finishing one strategy
        df = pd.DataFrame(questions_data)
        filename = f'Final_generated_questions_GPT_{strategy}.csv'
        df.to_csv(filename, index=False)
        print(f"\nSaved questions for strategy '{strategy}' to {filename}")    
    
