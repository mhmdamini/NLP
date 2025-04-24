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

# Define HF token globally so it's accessible everywhere
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN  # Set environment variable for token

class MorphologyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Structured prompt format
        instruction = (
            "You are answering a morphology question. "
            "Begin your answer with 'Answer: Choice X' for multiple choice or 'Answer: [text]' for open-ended questions. "
            "Provide a clear explanation afterward."
        )
        
        task_type = "Multiple choice" if pd.notna(item['Choice_1']) else "Open-ended"
        
        # Input format with clear separation
        input_text = (
            f"{instruction}\n\n"
            f"# Question Information\n"
            f"- Type: {task_type}\n"
            f"- Task: {item['Task']}\n"
            f"- Category: {item['Category']}\n"
            f"- Word: {item['Word']}\n"
            f"- Question: {item['Instruction']}\n"
        )
        
        if pd.notna(item['Choice_1']):
            choices = []
            choice_num = 1
            input_text += "\n# Available Choices\n"
            while True:
                choice_key = f'Choice_{choice_num}'
                if choice_key not in item or pd.isna(item[choice_key]):
                    break
                input_text += f"- Choice {choice_num}: {item[choice_key]}\n"
                choice_num += 1
        
        # Clear delimiter between input and expected output
        input_text += "\n# Your Answer:\n"
        
        # Target output format
        if pd.notna(item['Choice_1']):
            correct_choice = item[f'Choice_{item["Correct_Answer"]}']
            target_text = (
                f"Answer: Choice {item['Correct_Answer']}. "
                f"{correct_choice} is correct because it demonstrates the {item['Category'].lower()} "
                f"concept. In the word '{item['Word']}', we can identify the {item['Task'].lower()} "
                f"through proper morphological analysis. This is a key concept in understanding "
                f"how words are formed and structured in English."
            )
        else:
            target_text = (
                f"Answer: {str(item['Correct_Answer'])}. "
                f"This demonstrates the {item['Category'].lower()} concept in '{item['Word']}'. "
                f"When analyzing how this word {item['Task'].lower()}, we can see the morphological "
                f"principles at work. This helps us understand the structure and formation of words."
            )

        # Combine input and target with EOS token
        full_text = f"{input_text}{target_text}</s>"
        
        # Create encodings
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Create labels with masked loss for prompt
        input_only = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        input_length = input_only['input_ids'].shape[1]
        labels = encodings['input_ids'].clone()
        
        # Set prompt part to -100 to ignore in loss calculation
        labels[:, :input_length] = -100

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def prepare_data(csv_path):
    """Prepare and split the data for training with data augmentation"""
    df = pd.read_csv(csv_path)
    
    # Convert necessary columns to string
    for col in ['Correct_Answer', 'Word_Difficulty', 'Task_Difficulty']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Simple data augmentation: create small variations in questions
    augmented_data = []
    for _, row in df.iterrows():
        augmented_data.append(row.to_dict())  # Original row
        
        # Only augment if it's a multiple choice question
        if pd.notna(row.get('Choice_1', pd.NA)):
            # Variation 1: Slightly different instruction wording
            variation = row.to_dict()
            orig_instruction = variation['Instruction']
            
            if "what is" in orig_instruction.lower():
                variation['Instruction'] = orig_instruction.lower().replace("what is", "identify").capitalize()
                augmented_data.append(variation)
            elif "identify" in orig_instruction.lower():
                variation['Instruction'] = orig_instruction.lower().replace("identify", "what is").capitalize()
                augmented_data.append(variation)
    
    # Convert back to DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    
    # Split data with stratification
    try:
        train_df, val_df = train_test_split(
            augmented_df, 
            test_size=0.15,
            random_state=42,
            stratify=augmented_df[['Category', 'Task']].apply(lambda x: f"{x['Category']}_{x['Task']}", axis=1)
        )
    except ValueError:
        # Fallback to stratifying by just Category
        train_df, val_df = train_test_split(
            augmented_df, 
            test_size=0.15, 
            random_state=42, 
            stratify=augmented_df['Category']
        )
    
    print(f"Original data size: {len(df)}")
    print(f"Augmented data size: {len(augmented_df)}")
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    
    return train_df, val_df

def monitor_gpu_memory(message):
    """Helper function to monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"{message}: Allocated: {allocated:.2f} GB, Max: {max_allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def save_config_locally(model_name, output_dir):
    """Save model config locally to avoid authentication issues during training"""
    from huggingface_hub import hf_hub_download
    import shutil
    
    # Create directory structure
    base_path = os.path.join(output_dir, "base_model_config")
    os.makedirs(base_path, exist_ok=True)
    
    try:
        # Download config file
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            token=HF_TOKEN
        )
        
        # Copy to our directory
        shutil.copy(config_path, os.path.join(base_path, "config.json"))
        print(f"Config saved locally to {base_path}")
        
        return True
    except Exception as e:
        print(f"Error saving config locally: {e}")
        return False

def train_gemma_model(train_df, val_df, model_name="google/gemma-2b-it", output_dir="gemma_morphology", use_lora=True):
    """Train Gemma model with advanced techniques"""
    # Clean memory
    gc.collect()
    torch.cuda.empty_cache()
    
    monitor_gpu_memory("Initial GPU state")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config locally to avoid authentication issues
    save_config_locally(model_name, output_dir)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    
    # Save tokenizer locally to avoid authentication issues
    tokenizer_save_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_save_path)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model with memory optimizations
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,  # Always use float32 to avoid gradient issues
        device_map="auto"
    )
    
    monitor_gpu_memory("After model loading")
    
    # Apply LoRA for more efficient fine-tuning if requested
    if use_lora:
        print("Applying LoRA adapters...")
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,  # scaling factor
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Enable gradient checkpointing for full fine-tuning
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    monitor_gpu_memory("After model adaptation")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MorphologyDataset(train_df, tokenizer, max_length=512)
    val_dataset = MorphologyDataset(val_df, tokenizer, max_length=512)
    
    monitor_gpu_memory("After dataset creation")

    # Training arguments
    batch_size = 2 if use_lora else 1
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=12,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        learning_rate=3e-5 if use_lora else 2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=30,
        save_steps=30,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=4 if use_lora else 8,
        fp16=False,  # DISABLE fp16 to avoid gradient issues
        bf16=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=not use_lora,  # Enable for full fine-tuning only
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        report_to="none",  # Disable reporting to save memory
        run_name=f"gemma_morphology_{datetime.now().strftime('%Y%m%d_%H%M')}",
        hub_token=HF_TOKEN,  # Add token for Hugging Face API calls
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    monitor_gpu_memory("Before trainer initialization")

    # Initialize trainer with callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if not improving
    )
    
    monitor_gpu_memory("After trainer initialization")

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    print("Saving model...")
    model_save_path = os.path.join(output_dir, "final_model")
    
    if use_lora:
        # For LoRA, we save the adapter
        model.save_pretrained(model_save_path, token=HF_TOKEN)
    else:
        # For full model, save everything
        trainer.save_model(model_save_path)
    
    # Save tokenizer from the local copy to avoid authentication issues
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Model saved to {model_save_path}")

    return model, tokenizer

def test_model(model, tokenizer, test_questions):
    """Test the model on a set of questions"""
    model.eval()
    results = []
    
    for question in test_questions:
        # Structured prompt format
        instruction = (
            "You are answering a morphology question. "
            "Begin your answer with 'Answer: Choice X' for multiple choice or 'Answer: [text]' for open-ended questions. "
            "Provide a clear explanation afterward."
        )
        
        # Format input
        input_text = (
            f"{instruction}\n\n"
            f"# Question Information\n"
            f"- Type: Multiple choice\n"
            f"- Task: {question['Task']}\n"
            f"- Category: {question['Category']}\n"
            f"- Word: {question['Word']}\n"
            f"- Question: {question['Instruction']}\n"
            f"\n# Available Choices\n"
        )
        
        # Add choices
        for i, choice in enumerate(question['Choices'], 1):
            input_text += f"- Choice {i}: {choice}\n"
        
        # Add answer prompt
        input_text += "\n# Your Answer:\n"
        
        # Generation settings
        generation_config = {
            'max_new_tokens': 200,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.92,
            'top_k': 50,
            'repetition_penalty': 1.2,
        }
        
        # Generate answer
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **generation_config
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if input_text in generated_text:
            answer = generated_text[len(input_text):].strip()
        else:
            answer = generated_text
            
        results.append({
            'question': question,
            'input': input_text,
            'generated': answer
        })
    
    return results

def main():
    # Configure paths
    data_path = 'MC_data_MA2.csv'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"gemma_morphology"#_{timestamp}"
    
    # Display GPU information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Prepare data
    print("\nPreparing data...")
    train_df, val_df = prepare_data(data_path)
    
    # Define whether to use LoRA (recommended for better memory efficiency)
    use_lora = True  # Set to False for full fine-tuning if you have enough GPU memory
    
    # Train model
    print(f"\nTraining Gemma model {'with LoRA' if use_lora else ''}...")
    model, tokenizer = train_gemma_model(train_df, val_df, output_dir=output_dir, use_lora=use_lora)
    
    # Test questions
    test_questions = [
        {
            "Task": "Identify",
            "Category": "Derivation",
            "Word": "happiness",
            "Instruction": "What is the base word and suffix in 'happiness'?",
            "Choices": [
                "base: happy, suffix: -ness",
                "base: happ, suffix: -iness",
                "base: happi, suffix: -ness",
                "base: hap, suffix: -piness"
            ]
        },
        {
            "Task": "Analyze",
            "Category": "Compounding",
            "Word": "blackboard",
            "Instruction": "Identify the type of compound word in 'blackboard'.",
            "Choices": [
                "Endocentric compound",
                "Exocentric compound",
                "Copulative compound",
                "Appositional compound"
            ]
        }
    ]
    return model, tokenizer
    # Test the model
    print("\nTesting model on sample questions...")
    results = test_model(model, tokenizer, test_questions)
    
    # Print results
    print("\nSample Generated Answers:")
    for i, result in enumerate(results, 1):
        print(f"\nExample {i}:")
        print(f"Question: {result['question']['Instruction']}")
        print(f"Word: {result['question']['Word']}")
        print(f"Generated Answer: {result['generated']}")
        print("-" * 80)
    
if __name__ == "__main__":
    # Clean memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    main()