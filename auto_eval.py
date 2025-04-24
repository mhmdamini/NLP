!pip install language-tool-python spacy transformers torch nltk pandas openpyxl tqdm
import spacy
import language_tool_python
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import nltk
import pandas as pd
from tqdm import tqdm
# Download required data
nltk.download('punkt')
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    !python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
import subprocess

def install_java_if_needed():
    try:
        # Check if Java is already installed
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Java is already installed.")
            return
    except FileNotFoundError:
        print("Java not found. Installing...")

    # Update and install Java 17
    print("Installing OpenJDK 17...")
    subprocess.run(['sudo', 'apt', 'update'], check=True)
    subprocess.run(['sudo', 'apt', 'install', 'openjdk-17-jdk', '-y'], check=True)

    # Configure Java alternatives
    subprocess.run([
        'sudo', 'update-alternatives', '--install',
        '/usr/bin/java', 'java',
        '/usr/lib/jvm/java-17-openjdk-amd64/bin/java', '1'
    ], check=True)
    subprocess.run([
        'sudo', 'update-alternatives', '--set',
        'java', '/usr/lib/jvm/java-17-openjdk-amd64/bin/java'
    ], check=True)

    # Verify installation
    subprocess.run(['java', '-version'], check=True)

# Call the installation function
install_java_if_needed()


    
class GrammarEvaluator:
    def __init__(self):
        """Initialize the grammar evaluator with necessary tools"""
        try:
            self.language_tool = language_tool_python.LanguageTool('en-US')
            self.nlp = spacy.load('en_core_web_sm')
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.gpt2_model.to(self.device)
            print("Successfully initialized all components")
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            raise

    def calculate_error_density(self, text):
        try:
            matches = self.language_tool.check(text)
            word_count = len(text.split())
            error_count = len(matches)
            error_details = []
            for match in matches:
                error_details.append({
                    'message': match.message,
                    'replacements': match.replacements[:3],
                    'context': match.context,
                    'category': match.category
                })
            return {
                'error_count': error_count,
                'word_count': word_count,
                'error_density': (error_count / word_count * 100) if word_count > 0 else 0,
                'error_details': error_details
            }
        except Exception as e:
            print(f"Error calculating error density: {str(e)}")
            return None

    def calculate_complexity_metrics(self, text):
        try:
            doc = self.nlp(text)
            words = [token.text for token in doc if not token.is_punct]
            sentences = list(doc.sents)

            def get_depth(token):
                return 1 + max([get_depth(child) for child in token.children], default=0)

            metrics = {
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': np.mean([len([token for token in sent if not token.is_punct])
                                             for sent in sentences]) if sentences else 0,
                'unique_words_ratio': len(set(words)) / len(words) if words else 0,
                'pos_distribution': dict(nltk.FreqDist([token.pos_ for token in doc])),
                'avg_syntax_depth': np.mean([get_depth(sent.root) for sent in sentences]) if sentences else 0
            }
            return metrics
        except Exception as e:
            print(f"Error calculating complexity metrics: {str(e)}")
            return None

    def calculate_readability(self, text):
        try:
            def count_syllables(word):
                word = word.lower()
                count = 0
                vowels = 'aeiouy'
                if word[0] in vowels:
                    count += 1
                for index in range(1, len(word)):
                    if word[index] in vowels and word[index-1] not in vowels:
                        count += 1
                if word.endswith('e'):
                    count -= 1
                if count == 0:
                    count += 1
                return count

            doc = self.nlp(text)
            words = [token.text for token in doc if not token.is_punct]
            sentences = list(doc.sents)

            word_count = len(words)
            sentence_count = len(sentences)
            syllable_count = sum(count_syllables(word) for word in words)
            complex_words = len([word for word in words if count_syllables(word) >= 3])

            if word_count > 0 and sentence_count > 0:
                flesch = 206.835 - 1.015 * (word_count/sentence_count) - 84.6 * (syllable_count/word_count)
                gunning_fog = 0.4 * ((word_count/sentence_count) + 100 * (complex_words/word_count))
            else:
                flesch = 0
                gunning_fog = 0

            return {
                'flesch_reading_ease': flesch,
                'gunning_fog_index': gunning_fog,
                'avg_syllables_per_word': syllable_count/word_count if word_count > 0 else 0,
                'complex_word_ratio': complex_words/word_count if word_count > 0 else 0
            }
        except Exception as e:
            print(f"Error calculating readability: {str(e)}")
            return None

    def calculate_perplexity(self, text):
        try:
            encodings = self.gpt2_tokenizer(text, return_tensors='pt')
            input_ids = encodings.input_ids.to(self.device)
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=input_ids)
                loss = outputs.loss
            return torch.exp(loss).item()
        except Exception as e:
            print(f"Error calculating perplexity: {str(e)}")
            return None

    def evaluate(self, text):
        try:
            error_metrics = self.calculate_error_density(text)
            complexity_metrics = self.calculate_complexity_metrics(text)
            readability_metrics = self.calculate_readability(text)
            perplexity = self.calculate_perplexity(text)

            error_score = max(0, 1 - error_metrics['error_density']/10)
            complexity_score = min(1, complexity_metrics['avg_syntax_depth']/5)
            readability_score = min(1, max(0, readability_metrics['flesch_reading_ease']/100))
            perplexity_score = max(0, 1 - (perplexity-20)/100)

            final_score = (error_score * 0.4 +
                         complexity_score * 0.2 +
                         readability_score * 0.2 +
                         perplexity_score * 0.2) * 100

            return {
                'final_score': final_score,
                'subscores': {
                    'grammar': error_score * 100,
                    'complexity': complexity_score * 100,
                    'readability': readability_score * 100,
                    'fluency': perplexity_score * 100
                },
                'detailed_metrics': {
                    'error_metrics': error_metrics,
                    'complexity_metrics': complexity_metrics,
                    'readability_metrics': readability_metrics,
                    'perplexity': perplexity
                }
            }
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return None

# Process Excel file
input_file = '/content/drive/MyDrive/MC_Data/generated_items/GPT.xlsx'
output_file = '/content/drive/MyDrive/MC_Data/generated_items/evaluation/All_str_GPT.xlsx'

# Initialize evaluator
evaluator = GrammarEvaluator()

try:
    # Read Excel file
    print(f"Reading file: {input_file}")
    df = pd.read_excel(input_file)

    # Concatenate specified columns
    print("Processing sentences...")
    df['combined_text'] = df[['Text']].fillna('').astype(str).apply(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].astype(str).str.capitalize()
    df['question'] = df['Text'].astype(str).str.capitalize()

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['combined_text'].strip()
        if text:
            evaluation = evaluator.evaluate(text)
            if evaluation:
                results.append({
                    'Row': idx + 2,  # Excel row number (1-based + header)
                    'Prompting_strategy':row.get('strategy', ''),
                    'Question_Type':row.get('question_type', ''),
                    'Question':row.get('question',''),
                    'Correct_Answer': row.get('correct_answer', ''),
                    'Choice_1': row.get('choice_a', ''),
                    'Choice_2': row.get('choice_b', ''),
                    'Choice_3': row.get('choice_c', ''),
                    'Word_Difficulty': row.get('word_difficulty', ''),
                    'Task_Difficulty': row.get('task_difficulty', ''),
                    'Text': text,
                    'Overall_Score': evaluation['final_score'],
                    'Grammar_Score': evaluation['subscores']['grammar'],
                    'Complexity_Score': evaluation['subscores']['complexity'],
                    'Readability_Score': evaluation['subscores']['readability'],
                    'Fluency_Score': evaluation['subscores']['fluency'],
                    'Error_Count': evaluation['detailed_metrics']['error_metrics']['error_count'],
                    'Errors': '; '.join([error['message'] for error in
                                       evaluation['detailed_metrics']['error_metrics']['error_details']])
                })

    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)

    print("\nProcessing complete!")
    print(f"Total sentences processed: {len(results_df)}")
    print(f"Average score: {results_df['Overall_Score'].mean():.2f}")
    print(f"Total errors found: {results_df['Error_Count'].sum()}")

    # Display sample results
    print("\nSample results (first 5 rows):")
    print(results_df[['Row', 'Overall_Score', 'Error_Count', 'Errors']].head())

except Exception as e:
    print(f"Error processing file: {str(e)}")