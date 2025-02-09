"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

This script is designed to prepare and augment the Counsel Chat dataset for use in training a psychological AI model.
The main goals of this script are to clean, tokenize, and lemmatize the text data, generate variations of therapeutic
responses using a language model, and create a JSON file containing the processed data.

Functionality:
1. Clean the text data by removing unwanted characters and normalizing it.
2. Tokenize and lemmatize the text data to preserve key pronouns and verbs.
3. Generate text embeddings for similarity checks.
4. Create conversational context for specified question IDs.
5. Process chunks of data to generate variations and create data points.
6. Use multiprocessing to efficiently create JSON data.

Challenges:
1. Ensuring the text data is cleaned and normalized without losing important information.
2. Tokenizing and lemmatizing the text data while preserving key pronouns and verbs.
3. Generating meaningful and relevant therapeutic responses using a language model.
4. Handling potential errors and exceptions during data processing.
5. Efficiently processing large datasets using multiprocessing.

Methods:
1. clean_text: Cleans text by removing unwanted characters and normalizing it.
2. tokenize_and_lemmatize: Tokenizes and lemmatizes text, preserving key pronouns/verbs.
3. get_embedding: Generates a text embedding.
4. is_similar: Checks cosine similarity between embeddings.
5. create_context: Creates conversational context for the specified question IDs.
6. process_chunk: Processes a chunk of data, generating variations and creating data points.
7. create_json_data: Creates JSON data using multiprocessing.
"""

import pandas as pd
import json
import re
import html
import gc
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utilities.therapeutic_promt import prompt_templates
from utilities.keep_words import keep_words
from school_logging.log import ColoredLogger
from text_generator import TextGenerator
import torch.multiprocessing as mp
import spacy
import traceback
from typing import List, Dict, Any, Optional

# --- Load SpaCy (Outside Functions) ---
nlp = spacy.load("en_core_web_sm")

class DataAugmentation:
    def __init__(self, logger: ColoredLogger):
        self.logger = logger

    def clean_text(self, text: str) -> str:
        """
        Cleans text by removing unwanted characters and normalizing it, while preserving explicit ':' signs.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        text = html.unescape(text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)
        text = re.sub(r"\u2019", "'", text)
        text = re.sub(r"\u2014", "-", text)  # Replace em dash with hyphen
        text = re.sub(r"\u201c", '"', text)  # Replace left double quotation mark with double quote
        text = re.sub(r"\u201d", '"', text)  # Replace right double quotation mark with double quote
        text = re.sub(r"\u2026", "...", text)  # Replace ellipsis with three dots
        text = re.sub(r"[^a-zA-Z0-9\s.,?!'\":-]", "", text)  # Remove unwanted characters except ':'
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        """
        Tokenizes and lemmatizes text, preserving key pronouns/verbs.

        Args:
            text (str): The text to be tokenized and lemmatized.

        Returns:
            str: The tokenized and lemmatized text.
        """
        try:
            self.logger.debug("Tokenizing/lemmatizing: '%s...'", text[:50])
            doc = nlp(text)
            cleaned_tokens = [
                token.lemma_.lower() for token in doc
                if (token.lemma_.lower() in keep_words) or
                   (token.lemma_.lower() not in nlp.Defaults.stop_words and not token.is_punct and not token.is_space)
            ]
            cleaned_text = " ".join(cleaned_tokens)
            self.logger.debug("Tokenized/lemmatized text: '%s...'", cleaned_text[:50])
            return cleaned_text.strip()
        except Exception as exc:
            self.logger.error("Error in tokenize_and_lemmatize: '%s'\n'%s'", exc, traceback.format_exc())
            return ""

    def get_embedding(self, text: str, tokenizer: Any, model: Any, device: str) -> Optional[torch.Tensor]:
        """
        Generates a text embedding.

        Args:
            text (str): The text to be embedded.
            tokenizer (Any): The tokenizer to use.
            model (Any): The model to use for generating embeddings.
            device (str): The device to run the model on.

        Returns:
            Optional[torch.Tensor]: The generated embedding, or None if an error occurred.
        """
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding
        except Exception as exc:
            self.logger.error("Error in get_embedding: '%s'\n'%s'", exc, traceback.format_exc())
            return None

    def is_similar(self, text1: str, text2: str, tokenizer: Any, model: Any, device: str, threshold: float = 0.8) -> bool:
        """
        Checks cosine similarity between embeddings.

        Args:
            text1 (str): The first text to compare.
            text2 (str): The second text to compare.
            tokenizer (Any): The tokenizer to use.
            model (Any): The model to use for generating embeddings.
            device (str): The device to run the model on.
            threshold (float): The similarity threshold to consider texts as similar.

        Returns:
            bool: True if the texts are similar, False otherwise.
        """
        try:
            self.logger.debug("Checking similarity: '%s...' and '%s...'", text1[:50], text2[:50])
            embedding1 = self.get_embedding(text1, tokenizer, model, device)
            embedding2 = self.get_embedding(text2, tokenizer, model, device)
            if embedding1 is None or embedding2 is None:
                return False
            similarity = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]
            self.logger.debug("Cosine similarity: %s", similarity)
            return similarity >= threshold
        except Exception as exc:
            self.logger.error("Error in is_similar: '%s'\n'%s'", exc, traceback.format_exc())
            return False

    def create_context(self, df: pd.DataFrame, question_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Creates conversational context for the specified question IDs.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            question_ids (List[str]): The list of question IDs to create context for.

        Returns:
            List[Dict[str, Any]]: The list of context data.
        """
        self.logger.debug("Creating context for question IDs: %s", question_ids)
        context_data = []
        for q_id in question_ids:
            interactions = df[df['questionID'] == q_id].sort_values('interactionID')
            if not interactions.empty:
                context = []
                for _, row in interactions.iterrows():
                    context_entry = {
                        'questionID':    q_id,
                        'interactionID': row['interactionID'],
                        'questionText':  row['questionText'],
                        'answerText':    row['answerText'],
                        'context':       "\n".join(context)
                    }
                    context_data.append(context_entry)
                    context.append(f"User: {row['questionText']}")
                    context.append(f"Therapist: {row['answerText']}")
        return context_data

    def process_chunk(self, chunk_df: pd.DataFrame, model_name: str, device: str) -> List[Dict[str, Any]]:
        """
        Processes a chunk of data, generating variations and creating data points.

        Args:
            chunk_df (pd.DataFrame): The chunk of data to process.
            model_name (str): The name of the language model to use.
            device (str): The device to run the model on.

        Returns:
            List[Dict[str, Any]]: The list of processed data points.
        """
        chunk_data = []
        generator = None
        try:
            generator = TextGenerator(model_name, device, self.logger)

            for _, row in chunk_df.iterrows():
                variations = []

                # Generate Category Info, including the context and topic
                category_info = {}
                for cat, template in prompt_templates.items():
                    # 1. Prepare the prompt (including context and topic)
                    context = row['context']
                    text =    row['questionText']
                    topic =   row['topic']

                    # Tokenize and truncate context if needed
                    context_tokens = generator.tokenizer(context, return_tensors="pt", padding=False, truncation=False)['input_ids'][0]
                    max_context_length = 1024  # Maximum context length
                    if len(context_tokens) > max_context_length:
                        self.logger.warning("Truncating context for QID %s, IID %s", row['questionID'], row['interactionID'])
                        context_tokens = context_tokens[-max_context_length:]
                        context = generator.tokenizer.decode(context_tokens, skip_special_tokens=True)

                    prompt = template.format(text=text, context=context, topic=topic)

                    # 2. Generate text directly using generator.generate_text
                    generated_text = generator.generate_text(prompt)

                    # 3. Handle potential errors
                    if generated_text.startswith("Error:"):
                        self.logger.warning("Generation failed for category '%s'", cat)
                    category_info[cat] = generated_text  # Store result

                data_point = {
                    "context": row['context'],
                    "question": row['questionText'],
                    "variations": variations,
                    "answer": row['answerText'],
                    "metadata": {
                        "topic": row['topic'],
                        "questionTitle": row['questionTitle'],
                        "questionID": row['questionID'],
                        "interactionID": row['interactionID']
                    },
                    "categories": category_info
                }
                chunk_data.append(data_point)

            return chunk_data

        except Exception as exc:
            error_msg = traceback.format_exc()
            self.logger.error("Error in process_chunk: '%s'\n'%s'", exc, error_msg)
            return []
        finally:
            if generator is not None:
                del generator.model
                del generator.tokenizer
                del generator
            torch.cuda.empty_cache()
            gc.collect()

    def create_json_data(self, df: pd.DataFrame, num_processes: int, model_name: str, device: str) -> List[Dict[str, Any]]:
        """
        Creates JSON data using multiprocessing.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            num_processes (int): The number of processes to use.
            model_name (str): The name of the language model to use.
            device (str): The device to run the model on.

        Returns:
            List[Dict[str, Any]]: The list of JSON data.
        """
        self.logger.info("Creating JSON data with %d processes...", num_processes)
        try:
            df['interactionID'] = df.groupby('questionID').cumcount()
            context_data = self.create_context(df, df['questionID'].unique())
            context_df = pd.DataFrame(context_data)
            merged_df = pd.merge(df, context_df, on=['questionID', 'interactionID'], suffixes=['', '_context'], how='left')
            merged_df = merged_df.drop_duplicates(subset=['questionID', 'interactionID'])
            merged_df = merged_df.drop(columns=['questionText_context', 'answerText_context', 'context_y'], errors='ignore')
            merged_df.rename(columns={'context_x': 'context'}, inplace=True)

            chunks = [merged_df[i::num_processes] for i in range(num_processes)]
            results = []

            try:
                with mp.Pool(processes=num_processes) as pool:
                    # Pass model_name and device instead of generator
                    results = pool.starmap(self.process_chunk, [(chunk, model_name, device) for chunk in chunks])
                    pool.close()
                    pool.join()
            except Exception as exc:
                self.logger.error("Error in multiprocessing: '%s'\n'%s'", exc, traceback.format_exc())
            finally:
                if 'pool' in locals():
                    pool.terminate()

            json_data = [item for sublist in results for item in sublist]
            self.logger.info("JSON data creation complete.")
            return json_data

        except Exception as exc:
            self.logger.error("Error in create_json_data: '%s'\n'%s'", exc, traceback.format_exc())
            return []
