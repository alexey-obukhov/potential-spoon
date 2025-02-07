import pandas as pd
import json
import re
import html
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
from torch.cuda.amp import autocast
from sklearn.metrics.pairwise import cosine_similarity
from school_logging.log import ColoredLogger
import multiprocessing

# Load the model and tokenizer using Auto classes
model_name = "t5-base"  # You can replace this with any Seq2Seq model like "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Check for CUDA availability and set the device
# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Counsel Chat dataset for fine-tuning.')
    parser.add_argument('--filepath', type=str, default='https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv',
                        help='Path to the Counsel Chat CSV file.')
    parser.add_argument('--output_filepath', type=str, default='counsel_chat_formatted_paraphrased_gemini.json',
                        help='Path to save the formatted JSON data.')
    parser.add_argument('--log_level', type=str.upper, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode to process only the first question.')
    parser.add_argument('--num_processes', type=int, default=1,  # multiprocessing.cpu_count(),
                        help='Number of processes to use for multiprocessing. Default is CPU count.')
    return parser.parse_args()

def clean_text(text, logger):
    """Cleans text with more advanced techniques."""
    logger.debug(f"Cleaning text: '{text[:50]}...'")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\u2019", "'", text)
    text = re.sub(r"[^a-zA-Z\s']", "", text)
    text = re.sub(r"\s+", " ", text)

    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    cleaned_text = " ".join(cleaned_tokens)
    logger.debug(f"Cleaned text: '{cleaned_text[:50]}...'")
    return cleaned_text.strip()

def back_translate_with_marian(text, logger, lang="fr", tokenizer=None, model=None, device="cuda"):
    """Back-translates text using MarianMT models (takes tokenizer and model)."""
    logger.debug(f"Back-translating text: '{text[:50]}...' to language: {lang}")

    if tokenizer is None or model is None:  # Check for None
        logger.error(f"Tokenizer or model is None for language: {lang}")
        return None

    try:
        # Ensure model is on the correct device
        model = model.to(device)  # Move the model to the device

        # Translate from English to the target language
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated_ids = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

        # Translate from the target language back to English
        inputs_back = tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        back_translated_ids = model.generate(**inputs_back)
        back_translated_text = tokenizer.batch_decode(back_translated_ids, skip_special_tokens=True)[0]

        logger.debug(f"Back-translated text: '{back_translated_text[:50]}...'")
        return back_translated_text
    except Exception as e:
        logger.error(f"Error during back-translation with MarianMT: {e}")
        return None

def paraphrase(text, logger, num_return_sequences=2, max_length=96, tokenizer=None, model=None):
    """Generates paraphrases using a pre-trained T5 model (takes tokenizer and model)."""
    logger.debug(f"Paraphrasing text: '{text[:50]}...'")

    if tokenizer is None or model is None: # Check for None
        logger.error("Tokenizer or model is None for Paraphrase.")
        return []

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
        generated_ids = model.generate(
            input_ids=input_ids,
            num_return_sequences=num_return_sequences,
            num_beams=10,  # keep this low for testing
            max_length=max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        paraphrases = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        logger.debug(f"Generated paraphrases: {[p[:50] for p in paraphrases]}")
        return paraphrases
    except Exception as e:
        logger.error(f"Error in Paraphrase: {e}")
        return [] # return empty if errors

def is_similar(text1, text2, logger, threshold=0.8):
    """Checks if two texts are similar based on a normalized edit distance."""
    logger.debug(f"Checking similarity between: '{text1[:50]}...' and '{text2[:50]}...'")
    distance = edit_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    similarity = 1 - (distance / max_len) if max_len > 0 else 0
    logger.debug(f"Similarity score: {similarity}")
    return similarity >= threshold

def get_embedding(text):
    """Generates a dense vector (embedding) for a given text using AutoModelForSeq2SeqLM."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.encoder(**inputs)  # Use the encoder part of the model for embeddings
    # Using the mean of the last hidden state as the embedding (you can experiment with other strategies)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling across tokens
    return embedding

def is_similar(text1, text2, logger, threshold=0.8):
    """Checks if two texts are similar based on cosine similarity of their embeddings."""
    logger.debug(f"Checking similarity between: '{text1[:50]}...' and '{text2[:50]}...'")
    # Get embeddings for both texts
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    # Compute cosine similarity
    similarity = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]
    logger.debug(f"Cosine similarity score: {similarity}")
    return similarity >= threshold

def create_context(df, question_ids, logger):
    """Creates a conversational context for the specified question IDs."""
    logger.debug(f"Creating context for question IDs: {question_ids}")
    context_data = []
    for q_id in question_ids:
        subset = df[df['questionID'] == q_id].copy()
        subset['interactionID'] = range(len(subset))  # Create a unique ID for each interaction
        context = []
        for _, row in subset.iterrows():
            context_data.append({
                'questionID': q_id,
                'interactionID': row['interactionID'],
                'questionText': row['questionText'],
                'answerText': row['answerText'],
                'context': context.copy()
            })
            context.append({"speaker": "user", "text": row['questionText']})
            context.append({"speaker": "bot", "text": row['answerText']})
    return context_data

def process_chunk(chunk_df, logger):
    """Processes a chunk of data, generating variations and creating data points."""
    chunk_data = []

    # Initialize paraphrase model and tokenizer
    paraphrase_tokenizer = AutoTokenizer.from_pretrained(model_name)
    paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    marian_model_names = {
        "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
        "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
        "es": ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"),
        "it": ("Helsinki-NLP/opus-mt-en-it", "Helsinki-NLP/opus-mt-it-en"),
        "nl": ("Helsinki-NLP/opus-mt-en-nl", "Helsinki-NLP/opus-mt-nl-en"),
        "ru": ("Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"),
        "zh": ("Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en")}

    # Load models and tokenizers for each language
    marian_models = {
        lang: {
            'forward': MarianMTModel.from_pretrained(forward_model).to(device),
            'reverse': MarianMTModel.from_pretrained(reverse_model).to(device)
        }
        for lang, (forward_model, reverse_model) in marian_model_names.items()
    }

    marian_tokenizers = {
        lang: {
            'forward': MarianTokenizer.from_pretrained(forward_model),
            'reverse': MarianTokenizer.from_pretrained(reverse_model)
        }
        for lang, (forward_model, reverse_model) in marian_model_names.items()
    }

    # Process each row in the dataframe
    for _, row in chunk_df.iterrows():
        variations = []
        potential_variations = []

        # Translate each question in the row for every language
        for lang in marian_model_names.keys():
            # Prepare input for translation (forward: English -> target language)
            input_text = row['questionText']
            inputs = marian_tokenizers[lang]['forward']([input_text], return_tensors="pt", padding=True, truncation=True).to(device)

            # Forward translation (English -> target language)
            translated_ids = marian_models[lang]['forward'].generate(**inputs)
            translation = marian_tokenizers[lang]['forward'].decode(translated_ids[0], skip_special_tokens=True)

            # Reverse translation (target language -> English)
            reverse_inputs = marian_tokenizers[lang]['reverse']([translation], return_tensors="pt", padding=True, truncation=True).to(device)
            translated_back_ids = marian_models[lang]['reverse'].generate(**reverse_inputs)
            translated_back = marian_tokenizers[lang]['reverse'].decode(translated_back_ids[0], skip_special_tokens=True)

            # Add reverse translation to potential variations
            potential_variations.append(translated_back)

        # Generate paraphrases using T5 (assuming paraphrase function is still in use)
        paraphrases = paraphrase(row['questionText'], logger, num_return_sequences=2, max_length=96, tokenizer=paraphrase_tokenizer, model=paraphrase_model)
        potential_variations.extend(paraphrases)

        # Filter out similar variations
        for variation in potential_variations:
            if variation and not is_similar(row['questionText'], variation, logger):
                if not any(is_similar(variation, existing_variation, logger) for existing_variation in variations):
                    variations.append(variation)

        # Store the data point
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
            }
        }
        chunk_data.append(data_point)

    # Clear memory after processing
    del marian_models
    del marian_tokenizers
    del paraphrase_model
    del paraphrase_tokenizer
    torch.cuda.empty_cache()

    return chunk_data

def create_json_data(df, logger, num_processes):
    """Converts DataFrame to JSON, adding variations, using multiprocessing."""
    logger.info(f"Creating JSON data using {num_processes} processes...")

    question_ids = df['questionID'].unique()
    chunks = [question_ids[i::num_processes] for i in range(num_processes)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        context_results = pool.starmap(create_context, [(df, chunk, logger) for chunk in chunks])

    context_df = pd.DataFrame([item for sublist in context_results for item in sublist])
    df = df.drop(columns=['questionText', 'answerText'])
    merged_df = pd.merge(context_df, df, on='questionID', how='left')
    merged_df = merged_df.drop_duplicates(subset=['questionID', 'interactionID'])

    chunks = [merged_df[i::num_processes] for i in range(num_processes)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, [(chunk, logger) for chunk in chunks])

    json_data = [item for sublist in results for item in sublist]
    logger.info("JSON data creation complete.")
    return json_data

def prepare_counsel_chat_data(filepath, output_filepath, logger, test_mode, num_processes):
    """Loads, cleans, formats, and saves the Counsel Chat dataset."""
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    if test_mode:
        first_question_id = df['questionID'].unique()[1]
        df = df[df['questionID'] == first_question_id]
        logger.info("Test mode enabled. Processing only the first question.")

    df['questionText'] = df['questionText'].apply(lambda text: clean_text(text, logger))
    df['answerText'] = df['answerText'].apply(lambda text: clean_text(text, logger))

    json_data = create_json_data(df, logger, num_processes)

    logger.info(f"Saving formatted data to: {output_filepath}")
    with open(output_filepath, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Data preparation complete. Formatted data saved to {output_filepath}")
    print("Remember: This dataset is licensed under CC BY-NC 4.0. Non-commercial use only!")

def main():
    """Parses arguments and runs the data preparation."""
    args = parse_args()
    logger = ColoredLogger('MyLogger', verbose=args.log_level)

    prepare_counsel_chat_data(args.filepath, args.output_filepath, logger, args.test_mode, args.num_processes)

if __name__ == "__main__":
    main()