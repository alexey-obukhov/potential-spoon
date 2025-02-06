import pandas as pd
import json
import re
import html
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import torch
import argparse
import multiprocessing
from school_logging.log import ColoredLogger

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Counsel Chat dataset for fine-tuning.')
    parser.add_argument('--filepath', type=str, default='https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv',
                        help='Path to the Counsel Chat CSV file.')
    parser.add_argument('--output_filepath', type=str, default='counsel_chat_formatted_paraphrased.json',
                        help='Path to save the formatted JSON data.')
    parser.add_argument('--log_level', type=str.upper, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.')
    parser.add_argument('--test_mode', action='store_true',
                        help='Enable test mode to process only the first question.')
    return parser.parse_args()

# Install sacremoses (recommended for MarianMT)
# !pip install sacremoses

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

# Install school_logging from GitHub
# !pip install git+https://github.com/vertok/school_logging.git

# Download required NLTK resources
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

# Set device for transformers models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained paraphrase model and tokenizer
paraphrase_model_name = "t5-base"  # Using t5-base for minimal setup
paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name).to(device)

# Load MarianMT models and tokenizers for back-translation
marian_model_names = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "nl": "Helsinki-NLP/opus-mt-en-nl",
    "ro": "Helsinki-NLP/opus-mt-en-ro",
    "zh": "Helsinki-NLP/opus-mt-zh-en"
}

marian_models = {
    lang: MarianMTModel.from_pretrained(model_name).to(device)
    for lang, model_name in marian_model_names.items()
}
marian_tokenizers = {
    lang: MarianTokenizer.from_pretrained(model_name)
    for lang, model_name in marian_model_names.items()
}

def clean_text(text, logger):
    """Cleans text with more advanced techniques."""
    logger.info(f"Cleaning text: '{text[:50]}...'")
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
    logger.info(f"Cleaned text: '{cleaned_text[:50]}...'")
    return cleaned_text.strip() # Remove any leading or trailing whitespace

def back_translate_with_marian(text, logger, lang="fr"):
    """Back-translates text using MarianMT models."""
    logger.info(f"Back-translating text: '{text[:50]}...' to language: {lang}")

    if lang not in marian_models:
        logger.error(f"No MarianMT model found for language: {lang}")
        return None

    try:
        # Get the tokenizer and model for this language
        tokenizer = marian_tokenizers[lang]
        model = marian_models[lang]

        # Translate from English to the target language
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated_ids = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

        # Translate from the target language back to English
        inputs_back = tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        back_translated_ids = model.generate(**inputs_back)
        back_translated_text = tokenizer.batch_decode(back_translated_ids, skip_special_tokens=True)[0]

        logger.info(f"Back-translated text: '{back_translated_text[:50]}...'")
        return back_translated_text
    except Exception as e:
        logger.error(f"Error during back-translation with MarianMT: {e}")
        return None

def paraphrase(text, logger, num_return_sequences=1, max_length=96):
    """Generates paraphrases using a pre-trained T5 model."""
    logger.info(f"Paraphrasing text: '{text[:50]}...'")
    input_ids = paraphrase_tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    generated_ids = paraphrase_model.generate(
        input_ids=input_ids,
        num_return_sequences=num_return_sequences,
        num_beams=2,
        max_length=max_length,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    paraphrases = [paraphrase_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    logger.info(f"Generated paraphrases: {[p[:50] for p in paraphrases]}")
    return paraphrases

def create_context(df, logger):
    """Creates a limited context by concatenating previous question turns."""
    logger.info("Creating context...")
    # Sort the dataframe to ensure proper order
    df = df.sort_values(by=['questionID', 'answerText'])
    # Create context by shifting previous questionText values within the same questionID group
    df['context'] = df.groupby('questionID')['questionText'].transform(lambda x: x.shift(1, fill_value=""))
    # Handle the first row of each group, setting context to empty string for these rows
    df['context'] = df.apply(lambda row: "" if pd.isna(row['context']) else f"Previous turns: {row['context']}", axis=1)
    # Ensure that the first question in each group has an empty context (even if it's not NaN)
    df.loc[df.groupby('questionID').head(1).index, 'context'] = ""
    logger.info("Context created.")
    return df

# def create_context(df, logger):
#     """Creates a limited context by concatenating previous question turns within each questionID group."""
#     logger.debug("Creating context...")
#     # Sort ONLY by questionID.  This is the crucial change.
#     df = df.sort_values(by=['questionID'])  
#     df['context'] = df.groupby('questionID')['questionText'].transform(lambda x: ' '.join(x.shift(fill_value='')))
#     df['context'] = df['context'].apply(lambda x: "" if x == "" else f"Previous turns:{x}")
#     df.loc[df.groupby('questionID').head(1).index, 'context'] = ""
#     logger.debug("Context created.")
#     return df

def is_similar(text1, text2, logger, threshold=0.65):
    """Checks if two texts are similar based on a normalized edit distance."""
    logger.info(f"Checking similarity between: '{text1[:50]}...' and '{text2[:50]}...'")
    distance = edit_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    similarity = 1 - (distance / max_len) if max_len > 0 else 0
    logger.info(f"Similarity score: {similarity}")
    return similarity >= threshold

def create_json_data(df, logger):
    """Converts DataFrame to JSON, adding variations and checking for similarity in real-time."""
    json_data = []

    for index, row in df.iterrows():
        logger.info(f"Processing row {index}...")

        # Ensure 'context' is always an empty string if it's missing or NaN
        if pd.isna(row['context']) or row['context'].strip() == '':
            if pd.isna(row['context']):
                row['context'] = ""  # Explicitly set NaN to empty string
            logger.error(f"Row {index} has missing or empty context. Skipping this row.")
            continue  # Skip rows with missing context

        variations = []
        potential_variations = []

        # Back-translation with MarianMT
        for lang in ["fr", "de", "es", "it", "nl", "ro", "zh"]:
            logger.info(f"Attempting back-translation with language: {lang}")
            variation = back_translate_with_marian(row['questionText'], logger, lang=lang)
            if variation:
                potential_variations.append(variation)

        # Paraphrasing with T5
        logger.info("Attempting paraphrasing...")
        try:
            paraphrases = paraphrase(row['questionText'], logger, num_return_sequences=5, max_length=96)
            potential_variations.extend(paraphrases)
        except Exception as e:
            logger.error(f"Error generating paraphrases for question: {row['questionText']}. Error: {e}")

        # Filter variations based on similarity
        logger.info("Filtering variations based on similarity...")
        for variation in potential_variations:
            if variation and not is_similar(row['questionText'], variation, logger):
                is_unique = True
                for existing_variation in variations:
                    if is_similar(variation, existing_variation, logger):
                        is_unique = False
                        break
                if is_unique:
                    variations.append(variation)

        # Create data point with error handling
        logger.info("Creating JSON data point...")
        try:
            data_point = {
                "context": row['context'],
                "question": row['questionText'],
                "variations": variations,
                "answer": row['answerText'],
                "metadata": {
                    "topic": row['topic'],
                    "questionTitle": row['questionTitle']
                }
            }
            json_data.append(data_point)
        except Exception as e:
            logger.error(f"Error creating JSON data point for row {index}: {e}")

        logger.info(f"Processed row {index}.")

    logger.info("JSON data creation complete.")
    return json_data

def prepare_counsel_chat_data(filepath, output_filepath, logger, test_mode):
    """Loads, cleans, formats, and saves the Counsel Chat dataset."""
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded: {df.head()}")

    # Test mode: Process only the first question
    if test_mode:
        first_question_id = df['questionID'].unique()[0]
        df = df[df['questionID'] == first_question_id]
        logger.info(f"Test mode enabled. Processing only the first question (ID: {first_question_id}).")
        logger.info(f"Dataset after filtering for first question: {df.head()}")

    # Create context
    logger.info("Creating context...")
    df = create_context(df, logger)
    logger.info(f"Context after creation: {df[['questionText', 'context']].head()}")

    # Clean the questionText and answerText data
    logger.info("Cleaning text data...")
    df['questionText'] = df['questionText'].apply(lambda text: clean_text(text, logger))
    df['answerText'] = df['answerText'].apply(lambda text: clean_text(text, logger))
    logger.info(f"Cleaned text data: {df[['questionText', 'answerText']].head()}")

    # Concatenate answers for duplicate questions
    if test_mode:  # Handle test mode differently
        json_data = create_json_data(df, logger) # No Grouping if we are in test mode
    else: # Normal mode - group by questionID
        # Concatenate answers for duplicate questions
        df = df.groupby('questionID').agg({
            'context': 'first',
            'questionTitle': 'first',
            'questionText': 'first',
            'answerText': lambda x: ' '.join(x.unique()),
            'topic': 'first'
        }).reset_index()
        logger.info(df)
        # Drop unnecessary columns
        df = df.drop(columns=['questionID'])

        # Create JSON data
        json_data = create_json_data(df, logger)

    logger.info(f"Dataset after concatenating answers: {df.head()}")

    # Drop unnecessary columns

    logger.info("Dropping unnecessary columns...")
    df = df.drop(columns=['questionID'])

    # Create JSON data
    logger.info("Creating JSON data...")
    json_data = create_json_data(df, logger)
    logger.info(f"Generated JSON data: {json_data}")

    # Save the formatted data to the file
    logger.info(f"Saving formatted data to: {output_filepath}")
    with open(output_filepath, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Data preparation complete. Formatted data saved to {output_filepath}")
    print("Remember: This dataset is licensed under CC BY-NC 4.0. Non-commercial use only!")


def main():
    """Parses arguments and runs the data preparation."""
    args = parse_args()

    # Create a logger instance
    logger = ColoredLogger('MyLogger', verbose=args.log_level)

    logger.info(f"Using device: {device}")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.debug("This is debug message.")

    # Run in test mode
    prepare_counsel_chat_data(args.filepath, args.output_filepath, logger, args.test_mode)

if __name__ == "__main__":
    main()
