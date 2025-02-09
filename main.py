"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

This script is the main entry point for preparing and augmenting the Counsel Chat dataset for use in training a psychological AI model.
It parses command-line arguments, initializes the logger, and calls the data preparation function.

Usage:
    python main.py --filepath <path_to_csv> --output_filepath <path_to_output_json> [--log_level <log_level>] [--test_mode] [--num_processes <num_processes>] [--model_name <model_name>]
Simple Example:
    python main.py --test_mode
    (This will process only one question in the dataset.)

Default CLI Variables:
1. --filepath: Path to the CSV file. Default is 'data/20200325_counsel_chat.csv'.
2. --output_filepath: Path to the output JSON file. Default is 'counsel_chat_formatted.json'.
3. --log_level: Logging level. Default is 'INFO'. Choices are ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].
4. --test_mode: If set, process only the first question. Default is False.
5. --num_processes: Number of processes to use. Default is 1.
6. --model_name: Name of the language model to use. Default is 'microsoft/phi-2'.
"""

import argparse
from data_augmentation import DataAugmentation
from utilities.text_utils import clean_text
from school_logging.log import ColoredLogger
import torch
import torch.multiprocessing as mp
import pandas as pd
import json
import traceback

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Prepare Counsel Chat dataset.')
    parser.add_argument('--filepath', type=str, default='data/20200325_counsel_chat.csv',
                        help='Path to the CSV file.')
    parser.add_argument('--output_filepath', type=str, default='counsel_chat_formatted.json',
                        help='Output JSON path.')
    parser.add_argument('--log_level', type=str.upper, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level. Default is INFO.')
    parser.add_argument('--test_mode', action='store_true',
                        help='Process only the first question.')
    parser.add_argument('--num_processes', type=int, default=1,  # Adjust depending on gpu memorywith 1: ~2200MB 
                        help='Number of processes. Default is 1.')
    parser.add_argument('--model_name', type=str, default="microsoft/phi-2",
                        help='Name of the language model.')
    return parser.parse_args()

def prepare_counsel_chat_data(filepath: str, output_filepath: str, logger: ColoredLogger, test_mode: bool, num_processes: int, model_name: str) -> None:
    """
    Main data preparation function.

    Args:
        filepath (str): The path to the CSV file.
        output_filepath (str): The path to the output JSON file.
        logger (ColoredLogger): Logger for debugging and error messages.
        test_mode (bool): Whether to process only the first question.
        num_processes (int): The number of processes to use.
        model_name (str): The name of the language model to use.
    """
    logger.info("Loading data from: '%s'", filepath)
    try:
        data_augmentation = DataAugmentation(logger)
        df = pd.read_csv(filepath, encoding='utf-8', converters={
            'questionText': clean_text,
            'answerText': clean_text
        })
        required_columns = ['questionID', 'questionTitle', 'questionText', 'topic', 'answerText']
        if not all(col in df.columns for col in required_columns):
            logger.error("CSV file must contain columns: %s", ", ".join(required_columns))
            return

        if df.empty:
            logger.error("The loaded DataFrame is empty.")
            return

        if test_mode:
            first_question_id = df['questionID'].unique()[8]
            df = df[df['questionID'] == first_question_id]
            logger.info("Test mode: Processing only one question.")

        df['questionText_tokenized'] = df['questionText'].apply(lambda text: data_augmentation.tokenize_and_lemmatize(text))
        df['answerText_tokenized'] = df['answerText'].apply(lambda text: data_augmentation.tokenize_and_lemmatize(text))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        json_data = data_augmentation.create_json_data(df, num_processes, model_name, device)

        if not json_data:
            logger.warning("No JSON data generated.")
            return

        logger.info("Saving to: '%s'", output_filepath)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)  # Ensure Unicode characters are not escaped
        logger.info("Data preparation complete. Saved to %s", output_filepath)

    except Exception as exc:
        logger.error("Error in prepare_counsel_chat_data: '%s'\n'%s'", exc, traceback.format_exc())

if __name__ == "__main__":
    args = parse_args()
    logger = ColoredLogger('MyLogger', verbose=args.log_level)
    try:
        mp.set_start_method('spawn')
        prepare_counsel_chat_data(args.filepath, args.output_filepath, logger, args.test_mode, args.num_processes, args.model_name)
    except Exception as exc:
        logger.error("An unexpected error occurred in main: '%s'\n'%s'", exc, traceback.print_exc())
