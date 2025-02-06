import unittest
import pandas as pd
from io import StringIO
import os
import json
from prepare_data_gemini import clean_text, back_translate_with_marian, paraphrase, \
    create_context, is_similar, create_json_data, prepare_counsel_chat_data
from school_logging.log import ColoredLogger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel
import torch

# MarianModelLoader class (added here to be part of the same file)
class MarianModelLoader:
    _models = None

    @staticmethod
    def get_models():
        if MarianModelLoader._models is None:
            MarianModelLoader._models = MarianModelLoader.load_models()
        return MarianModelLoader._models

    @staticmethod
    def load_models():
        logger = ColoredLogger('TestLogger', verbose='DEBUG')
        languages = ['fr', 'de', 'es', 'it', 'nl', 'ro', 'zh']
        models = {}
        for lang in languages:
            try:
                models[lang] = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-en-{lang}')
                logger.info(f"MarianMT model for {lang} loaded successfully.")
            except Exception as e:
                logger.info(f"Error loading MarianMT model for {lang}: {e}")
        return models

class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        # Create a logger instance for testing
        self.logger = ColoredLogger('TestLogger', verbose='DEBUG')

        # Sample data for testing
        self.sample_data = """questionID,questionTitle,questionText,answerText,topic
    1,Test Question 1,This is the first test question.,This is the first test answer.,Test
    1,Test Question 1,This is the first test question.,This is the second test answer.,Test
    2,Test Question 2,This is the second test question.,This is another test answer.,Test
    3,Test Question 3,This is the third test question.,This is the third test answer.,Test"""
        self.df = pd.read_csv(StringIO(self.sample_data))

        # Set device for transformers models
        self.device = torch.device("cpu")  # Avoid GPU issues during tests

        # Load pre-trained paraphrase model and tokenizer
        self.paraphrase_model_name = "t5-base"
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained(self.paraphrase_model_name)
        try:
            self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(self.paraphrase_model_name).to(self.device)
            self.logger.debug("T5 paraphrase model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading T5 paraphrase model: {e}")

        # Load MarianMT models using the MarianModelLoader
        self.marian_models = MarianModelLoader.get_models()

        # Ensure context column is present in df
        self.df = create_context(self.df.copy(), self.logger)

    def test_clean_text(self):
        text = "<h1>This is a <b>test</b> string!</h1> With some URLs: https://www.example.com."
        cleaned_text = clean_text(text, self.logger)
        self.assertEqual(cleaned_text, "test string url")

    def test_back_translate_with_marian(self):
        text = "This is a test sentence."
        for lang in ["fr", "de", "es"]:
            translated_text = back_translate_with_marian(text, self.logger, lang=lang)
            self.assertIsNotNone(translated_text)
            self.assertNotEqual(translated_text, text)

    def test_paraphrase(self):
        text = "This is a slightly more complex test sentence to check if paraphrasing works correctly."
        paraphrases = paraphrase(text, self.logger)
        self.assertIsNotNone(paraphrases)
        self.assertGreater(len(paraphrases), 0)
        self.assertNotEqual(paraphrases[0], text)

    def test_create_context(self):  # needs deeper analysis...
        self.logger.info(self.df)
        df_with_context = create_context(self.df.copy(), self.logger)

        self.assertEqual(df_with_context['context'].iloc[0], "")
        self.assertEqual(df_with_context['context'].iloc[1], "Previous turns: This is the first test question.")
        self.assertEqual(df_with_context['context'].iloc[2], "") # QuestionID 2 starts fresh
        self.assertEqual(df_with_context['context'].iloc[3], "") # QuestionID 3 starts fresh

    def test_is_similar(self):
        text1 = "This is a test sentence."
        text2 = "This is a test string."
        text3 = "Completely different text."
        self.assertFalse(is_similar(text1, text2, self.logger, threshold=0.9))
        self.assertFalse(is_similar(text1, text3, self.logger, threshold=0.9))

    def test_create_json_data(self):
        json_data = create_json_data(self.df.copy(), self.logger)
        self.assertIsNotNone(json_data)
        self.assertGreater(len(json_data), 0)
        self.assertIn('context', json_data[0])
        self.assertIn('question', json_data[0])
        self.assertIn('variations', json_data[0])
        self.assertIn('answer', json_data[0])
        self.assertIn('metadata', json_data[0])

    def test_prepare_counsel_chat_data(self):
        # Create a temporary output file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
            output_filepath = tmpfile.name

        # Call the main function with the test data
        # Create a StringIO object to simulate the CSV file
        csv_file = StringIO(self.sample_data)

        # Call the main function with the StringIO object
        prepare_counsel_chat_data(filepath=csv_file, output_filepath=output_filepath, logger=self.logger, test_mode=False)

        # Load the generated JSON data
        with open(output_filepath, "r") as f:
            generated_data = json.load(f)

        # Assert that the JSON data is not empty
        self.assertIsNotNone(generated_data)
        self.assertGreater(len(generated_data), 0)

        # Clean up the temporary file
        os.remove(output_filepath)

if __name__ == '__main__':
    unittest.main()
