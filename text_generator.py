"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

This class is designed to generate text using a pre-trained language model. It utilizes the `transformers` library
to load the model and tokenizer, and it supports 4-bit quantization to save resources. The default model used is
'microsoft/phi-2', which is a very lightweight model.

Functionality:
1. Initialize the model and tokenizer with 4-bit quantization.
2. Generate text based on a given prompt with various tuning parameters.
3. Clean and decode the generated text to handle Unicode characters and unwanted symbols.

Usage:
    from text_generator import TextGenerator
    logger = ColoredLogger('TextGeneratorLogger', verbose='INFO')
    generator = TextGenerator(model_name='microsoft/phi-2', device='cuda', logger=logger)
    prompt = "Once upon a time"
    generated_text = generator.generate_text(prompt)
    print(generated_text)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import traceback
from typing import Any

from utilities.text_utils import clean_text

class TextGenerator:
    def __init__(self, model_name: str, device: str, logger: Any) -> None:
        """
        Initializes the TextGenerator with the specified model and device.

        Args:
            model_name (str): The name of the pre-trained language model.
            device (str): The device to run the model on ('cuda' or 'cpu').
            logger (Any): Logger for debugging and error messages.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.logger = logger
        self.logger.name = "TextGenerator"

    def generate_text(self, prompt: str, max_new_tokens: int = 60, temperature: float = 0.4, top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.5, do_sample: bool = True) -> str:
        """
        Generates text with tuned parameters.

        Args:
            prompt (str): The input prompt to generate text from.
            max_new_tokens (int): The maximum number of new tokens to generate. Default is 60.
            temperature (float): The temperature for sampling. Default is 0.4.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k filtering. Default is 40.
            top_p (float): The cumulative probability for top-p filtering. Default is 0.9.
            repetition_penalty (float): The penalty for repetition. Default is 1.5.
            do_sample (bool): Whether to use sampling. Default is True.

        Returns:
            str: The generated text.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            attention_mask = inputs.attention_mask

            if torch.isnan(inputs.input_ids).any():
                self.logger.error("NaN values found in input_ids!")
                return "Error: Invalid input."

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text.replace(prompt, "").strip()
            generated_text = clean_text(generated_text)  # Clean and decode the generated text
            return generated_text

        except RuntimeError as e:
            if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
                self.logger.error("RuntimeError: %s\n%s", e, traceback.format_exc())
                return "Error: Numerical instability."
            else:
                raise
        except Exception as e:
            self.logger.error("Unexpected error: %s\n%s", e, traceback.format_exc())
            return "Error: Generation failed."
