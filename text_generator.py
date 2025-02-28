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
from typing import Dict
import logging
import torch

class TextGenerator:
    def __init__(self, model_name: str, device: str, prompt_templates: Dict[str, str] = None):
        """
        Initializes the TextGenerator with the specified model and device.

        Args:
            model_name (str): The name of the pre-trained language model.
            device (str): The device to run the model on ('cuda' or 'cpu').
            logger (Any): Logger for debugging and error messages.
        """

        self.model_name = model_name
        self.device = device
        self.prompt_templates = prompt_templates or {}
        
        # Setup proper logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Adjusted quantization config for 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16  # Use fp16 for stability
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
            truncation_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with matching dtypes
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            bnb_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    def generate_text_from_prompt(self, prompt, max_length=256, num_beams=2, 
                                  no_repeat_ngram_size=2, temperature: float = 0.4, top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.5):
        """Generate text from prompt with memory-efficient parameters."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Adjusted generation parameters for stability
            # temperature = max(torch.finfo(torch.float16).eps, temperature)  # Prevent NaNs
            # top_p = min(0.99, max(0.8, top_p))  # Keep within safe range
            # top_k = max(1, top_k)

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,  
                num_beams=num_beams,  
                temperature=temperature,  
                top_k=top_k,  
                top_p=top_p,  
                no_repeat_ngram_size=no_repeat_ngram_size,  
                early_stopping=True,
                repetition_penalty=repetition_penalty, 
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  
            )

            # Decode and return text
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).replace(prompt, "").strip()

            return generated_text

        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear memory only on error

            return ""

    def generate_category_info(self, text: str, topic: str = 'emotional_support') -> Dict[str, str]:
        """Generate responses for each template category."""
        if not self.prompt_templates:
            raise ValueError("No prompt templates provided")
            
        category_info = {}
        self.logger.info(f"Starting category generation for text: {text[:50]}...")

        for category, template in self.prompt_templates.items():
            try:
                prompt = template.format(text=text, topic=topic)
                self.logger.info(f"\n{'='*50}\nCategory: {category}\nPrompt: {prompt}\n{'='*50}")

                response = self.generate_text_from_prompt(
                    prompt,
                    temperature=0.6,  # Lowered for stability
                    top_p=0.85,  # Avoid high variance in sampling
                    num_beams=2,  # Reduce from 4 to minimize compute
                    max_length=384  # Reduce length to avoid memory issues
                )

                category_info[category] = response
                self.logger.info(f"Generated response for {category}: {response[:100]}...")

            except KeyError as ke:
                self.logger.error(f"Template format error for {category}: {ke}")
                category_info[category] = ""
            except Exception as e:
                self.logger.error(f"Error generating {category}: {e}")
                category_info[category] = ""

        return category_info
