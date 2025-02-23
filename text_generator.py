import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import re
import logging
import traceback
from utilities.text_utils import clean_text  # Import clean_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model_name: str, device: str, use_bfloat16: bool = False):
        self.device = device
        self.model_name = model_name
        self.use_bfloat16 = use_bfloat16
        self.tokenizer = None
        self.model = None
        self.toxic_tokenizer = None
        self.toxic_model = None
        # Don't load models in __init__ anymore

    def _load_language_model(self):
        """Loads the language model and tokenizer."""
        logger.info(f"Loading language model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {"trust_remote_code": True}
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"  # Use automatic device mapping
                if self.use_bfloat16 and torch.cuda.is_bf16_supported():
                    model_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    model_kwargs["torch_dtype"] = torch.float16  # Use float16 for Phi-1.5
            elif self.device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32 #Prevent error with phi-2

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.model.eval()  # Put the model in evaluation mode

        except Exception as e:
            logger.error(f"Error loading language model {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

    def _unload_language_model(self):
        """Unloads the language model and tokenizer from memory."""
        logger.info(f"Unloading language model: {self.model_name}")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()  # Clear GPU cache

    def _load_toxicity_model(self):
        """Loads the toxicity model and tokenizer."""
        logger.info("Loading toxicity model: facebook/roberta-hate-speech-dynabench-r4-target")
        try:
            self.toxic_tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
            self.toxic_model = AutoModelForSequenceClassification.from_pretrained(
                "facebook/roberta-hate-speech-dynabench-r4-target",
                 torch_dtype=torch.float32, # Use float32 for CPU
            )
            # ALWAYS keep the toxicity model on CPU
            self.toxic_model.to("cpu")  # Explicitly on CPU
            self.toxic_model.eval()
        except Exception as e:
            logger.error(f"Error loading toxicity model: {e}\n{traceback.format_exc()}")
            raise

    def _unload_toxicity_model(self):
        """Unloads the toxicity model and tokenizer from memory."""
        logger.info("Unloading toxicity model")
        if self.toxic_model is not None:
            del self.toxic_model
            self.toxic_model = None
        if self.toxic_tokenizer is not None:
            del self.toxic_tokenizer
            self.toxic_tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7,
                      top_p: float = 0.95, no_repeat_ngram_size: int = 2) -> str:
        """Generates text, loading the language model if needed."""
        self._load_language_model()  # Load on demand
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    use_cache=True
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):]  # Remove the prompt

            # Clean up the generated text using the existing clean_text function
            generated_text = clean_text(generated_text)

            # Check if the generated text is empty
            if not generated_text:
                logger.warning("The language model generated an empty response.")
                return "I'm sorry, I couldn't generate a meaningful response to your question. Please try again."

            return generated_text
        except Exception as e:
            logger.error(f"Error in text generation: {e}\n{traceback.format_exc()}")
            return "I'm sorry, I encountered an error while generating a response."
        finally:
            self._unload_language_model() # Unload after use

    def is_toxic(self, text: str) -> bool:
        """Checks if text is toxic, loading the toxicity model if needed."""
        self._load_toxicity_model()  # Load on demand
        try:
            inputs = self.toxic_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)  # No .to(device)
            with torch.no_grad():
                outputs = self.toxic_model(**inputs)  # No .to(device)
            toxic_score = outputs.logits.softmax(dim=-1)[0, -1].item()
            return toxic_score > 0.7
        except Exception as e:
            logger.error(f"Error during toxicity check: {e}\n{traceback.format_exc()}")
            return False
        finally:
            self._unload_toxicity_model()  # Unload after use


    def get_embedding(self, text: str) -> torch.Tensor:
        """Generates embeddings, loading the language model if needed."""
        self._load_language_model() # Load on demand
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)  # CRITICAL: Add this!

            # Corrected: Access hidden states from the 'outputs' object correctly.
            hidden_states = outputs.hidden_states[-1]  # Get the last hidden state
            embeddings = hidden_states.mean(dim=1)  # Mean pooling
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}\n{traceback.format_exc()}")
            return None
        finally:
            self._unload_language_model()  # Unload after use
