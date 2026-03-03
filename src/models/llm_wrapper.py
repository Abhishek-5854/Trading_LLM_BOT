"""LLM wrapper for inference with optional LoRA adapters.

Supports:
- Base model inference (plain transformers)
- LoRA adapter loading (fine-tuned weights)
- 4-bit quantization for memory efficiency on CPU/small GPU
"""
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class LLMWrapper:
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        use_4bit: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize LLM wrapper.
        
        Args:
            model_name_or_path: base model (e.g., 'mistralai/Mistral-7B-Instruct-v0.1')
            adapter_path: path to LoRA adapters (optional)
            use_4bit: whether to use 4-bit quantization
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name_or_path
        self.adapter_path = adapter_path
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device if device == "cpu" else "auto",
            )
        
        # Load adapters if provided
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapters from {adapter_path}")
        
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=False,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def predict_action(self, prompt: str) -> str:
        """
        Predict trading action (BUY, SELL, HOLD) from prompt.
        
        Extracts the first word from model output and validates it's a valid action.
        """
        response = self.generate(prompt, max_new_tokens=10)
        
        # Extract first word
        first_word = response.split()[0].upper() if response else "HOLD"
        
        # Validate against allowed actions
        valid_actions = ["BUY", "SELL", "HOLD"]
        if first_word in valid_actions:
            return first_word
        else:
            # Default to HOLD if unrecognized
            return "HOLD"
