from pydantic import ConfigDict, Field, model_validator
from langchain.llms.base import LLM
from typing import Dict, Any, Optional

from src.model.GPTModel import GPTLanguageModel
from src.tokenizer.regex_tokenizer import RegexTokenizer
from src.inference_repo.inference import order_response
import torch
import logging
import os

# Use logging after the debugging finished

base_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(base_dir, "..", "tokenizer", "tokenizer_model.model")
model_file = os.path.normpath(model_file)

checkpoint_file = os.path.join(base_dir, "..", "inference_repo", "fine_tuned_checkpoint_9.pth")
checkpoint_file = os.path.normpath(checkpoint_file)

class GPTModelLLM(LLM):
    """GPT Model which integrates with langchain"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid', validate_assignment=True
    )
 
    model: GPTLanguageModel = Field(default=None, description="Custom GPTLanguage Model")
    tokenizer: RegexTokenizer = Field(default=None, description="Custom Tokenizer")
    max_length: int = Field(default=1024)
    temperature: float = Field(default=0.8, ge=0.2, le=2.0, description="Temperature hyperparameter")
    max_new_tokens: int = Field(default=256, description="Maximum new tokens to be generated")
    top_p: Optional[float] = Field(default=None, description="Top p value")
    top_k: Optional[int] = Field(default=None, description="Top k tokens to pick")

    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    @model_validator(mode="after")
    def load_dependencies(self):
        """Auto load tokenizer and model after initialization"""
        
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()

        if self.model is None:
            self.model = self.load_model()
        
        return self

    def load_model(self):
        """Load the model"""
        vocab_size = len(self.tokenizer.vocab)
        n_embedding = 384
        padding_token = 3077
        n_head = 8
        n_layer = 6
        block_size = 1024
        dropout = 0.2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model = GPTLanguageModel(n_embedding=n_embedding, n_head=n_head,
                                    n_layer=n_layer, block_size=block_size,
                                    dropout=dropout, device=device,
                                    padding_token=padding_token, 
                                    vocab_size=vocab_size
                            )
            
            checkpoint = torch.load(checkpoint_file,
                                    map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        except Exception as e:
            logging.error("Error while loading the model:", e)
            raise
        
        return model

    def load_tokenizer(self):
        """Load the tokenizer"""
        try:
            tokenizer = RegexTokenizer()
            tokenizer.load(model_file=model_file)
        
        except Exception as e:
            logging.error("Error while loading the tokenizer", e)
            raise

        return tokenizer

    def _format_prompt(self, prompts: str):
        """The format of the model which is trained on"""

        format = f"<|startoftext|><|User|>{prompts}<|Assistant|>"
        return format

    def _call(self, prompts: str, stop: Optional[list[str]]=None
             ):
        """Main inference point of the model"""

        try:
            formatted_prompt = self._format_prompt(prompts=prompts)
            inputs = self.tokenizer.encode(formatted_prompt, allowed_special='all')
            inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(self.device)
                    
            # else:
            #     print("tokenizer has no `encode` attribute")
            
            input_length = inputs.shape[1] if len(inputs.shape) > 1 else len(inputs)
            max_new_tokens = self.max_new_tokens

            if max_new_tokens <= 0:
                return "Please short the prompt for better answer"
            

            with torch.inference_mode():
                output = self.model.generate(input_tokens=inputs,
                                            max_new_tokens=self.max_new_tokens,
                                            block_size=self.max_length,
                                            temperature=self.temperature, top_k=self.top_k,
                                            top_p=self.top_p)

            output = output.squeeze().tolist()
            response_tokens = output[input_length:]
            response = self.tokenizer.decode(response_tokens)
            response = order_response(response)
            
            response = response.replace("<|endoftext|>", "").strip()

            if stop:
                for s in stop:
                    if s in response:
                        response = response.split(s)[0]

            return response

        except Exception as e:
            logging.error("Error while inference", e)
            raise e
    
    @property
    def _llm_type(self) -> str:
        """Return the identifier for LLM"""
        return "Custom GPT model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters"""
        return {
            "model_path": checkpoint_file,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p
        }


    

