from tabnanny import check
from src.model.GPTModel import GPTLanguageModel
from src.tokenizer.regex_tokenizer import RegexTokenizer
import torch
import re
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(base_dir, "..", "tokenizer", "tokenizer_model.model")
model_file = os.path.normpath(model_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = RegexTokenizer()
tokenizer.load(model_file=model_file)
vocab_size = len(tokenizer.vocab)
n_embedding = 384
padding_token = 3077
n_head = 8
n_layer = 6
block_size = 1024
dropout = 0.2
model = GPTLanguageModel(n_embedding=n_embedding, n_head=n_head,
                        n_layer=n_layer, block_size=block_size,
                        dropout=dropout, device=device,
                        padding_token=padding_token, 
                        vocab_size=vocab_size
                    )


TEST_ENV = os.getenv("TEST_ENV", "0") == "1"

def load_checkpoint(mode, base_dir):
    checkpoint_file = os.path.join(base_dir, "fine_tune_checkpoint_9.pth")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

if not TEST_ENV:
    checkpoint = load_checkpoint(model, base_dir)
else:
    checkpoint = None

model.load_state_dict(checkpoint["model_state_dict"])

prompt = "What is symptoms of Cancer?"
formatted_prompt = f"<|startoftext|><|User|>{prompt}<|Assistant|>"
input_ids = tokenizer.encode(formatted_prompt, allowed_special='all')
input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)


def order_response(text: str) -> str:
    text = re.sub(r"^[A-D]\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", " ", text)
    last_period = text.rfind(".")
    if last_period != -1:
        return text[:last_period+1]
        
    return text.strip()

model.eval()
with torch.inference_mode():
    output = model.generate(input_ids, 200,
                           block_size, 0.9,
                           top_k=40, top_p=0.9)
output = output.squeeze().tolist()
response_tokens = output[input_ids.shape[1]:]
response = tokenizer.decode(response_tokens)
response = order_response(response)


if __name__ == "__main__":
    print("User:", prompt, "\n")
    print("Assistant:", response.replace("<|endoftext|>", ""))
    # print(tokenizer.encode(formatted_prompt, allowed_special='all'))
