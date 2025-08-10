import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the model to tested
model = "../models/new_kldiv/model_kldiv_hard_trained"
# Laod the tokenizer used
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

mmlu = evaluate.load("mmlu")
dataset_mmlu = load_dataset("cais/mmlu", split="test")
result_mmlu = mmlu.compute(model=model, tokenizer=tokenizer, data=dataset_mmlu)

print(result_mmlu)