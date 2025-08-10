from transformers import pipeline
import torch

model_id = "../models/new_kldiv/model_kldiv_continued" #
# model_id = "models/teacher_kl_div/model_kldiv_continued_605500"
# model_id = "meta-llama/Llama-3.2-1B"

print("Before pipeline")

pipe = pipeline(
    "text-generation", 
    max_new_tokens=300,
    top_p=0.95,
    temperature=0.9,
    num_return_sequences=3,
    do_sample=True,
    model=model_id, 
    tokenizer="meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16, 
    device=0
)

print("Generating...\n")

output = pipe("The difference between a fastball and a curveball is")
for out in output:
    print(out['generated_text'])
    print()
