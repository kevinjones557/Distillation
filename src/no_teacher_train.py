from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import sys, os
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.set_float32_matmul_precision('high')

model_id = "meta-llama/Llama-3.2-1B"
dataset_name = "HuggingFaceFW/fineweb"

device = torch.device('cuda')

# get GPU info
gpu_name = torch.cuda.get_device_name(device)
total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
print(f"Using GPU: {gpu_name}", flush=True)
print(f"Total memory: {total_memory:.2f} GB", flush=True)

# tokenizer and models, can use same tokenizer for both
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model_teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
model_student = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# remove half of the hidden layers, but we want to keep the last layer
compare_against = 1 if len(model_student.model.layers) % 2 == 0 else 0
pruned_layers = [layer for idx, layer in enumerate(model_student.model.layers) if idx % 2 == compare_against]
model_student.model.layers = torch.nn.ModuleList(pruned_layers)
model_student.config.num_hidden_layers = len(pruned_layers)

model_student.gradient_checkpointing_enable()
model_student.config.use_flash_attention = True

# resize embedding layer because of padding token we added
# model_teacher.resize_token_embeddings(len(tokenizer))
# model_teacher.eval()

model_student.resize_token_embeddings(len(tokenizer))
model_student.train()

print(f"Student Model Params = {sum(p.numel() for p in model_student.parameters()) / 1000 ** 2} million", flush=True)
# print(f"Teacher Model Params = {sum(p.numel() for p in model_teacher.parameters()) / 1000 ** 2} million", flush=True)

# stream fineweb dataset
batch_size = 16
dataset = load_dataset(dataset_name, name="CC-MAIN-2024-10", split="train", streaming=True)
dataloader = DataLoader(dataset, batch_size=batch_size)

cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'))
temperature = 1.0
alpha = 0.5
optimizer = torch.optim.AdamW(model_student.parameters(), lr=1e-4)

allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
print(f"Allocated Memory before training: {allocated_memory:.2f} GB", flush=True)

iteration = 0
total_tokens = 0

# training loop
for batch in dataloader: #tqdm(dataloader, desc="Processing batches", file=sys.stdout):
    texts = batch['text']
    # tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    labels = input_ids.clone()
    labels = labels[:, 1:] # shape is batch size, seq length

    start_time = time.time()
    torch.cuda.synchronize()

    # with torch.no_grad():
    #     teacher_outputs = model_teacher(input_ids=input_ids)
    #     soft_labels = teacher_outputs.logits
    #     soft_probs = F.softmax(soft_labels, dim=-1)

    student_outputs = model_student(input_ids=input_ids)
    student_logits = student_outputs.logits # shape is batch size, seq len (1024), vocab_size
    student_logits = student_logits[:, :-1, :] # remove last prediction because we don't have a label

    # Soft label loss (KLDivLoss expects log-probs for input and probs for target)
    # student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # soft_loss = F.kl_div(student_log_probs, soft_probs, reduction='batchmean') * (temperature ** 2)

    hard_loss = cross_entropy_loss(student_logits.reshape(-1, len(tokenizer)), labels.reshape(-1))

    # loss = alpha * soft_loss + (1 - alpha) * hard_loss
    loss = 0.0

    optimizer.zero_grad()
    hard_loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    end_time = time.time()

    batch_time = end_time - start_time
    iter_tokens = inputs['attention_mask'].sum().item()
    total_tokens += iter_tokens

    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Cross Entropy Loss: {hard_loss:.2f}, \
        Total loss: {loss:.4f}, \
        Throughput: {iter_tokens/batch_time:.2f} tokens/s, \
        Allocated Memory: {allocated_memory:.2f}, \
        Tokens Processed (total): {(total_tokens / 1000 ** 2):.4} M", flush=True)

    iteration += 1

    # if iteration % 500 == 0:
    #     model_student.save_pretrained(f"models/no_teacher/model_lower{iteration}")
    #     print(f"Saved model, skip count = {batch_size * iteration}")
