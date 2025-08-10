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

student_model_id = "../models/new_kldiv/model_kldiv_hard_trained"
optimizer_path = "../optimizers/new_optim_kldiv_hard_trained.pt"
dataset_name = "HuggingFaceFW/fineweb"

device = torch.device('cuda')

# get GPU info
gpu_name = torch.cuda.get_device_name(device)
total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
print(f"Using GPU: {gpu_name}", flush=True)
print(f"Total memory: {total_memory:.2f} GB", flush=True)

flag_path = "../still_training_kldiv.flag"
if os.path.exists(flag_path):
    os.remove(flag_path)

skip_count = 4264000
token_count = 0
iteration_begin = 0

file_name = "../logs/train_new_hard.out"
if os.path.exists(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        last_line = lines[-1].split()
        iteration_begin = int(last_line[1][:-1])
        skip_count = int(last_line[-1])
        token_count = int(float(lines[-3].split()[-2]) * 1000 ** 2)

log_file = open(file_name, "a")

# tokenizer and models, can use same tokenizer for both
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model_student = AutoModelForCausalLM.from_pretrained(student_model_id, torch_dtype=torch.bfloat16).to(device)

model_student.gradient_checkpointing_enable()
model_student.config.use_flash_attention = True

# model_student.resize_token_embeddings(len(tokenizer))
model_student.train()

print(f"Student Model Params = {sum(p.numel() for p in model_student.parameters()) / 1000 ** 2} million", flush=True)

# stream fineweb dataset
batch_size = 8
dataset = load_dataset(dataset_name, name="CC-MAIN-2024-10", split="train", streaming=True)
val_size = 100

val_dataset = dataset.take(val_size)
train_dataset = dataset.skip(val_size + skip_count)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'))
temperature = 1
optimizer = torch.optim.AdamW(model_student.parameters(), lr=1e-4)
if (os.path.exists(optimizer_path)):
    optimizer.load_state_dict(torch.load(optimizer_path))

allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
print(f"Allocated Memory before training: {allocated_memory:.2f} GB", flush=True)

iteration = iteration_begin
total_tokens = token_count

# training loop
for batch in train_dataloader: #tqdm(dataloader, desc="Processing batches", file=sys.stdout):
    texts = batch['text']
    # tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    labels = input_ids.clone()
    labels = labels[:, 1:] # shape is batch size, seq length

    start_time = time.time()
    torch.cuda.synchronize()

    student_outputs = model_student(input_ids=input_ids)
    student_logits = student_outputs.logits # shape is batch size, seq len (1024), vocab_size

    # hard label loss
    student_logits = student_logits[:, :-1, :] # remove last prediction because we don't have a hard label
    hard_loss = cross_entropy_loss(student_logits.reshape(-1, len(tokenizer)), labels.reshape(-1))

    optimizer.zero_grad()
    hard_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
    optimizer.step()

    torch.cuda.synchronize()
    end_time = time.time()

    batch_time = end_time - start_time
    iter_tokens = inputs['attention_mask'].sum().item()
    total_tokens += iter_tokens

    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
    log_file.write(f"Hard loss: {hard_loss:.2f}, \
        Throughput: {iter_tokens/batch_time:.2f} tokens/s, \
        Allocated Memory: {allocated_memory:.2f}, \
        Tokens Processed (total): {(total_tokens / 1000 ** 2):.6} M\n")
    log_file.flush()

    iteration += 1
    skip_count += batch_size

    if iteration % 500 == 0:
        model_student.eval()

        val_hard_loss = 0.0
        val_soft_loss = 0.0
        val_token_count = 0

        with torch.no_grad():
            for i, val_batch in enumerate(val_dataloader):

                val_texts = val_batch['text']
                val_inputs = tokenizer(val_texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
                val_input_ids = val_inputs['input_ids']
                val_labels = val_input_ids[:, 1:].clone()

                student_val_outputs = model_student(input_ids=val_input_ids)
                student_val_logits = student_val_outputs.logits

                student_val_logits = student_val_logits[:, :-1, :]
                hard_loss_val = cross_entropy_loss(student_val_logits.reshape(-1, len(tokenizer)), val_labels.reshape(-1))

                val_hard_loss += hard_loss_val.item()
                val_token_count += 1

        avg_hard_val = val_hard_loss / val_token_count

        log_file.write(f"[Validation Loss] Hard: {avg_hard_val:.4f}\n")
        log_file.flush()

        model_student.train()
        model_student.save_pretrained(student_model_id)
        torch.save(optimizer.state_dict(), optimizer_path)
        log_file.write(f"Iteration {iteration}, Saved model, skip count = {skip_count}\n")
        log_file.flush()

        if (iteration - iteration_begin) == 14000:
            log_file.close()
            with open("../still_training_kldiv_hard.flag", "w") as f:
                f.write("continue")
            break
