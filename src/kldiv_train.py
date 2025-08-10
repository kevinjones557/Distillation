from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os
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
model_teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
model_student = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# remove half of the hidden layers, but we want to keep the last layer
compare_against = 1 if len(model_student.model.layers) % 2 == 0 else 0
pruned_layers = [layer for idx, layer in enumerate(model_student.model.layers) if idx % 2 == compare_against]
model_student.model.layers = torch.nn.ModuleList(pruned_layers)
model_student.config.num_hidden_layers = len(pruned_layers)

model_student.gradient_checkpointing_enable()
model_student.config.use_flash_attention = True

# resize embedding layer because of padding token we added
model_teacher.resize_token_embeddings(len(tokenizer))
model_teacher.eval()

model_student.resize_token_embeddings(len(tokenizer))
model_student.train()

print(f"Student Model Params = {sum(p.numel() for p in model_student.parameters()) / 1000 ** 2} million", flush=True)
print(f"Teacher Model Params = {sum(p.numel() for p in model_teacher.parameters()) / 1000 ** 2} million", flush=True)

# stream fineweb dataset
batch_size = 8
dataset = load_dataset(dataset_name, name="CC-MAIN-2024-10", split="train", streaming=True)
val_size = 100

val_dataset = dataset.take(val_size)          # takes first 100 samples
train_dataset = dataset.skip(val_size)        # skips first 100 samples

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'))
temperature = 1
alpha = 0.8
optimizer = torch.optim.AdamW(model_student.parameters(), lr=1e-4)

allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
print(f"Allocated Memory before training: {allocated_memory:.2f} GB", flush=True)

iteration = 0
total_tokens = 0

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

    with torch.no_grad():
        teacher_outputs = model_teacher(input_ids=input_ids)
        soft_labels = teacher_outputs.logits / temperature
        soft_probs = F.softmax(soft_labels, dim=-1)

    student_outputs = model_student(input_ids=input_ids)
    student_logits = student_outputs.logits # shape is batch size, seq len (1024), vocab_size

    # soft label loss
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence per token
    kl_div = F.kl_div(
        student_log_probs,
        soft_probs,
        reduction='none'
    )

    # sum KL over vocab dim, then mask
    kl_div_per_token = kl_div.sum(dim=-1)   # shape: (batch, seq_len)
    attention_mask = inputs['attention_mask'].float()
    masked_kl = kl_div_per_token * attention_mask

    # final soft loss
    soft_loss = masked_kl.sum() / attention_mask.sum()

    # hard label loss
    student_logits = student_logits[:, :-1, :] # remove last prediction because we don't have a hard label
    hard_loss = cross_entropy_loss(student_logits.reshape(-1, len(tokenizer)), labels.reshape(-1))

    loss = alpha * soft_loss + (1 - alpha) * hard_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
    optimizer.step()

    torch.cuda.synchronize()
    end_time = time.time()

    batch_time = end_time - start_time
    iter_tokens = inputs['attention_mask'].sum().item()
    total_tokens += iter_tokens

    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Hard loss: {hard_loss:.2f}, \
        Soft loss: {soft_loss:.4f}, \
        Throughput: {iter_tokens/batch_time:.2f} tokens/s, \
        Allocated Memory: {allocated_memory:.2f}, \
        Tokens Processed (total): {(total_tokens / 1000 ** 2):.4} M", flush=True)

    iteration += 1

    if iteration % 500 == 0:
        model_student.eval()

        val_batches = 10
        val_hard_loss = 0.0
        val_soft_loss = 0.0
        val_token_count = 0

        with torch.no_grad():
            for i, val_batch in enumerate(val_dataloader):
                if i >= val_batches:
                    break

                val_texts = val_batch['text']
                val_inputs = tokenizer(val_texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
                val_input_ids = val_inputs['input_ids']
                val_labels = val_input_ids[:, 1:].clone()

                teacher_val_outputs = model_teacher(input_ids=val_input_ids)
                teacher_soft_logits = teacher_val_outputs.logits / temperature
                teacher_soft_probs = F.softmax(teacher_soft_logits, dim=-1)

                student_val_outputs = model_student(input_ids=val_input_ids)
                student_val_logits = student_val_outputs.logits

                student_val_log_probs = F.log_softmax(student_val_logits / temperature, dim=-1)
                val_kl_div = F.kl_div(student_val_log_probs, teacher_soft_probs, reduction='none')
                val_kl_per_token = val_kl_div.sum(dim=-1)
                val_attention_mask = val_inputs['attention_mask'].float()
                val_masked_kl = val_kl_per_token * val_attention_mask
                soft_loss_val = val_masked_kl.sum() / val_attention_mask.sum()

                student_val_logits = student_val_logits[:, :-1, :]
                hard_loss_val = cross_entropy_loss(student_val_logits.reshape(-1, len(tokenizer)), val_labels.reshape(-1))

                val_soft_loss += soft_loss_val.item()
                val_hard_loss += hard_loss_val.item()
                val_token_count += 1

        avg_soft_val = val_soft_loss / val_token_count
        avg_hard_val = val_hard_loss / val_token_count

        print(f"[Validation Loss] Hard: {avg_hard_val:.4f}, Soft: {avg_soft_val:.4f}", flush=True)

        model_student.train()

        model_student.save_pretrained(f"../models/new_kldiv/model_kldiv")
        torch.save(optimizer.state_dict(), f"../optimizers/new_optim_kldiv.pt")
        print(f"Iteration {iteration}, Saved model, skip count = {batch_size * iteration}")
        break