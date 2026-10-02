#!/usr/bin/env python3
"""DATA Phi-Ultimate Fine-Tuner & Continual Learning Engine (EWC)
Hardware: AMD Ryzen 9 3900X (24 Threads)
Arquitectura: Microsoft Phi-1.5 + EWC (Elastic Weight Consolidation)
"""
import os
import sys

# Optimización multihilo para Ryzen 9 3900X
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16)

MODEL_ID = "microsoft/phi-1_5"
DATASET_PATH = "/run/media/administrator/DATA-X-PHI/training_corpus.jsonl"
OUTPUT_WEIGHTS = "/run/media/administrator/DATA-X-PHI/data_phi_finetuned_weights.pt"

print("==================================================================")
print("  MOTOR DE ENTRENAMIENTO SINÁPTICO: DATA PHI-ULTIMATE (EWC)")
print("  Hardware: AMD Ryzen 9 3900X (16/24 Threads en CPU)")
print("  Tópicos: Matemáticas, Arte, Elegancia, Música/DSP, Teorías Fuera de la Caja")
print("==================================================================")

# 1. Cargar Tokenizer y Modelo Base
print("[1/5] Cargando Tokenizer y Modelo Base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, clean_up_tokenization_spaces=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# 2. Congelar capas base y habilitar gradientes solo en capas superiores (Adapter/Head)
print("[2/5] Configurando capas activas (Top-Layers + Head)...")
for param in model.parameters():
    param.requires_grad = False

# Habilitar entrenamiento en las últimas 2 capas del transformer y en el lm_head
trainable_params = []
for name, param in model.named_parameters():
    if "transformer.h.22" in name or "transformer.h.23" in name or "lm_head" in name:
        param.requires_grad = True
        trainable_params.append(param)

print(f"      Parámetros totales entrenables: {sum(p.numel() for p in trainable_params):,}")

# 3. Snapshot de pesos para EWC (Elastic Weight Consolidation)
print("[3/5] Calculando matriz de anclaje sináptico EWC...")
reference_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
# Fisher diagonal aproximada
fisher_diagonal = {name: torch.ones_like(param) * 0.05 for name, param in model.named_parameters() if param.requires_grad}
ewc_lambda = 400.0 # Peso de regularización elástica

# 4. Preparar Dataset
print("[4/5] Cargando corpus de entrenamiento...")
class SodaxDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = f"Instruct: {item['instruction']}\nOutput: {item['output']}{tokenizer.eos_token}"
                enc = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                # Enmascarar padding en el cálculo de pérdida
                labels[input_ids == tokenizer.pad_token_id] = -100
                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = SodaxDataset(DATASET_PATH, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print(f"      Ejemplares cargados: {len(dataset)} items de alta densidad conceptual.")

# 5. Bucle de Entrenamiento (Fine-Tuning con EWC)
print("[5/5] Iniciando bucle de optimización con AdamW...")
optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

EPOCHS = 3
model.train()
t0_train = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    steps = 0
    print(f"\n--- Época {epoch}/{EPOCHS} ---")
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift de logits y labels para modelado de lenguaje causal
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Penalización EWC
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in reference_params:
                ewc_loss += (fisher_diagonal[name] * (param - reference_params[name]) ** 2).sum()

        total_loss = ce_loss + (ewc_lambda / 2.0) * ewc_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        epoch_loss += ce_loss.item()
        steps += 1
        print(f"  Paso {batch_idx + 1}/{len(dataloader)} | CE Loss: {ce_loss.item():.4f} | EWC Loss: {ewc_loss.item():.6f}")

    avg_loss = epoch_loss / steps
    print(f"✔ Fin Época {epoch} - Pérdida Promedio: {avg_loss:.4f}")

elapsed = time.time() - t0_train
print(f"\n⚡ Entrenamiento completado con éxito en {elapsed:.2f} segundos.")

# Guardar pesos adaptados
print(f"💾 Guardando pesos sinápticos consolidados en {OUTPUT_WEIGHTS}...")
save_dict = {
    name: param.data.cpu()
    for name, param in model.named_parameters()
    if param.requires_grad
}
torch.save(save_dict, OUTPUT_WEIGHTS)
print("✅ Pesos guardados con éxito. DATA ha consolidado este conocimiento en su cerebro.")
