"""Pre-compute ESM-2 embeddings for all modlAMP training data.

Run this ONCE on a GPU machine to cache embeddings for fast pipeline execution.
Output: data/modlamp_embeddings.npz (~26MB)
"""

import sys
import io
import os
import time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "modlamp_embeddings.npz")


def main():
    print("=" * 60)
    print("AgriAMP — Pre-compute ESM-2 Embeddings")
    print("=" * 60)

    # 1. Load modlAMP data
    print("\n[1/3] Loading modlAMP dataset...")
    from modlamp.datasets import load_AMPvsUniProt
    data = load_AMPvsUniProt()
    seqs = data["sequences"]
    targets = np.array(data["target"])

    amp_seqs = [seqs[i] for i in range(len(seqs)) if targets[i] == 1]
    non_amp_seqs = [seqs[i] for i in range(len(seqs)) if targets[i] == 0]
    print(f"  AMPs: {len(amp_seqs)}, non-AMPs: {len(non_amp_seqs)}")

    # Clean sequences
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

    def clean(s):
        return "".join(c for c in s.upper() if c in standard_aa)

    all_seqs_raw = amp_seqs + non_amp_seqs
    all_labels = [1] * len(amp_seqs) + [0] * len(non_amp_seqs)

    cleaned = []
    labels_clean = []
    for s, lbl in zip(all_seqs_raw, all_labels):
        cs = clean(s)
        if len(cs) >= 5:
            cleaned.append(cs)
            labels_clean.append(lbl)

    # Deduplicate (keep first occurrence)
    seen = set()
    unique_seqs = []
    unique_labels = []
    for s, lbl in zip(cleaned, labels_clean):
        if s not in seen:
            seen.add(s)
            unique_seqs.append(s)
            unique_labels.append(lbl)

    print(f"  After cleaning/dedup: {len(unique_seqs)} sequences")
    print(f"    AMPs: {sum(unique_labels)}, non-AMPs: {len(unique_labels) - sum(unique_labels)}")

    # 2. Load ESM-2 and embed
    print("\n[2/3] Loading ESM-2 model...")
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  Model: {model_name}")

    print(f"\n  Embedding {len(unique_seqs)} sequences in batches of 8...")
    start = time.time()
    all_embeddings = []
    batch_size = 8

    for i in range(0, len(unique_seqs), batch_size):
        batch = unique_seqs[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        for j in range(len(batch)):
            seq_len = (inputs["attention_mask"][j] == 1).sum().item() - 2
            if seq_len > 0:
                emb = outputs.last_hidden_state[j, 1:1 + seq_len, :].mean(dim=0)
            else:
                emb = outputs.last_hidden_state[j, 0, :]
            all_embeddings.append(emb.cpu().numpy())

        del inputs, outputs
        if device == "cuda":
            torch.cuda.empty_cache()

        done = min(i + batch_size, len(unique_seqs))
        if done % 200 == 0 or done == len(unique_seqs):
            elapsed = time.time() - start
            print(f"    {done}/{len(unique_seqs)} ({elapsed:.1f}s)")

    embeddings_array = np.array(all_embeddings)
    elapsed = time.time() - start
    print(f"  Done: {embeddings_array.shape} in {elapsed:.1f}s")

    # 3. Save
    print(f"\n[3/3] Saving to {OUTPUT_PATH}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        sequences=np.array(unique_seqs, dtype=object),
        labels=np.array(unique_labels, dtype=np.int32),
        embeddings=embeddings_array,
    )
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  Saved: {size_mb:.1f} MB")
    print(f"\n{'=' * 60}")
    print(f"Pre-computation complete.")
    print(f"  Sequences: {len(unique_seqs)}")
    print(f"  Embedding dim: {embeddings_array.shape[1]}")
    print(f"  File: {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
