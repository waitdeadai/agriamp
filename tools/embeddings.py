"""Tool 2: Generate protein embeddings using ESM-2 protein language model."""

import numpy as np
from tools import BaseTool, ToolResult

# Model preference order (largest that fits in VRAM first)
MODEL_CONFIGS = [
    ("facebook/esm2_t33_650M_UR50D", 1280, "650M params"),
    ("facebook/esm2_t30_150M_UR50D", 640, "150M params"),
    ("facebook/esm2_t12_35M_UR50D", 480, "35M params"),
    ("facebook/esm2_t6_8M_UR50D", 320, "8M params"),
]


class EmbeddingsTool(BaseTool):
    name = "ESM-2 Embeddings"
    description = "Genera representaciones vectoriales con ESM-2 protein language model"
    icon = "🧬"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_name = None
        self._embed_dim = None
        self._model_desc = None

    def _load_model(self):
        """Load the best ESM-2 model that fits in available resources."""
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_name, embed_dim, desc in MODEL_CONFIGS:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name).to(self._device)
                self._model.eval()
                self._model_name = model_name
                self._embed_dim = embed_dim
                self._model_desc = desc
                return True
            except Exception:
                continue

        return False

    def _get_embedding(self, sequence: str) -> np.ndarray:
        """Get mean-pooled embedding for a single sequence."""
        import torch

        inputs = self._tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling over sequence (exclude BOS/EOS tokens)
        embedding = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()
        return embedding

    def _get_embeddings_batch(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        """Get embeddings for multiple sequences in batches."""
        import torch

        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Mean pooling for each sequence in batch
            for j in range(len(batch)):
                seq_len = (inputs["attention_mask"][j] == 1).sum().item() - 2  # exclude BOS/EOS
                if seq_len > 0:
                    emb = outputs.last_hidden_state[j, 1 : 1 + seq_len, :].mean(dim=0)
                else:
                    emb = outputs.last_hidden_state[j, 0, :]
                all_embeddings.append(emb.cpu().numpy())

            # Free GPU memory
            del inputs, outputs
            if self._device == "cuda":
                torch.cuda.empty_cache()

        return np.array(all_embeddings)

    def _execute(self, sequences: list[str]) -> ToolResult:
        if not sequences:
            return ToolResult(status="error", message="No hay secuencias para procesar.")

        # Load model if not loaded
        if self._model is None:
            loaded = self._load_model()
            if not loaded:
                return ToolResult(
                    status="error",
                    message="No se pudo cargar ningún modelo ESM-2.",
                )

        # Clean sequences (remove non-standard amino acids)
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        clean_sequences = []
        for seq in sequences:
            clean = "".join(c for c in seq.upper() if c in standard_aa)
            if len(clean) >= 5:  # minimum peptide length
                clean_sequences.append(clean)

        if not clean_sequences:
            return ToolResult(status="error", message="Ninguna secuencia válida para procesar.")

        # Generate embeddings
        embeddings = self._get_embeddings_batch(clean_sequences)

        msg = (
            f"Generé representaciones vectoriales de {len(clean_sequences)} péptidos "
            f"usando ESM-2 ({self._model_desc}), un modelo de lenguaje proteico "
            f"entrenado en 65 millones de secuencias (UniRef50). "
            f"Cada péptido es un vector de {self._embed_dim} dimensiones "
            f"que captura propiedades evolutivas y estructurales. "
            f"Dispositivo: {self._device.upper()}."
        )

        return ToolResult(
            status="success",
            message=msg,
            data={
                "embeddings": embeddings,
                "sequences": clean_sequences,
                "model": self._model_name,
                "model_desc": self._model_desc,
                "embed_dim": self._embed_dim,
                "device": self._device,
                "n_sequences": len(clean_sequences),
            },
        )
