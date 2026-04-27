"""ESM-2 embedding extraction and pocket pooling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MODEL = "esm2_t12_35M_UR50D"
ESM_EMBED_DIMS: dict[str, int] = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
}
ESM_LAST_LAYER: dict[str, int] = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}


def embed_dim_for(model_name: str) -> int:
    if model_name not in ESM_EMBED_DIMS:
        raise KeyError(f"unknown ESM-2 model: {model_name!r}")
    return ESM_EMBED_DIMS[model_name]


@dataclass
class ESMEmbedder:
    """Lazily-loaded ESM-2 model that produces per-residue embeddings."""

    model_name: str = DEFAULT_MODEL
    device: str = "cpu"
    layer: int | None = None  # None -> last layer of the chosen model
    max_window: int = 1022  # ESM-2 hard limit minus BOS/EOS

    _model: object | None = field(default=None, init=False, repr=False)
    _alphabet: object | None = field(default=None, init=False, repr=False)
    _batch_converter: object | None = field(default=None, init=False, repr=False)

    @property
    def embed_dim(self) -> int:
        return embed_dim_for(self.model_name)

    @property
    def effective_layer(self) -> int:
        return self.layer if self.layer is not None else ESM_LAST_LAYER[self.model_name]

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import esm

        load_fn = getattr(esm.pretrained, self.model_name)
        model, alphabet = load_fn()
        model.eval()
        model.to(self.device)
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()

    def embed_chain(self, sequence: str) -> np.ndarray:
        if not sequence:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        self._lazy_load()

        if len(sequence) <= self.max_window:
            return self._embed_window(sequence)

        stride = max(1, self.max_window // 2)
        out = np.zeros((len(sequence), self.embed_dim), dtype=np.float32)
        counts = np.zeros(len(sequence), dtype=np.int32)

        start = 0
        while start < len(sequence):
            end = min(start + self.max_window, len(sequence))
            emb = self._embed_window(sequence[start:end])
            out[start:end] += emb
            counts[start:end] += 1
            if end == len(sequence):
                break
            start += stride

        out /= counts[:, None].clip(min=1)
        return out

    def _embed_window(self, sequence: str) -> np.ndarray:
        import torch

        assert self._batch_converter is not None and self._model is not None
        _, _, tokens = self._batch_converter([("seq", sequence)])
        tokens = tokens.to(self.device)
        with torch.inference_mode():
            results = self._model(
                tokens,
                repr_layers=[self.effective_layer],
                return_contacts=False,
            )
        rep = results["representations"][self.effective_layer][0, 1 : 1 + len(sequence)]
        return rep.detach().cpu().float().numpy()


def pool_pocket_embedding(
    chain_embeddings: dict[str, np.ndarray],
    pocket_positions: dict[str, list[int]],
    embed_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool over pocket residues and whole protein. Returns (pocket, whole)."""
    pocket_chunks: list[np.ndarray] = []
    whole_chunks: list[np.ndarray] = []
    for cid, emb in chain_embeddings.items():
        if emb.size:
            whole_chunks.append(emb)
        positions = pocket_positions.get(cid, [])
        if positions:
            pocket_chunks.append(emb[positions])

    if whole_chunks:
        whole_pool = np.concatenate(whole_chunks, axis=0).mean(axis=0)
    else:
        whole_pool = np.zeros(embed_dim, dtype=np.float32)

    if pocket_chunks:
        pocket_pool = np.concatenate(pocket_chunks, axis=0).mean(axis=0)
    else:
        pocket_pool = whole_pool

    return pocket_pool.astype(np.float32), whole_pool.astype(np.float32)
