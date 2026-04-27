"""PyG dataset: ligand graphs + cached ESM embeddings + pK labels."""

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from plb.data.cache import DEFAULT_ESM_MODEL, esm_cache_dir
from plb.data.ligand import ligand_graph_from_sdf

log = logging.getLogger(__name__)

LIGAND_CACHE_RELPATH = "cache/ligand_graphs.pt"


def ligand_cache_path(data_root: Path | str) -> Path:
    return Path(data_root) / LIGAND_CACHE_RELPATH


def _ligand_dict_from_sdf(sdf_path: Path) -> dict[str, torch.Tensor | str]:
    try:
        graph = ligand_graph_from_sdf(sdf_path, sanitize=True)
    except Exception:
        graph = ligand_graph_from_sdf(sdf_path, sanitize=False)
    return {
        "x": torch.from_numpy(np.asarray(graph.node_feats, dtype=np.float32)),
        "edge_index": torch.from_numpy(np.asarray(graph.edge_index, dtype=np.int64)),
        "edge_attr": torch.from_numpy(np.asarray(graph.edge_feats, dtype=np.float32)),
        "smiles": graph.smiles,
    }


def _refined_sdf_path(refined_dir: Path, pdb_id: str) -> Path:
    return refined_dir / pdb_id / f"{pdb_id}_ligand.sdf"


def load_ligand_graph_cache(data_root: Path | str) -> dict[str, dict]:
    path = ligand_cache_path(data_root)
    if not path.is_file():
        return {}
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"corrupt ligand cache at {path}: expected dict, got {type(blob)}")
    return blob


def build_data_object(
    pdb_id: str,
    ligand: Mapping[str, torch.Tensor | str],
    esm_pocket: np.ndarray,
    esm_whole: np.ndarray,
    pK: float,
) -> Data:
    return Data(
        x=ligand["x"],
        edge_index=ligand["edge_index"],
        edge_attr=ligand["edge_attr"],
        esm_pocket=torch.from_numpy(np.asarray(esm_pocket, dtype=np.float32)).unsqueeze(0),
        esm_whole=torch.from_numpy(np.asarray(esm_whole, dtype=np.float32)).unsqueeze(0),
        y=torch.tensor([float(pK)], dtype=torch.float32),
        pdb_id=pdb_id,
    )


class PDBbindGraphDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        pdb_ids: Iterable[str],
        labels: Mapping[str, float],
        data_root: Path | str,
        refined_dir: Path | str,
        esm_model: str = DEFAULT_ESM_MODEL,
        ligand_cache: Mapping[str, dict] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.refined_dir = Path(refined_dir)
        self.esm_model = esm_model

        cache = (
            dict(ligand_cache)
            if ligand_cache is not None
            else load_ligand_graph_cache(self.data_root)
        )
        esm_dir = esm_cache_dir(self.data_root, esm_model)

        self.samples: list[Data] = []
        self.skipped: list[tuple[str, str]] = []  # (pdb_id, reason)

        for pdb_id in pdb_ids:
            pdb_id = pdb_id.lower()
            if pdb_id not in labels:
                self.skipped.append((pdb_id, "missing label"))
                continue

            esm_path = esm_dir / f"{pdb_id}.npz"
            if not esm_path.is_file():
                self.skipped.append((pdb_id, "missing esm cache"))
                continue

            ligand = cache.get(pdb_id)
            if ligand is None:
                sdf = _refined_sdf_path(self.refined_dir, pdb_id)
                if not sdf.is_file():
                    self.skipped.append((pdb_id, "missing ligand sdf"))
                    continue
                try:
                    ligand = _ligand_dict_from_sdf(sdf)
                except Exception as exc:  # pragma: no cover - defensive
                    self.skipped.append((pdb_id, f"ligand parse failed: {exc}"))
                    continue

            with np.load(esm_path) as data:
                esm_pocket = np.asarray(data["pocket"], dtype=np.float32)
                esm_whole = np.asarray(data["whole"], dtype=np.float32)

            self.samples.append(
                build_data_object(
                    pdb_id=pdb_id,
                    ligand=ligand,
                    esm_pocket=esm_pocket,
                    esm_whole=esm_whole,
                    pK=float(labels[pdb_id]),
                )
            )

        if self.skipped:
            log.warning(
                "PDBbindGraphDataset skipped %d/%d entries (first 5: %s)",
                len(self.skipped),
                len(self.skipped) + len(self.samples),
                self.skipped[:5],
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        return self.samples[idx]


def make_dataloader(
    dataset: PDBbindGraphDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
