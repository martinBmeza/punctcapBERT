import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class TokenChunkDataset(Dataset):
    """
    .npz:
      - token_ids: (T,)
      - tokens: (T,)  optional
      - word_ids: (T,)  optional
      - labels_punt_ini: (T,)  # binary 0/1
      - labels_punt_fin: (T,)  # int 0..3
      - labels_cap: (T,)       # int 0..3
      - token_start, token_end, orig_instance_id, original_text (optional)
    """

    def __init__(self, metadata_path: str, filter_split: Optional[str] = None):
        """
        metadata_path: path to metadata_{split}.parquet
        filter_split: if metadata contains a 'split' column, can filter rows by this value
        """
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # load metadata: parquet or csv
        ext = self.metadata_path.suffix.lower()
        if ext in [".parquet"]:
            self.df = pd.read_parquet(self.metadata_path)
        elif ext in [".csv", ".gz"]:
            # support compressed .csv.gz as well
            self.df = pd.read_csv(self.metadata_path)
        else:
            # try to read as parquet/csv by autodetection
            try:
                self.df = pd.read_parquet(self.metadata_path)
            except Exception:
                self.df = pd.read_csv(self.metadata_path)

        if filter_split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == filter_split].reset_index(drop=True)

        if "path" not in self.df.columns:
            raise ValueError("metadata file must contain 'path' column pointing to chunk .npz files")
        self.paths = self.df["path"].tolist()

    def __len__(self):
        return len(self.paths)

    def _load_npz(self, path):
        try:
            with np.load(path, allow_pickle=True) as dd:
                data = dict(dd)
        except Exception as e:
            raise RuntimeError(f"Error loading npz at {path}: {e}")
        return data

    def __getitem__(self, idx):
        path = self.paths[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunk file not found: {path}")
        data = self._load_npz(path)

        token_ids = data.get("token_ids")
        labels_p_ini = data.get("labels_punt_ini")
        labels_p_fin = data.get("labels_punt_fin")
        labels_cap = data.get("labels_cap")
        word_ids = data.get("word_ids") if "word_ids" in data else None
        original_text = None
        if "original_text" in data:
            raw = data["original_text"]
            if isinstance(raw, np.ndarray):
                try:
                    original_text = str(raw.tolist())
                except Exception:
                    original_text = str(raw)
            else:
                original_text = str(raw)

        if token_ids is None:
            raise ValueError(f"token_ids missing in {path}")

        T = int(len(token_ids))
        token_ids = np.array(token_ids, dtype=np.int64)
        labels_p_ini = np.array(labels_p_ini, dtype=np.int64) if labels_p_ini is not None else np.zeros(T, dtype=np.int64)
        labels_p_fin = np.array(labels_p_fin, dtype=np.int64) if labels_p_fin is not None else np.zeros(T, dtype=np.int64)
        labels_cap = np.array(labels_cap, dtype=np.int64) if labels_cap is not None else np.zeros(T, dtype=np.int64)
        if word_ids is not None:
            word_ids = np.array(word_ids, dtype=np.int64)
        else:
            # si no esta word_ids crear un mapeo por defecto donde cada token es su propia palabra 
            word_ids = np.arange(T, dtype=np.int64)

        item = {
            "token_ids": token_ids,
            "labels_p_ini": labels_p_ini,
            "labels_p_fin": labels_p_fin,
            "labels_cap": labels_cap,
            "word_ids": word_ids,
            "n_tokens": T,
            "path": path,
            "original_text": original_text
        }
        if "tokens" in data:
            item["tokens"] = np.array(data["tokens"], dtype=object)
        return item


def collate_fn_pad(batch: List[Dict[str, Any]], pad_value: int = 0, ignore_index: int = -100):
    """
      - token_ids: LongTensor (B, T)
      - attention_mask: BoolTensor (B, T)
      - labels_p_ini: FloatTensor (B, T)  (0/1 floats)  # for BCEWithLogitsLoss
      - labels_p_fin: LongTensor (B, T)  (0..C-1) with pad positions = ignore_index (for CrossEntropy)
      - labels_cap: LongTensor (B, T)  (0..C-1) with pad = ignore_index
      - word_ids: LongTensor (B, T) with pad positions = -1
      - lengths: LongTensor (B,)
      - paths: list of paths
      - original_texts: list of original_text (or None) length B
    """
    batch_size = len(batch)
    lengths = [int(x["n_tokens"]) for x in batch]
    max_len = max(lengths)

    token_ids_batch = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    labels_p_ini = torch.zeros((batch_size, max_len), dtype=torch.float32)
    labels_p_fin = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
    labels_cap = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)

    # pad con -1 para padding positions/noword
    word_ids_batch = torch.full((batch_size, max_len), -1, dtype=torch.long)

    paths = []
    original_texts: List[Optional[str]] = []
    for i, item in enumerate(batch):
        L = item["n_tokens"]
        token_ids_batch[i, :L] = torch.from_numpy(item["token_ids"]).to(torch.long)
        attention_mask[i, :L] = 1

        labels_p_ini[i, :L] = torch.from_numpy(item["labels_p_ini"]).float()
        labels_p_fin[i, :L] = torch.from_numpy(item["labels_p_fin"]).to(torch.long)
        labels_cap[i, :L] = torch.from_numpy(item["labels_cap"]).to(torch.long)

        word_ids_batch[i, :L] = torch.from_numpy(item["word_ids"]).to(torch.long)

        paths.append(item.get("path"))
        original_texts.append(item.get("original_text"))

    batch_out = {
        "token_ids": token_ids_batch,
        "attention_mask": attention_mask,
        "labels_p_ini": labels_p_ini,
        "labels_p_fin": labels_p_fin,
        "labels_cap": labels_cap,
        "word_ids": word_ids_batch,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "paths": paths,
        "original_texts": original_texts
    }
    return batch_out