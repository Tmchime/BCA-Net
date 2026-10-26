"""对齐图像、NIR和VOC，生成训练及评估数据集。"""

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

LABELS = {"H": 0, "M": 1, "L": 2}
B3_CLASSES = {"DX": "H", "LM": "H", "MMK": "H", "PLBZ": "L", "XCX": "M", "XHL": "L"}
B3_SITES = {
    "DX": "Nanjing",
    "PLBZ": "Nanjing",
    "LM": "Kaiyuan",
    "MMK": "Kaiyuan",
    "XCX": "Quanjiao",
    "XHL": "Quanjiao",
}


def sg_snv(raw, cfg):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != cfg.nir_dim or not np.isfinite(raw).all():
        raise ValueError("Invalid raw NIR")
    sm = savgol_filter(raw, cfg.sg_window, cfg.sg_order, axis=1)
    std = sm.std(axis=1, keepdims=True, ddof=1)
    if (std < 1e-8).any():
        raise ValueError("Constant spectrum cannot be standardized")
    return ((sm - sm.mean(1, keepdims=True)) / std).astype(np.float32)


def transform(train, cfg):
    ops = [transforms.Resize((cfg.image_size, cfg.image_size))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    return transforms.Compose(
        ops
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class LeafDataset(Dataset):
    def __init__(self, dm, ids, train, hide_labels):
        self.dm, self.ids, self.hide_labels = dm, list(ids), hide_labels
        self.tx = transform(train, dm.cfg)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, j):
        i = self.ids[j]
        r = self.dm.frame.iloc[i]
        with Image.open(self.dm.images[i]) as im:
            image = self.tx(im.convert("RGB"))
        return {
            "image": image,
            "nir": torch.from_numpy(self.dm.nir[i].copy()),
            "voc": torch.from_numpy(self.dm.voc[i].copy()),
            "label": -1 if self.hide_labels else int(self.dm.y[i]),
            "sample_id": r.sample_id,
            "cultivar": r.cultivar,
            "batch": r.batch,
            "site": r.site,
        }


class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        root = Path(cfg.data_dir).resolve()
        required = [root / n for n in ["manifest.csv", "nir.csv", "voc.csv", "wavelengths.csv"]]
        if not all(p.is_file() for p in required):
            raise FileNotFoundError(f"Required: {required}")
        self.synthetic = (root / "SYNTHETIC_ONLY").exists()
        if cfg.strict_paper_counts and self.synthetic:
            raise ValueError("Set strict_paper_counts=False for synthetic test inputs")
        frame = pd.read_csv(required[0], dtype=str, keep_default_na=False)
        cols = {
            "sample_id",
            "plant_id",
            "batch",
            "cultivar",
            "site",
            "nitrogen",
            "label",
            "image_path",
        }
        if not cols.issubset(frame):
            raise ValueError(f"Missing manifest columns: {cols - set(frame)}")
        if frame.sample_id.duplicated().any() or frame.plant_id.duplicated().any():
            raise ValueError("sample_id and plant_id must be globally unique")
        if frame[list(cols)].eq("").any().any():
            raise ValueError("Empty required metadata")
        if not set(frame.label).issubset(LABELS) or not set(frame.batch).issubset(
            {"B1", "B2", "B3"}
        ):
            raise ValueError("Use explicit H/M/L and B1/B2/B3")
        self.frame = frame.reset_index(drop=True)
        self.y = frame.label.map(LABELS).to_numpy(dtype=np.int64)
        self.images = [
            (root / str(p)).resolve() if not Path(p).is_absolute() else Path(p)
            for p in frame.image_path
        ]
        if any(not p.is_file() for p in self.images):
            raise FileNotFoundError("Missing image in manifest")
        if len(set(map(str, self.images))) != len(self.images):
            raise ValueError("Repeated image path")
        nir = pd.read_csv(required[1], dtype={"sample_id": str})
        if nir.sample_id.duplicated().any() or set(nir.sample_id) != set(frame.sample_id):
            raise ValueError("NIR IDs must be one-to-one with manifest")
        nircols = [c for c in nir.columns if c != "sample_id"]
        if len(nircols) != cfg.nir_dim:
            raise ValueError(f"Expected {cfg.nir_dim} NIR features")
        self.nir = sg_snv(
            nir.set_index("sample_id").loc[frame.sample_id, nircols].to_numpy(float), cfg
        )
        wl = pd.read_csv(required[3])
        if list(wl.columns) != ["wavelength_nm"]:
            raise ValueError("wavelengths.csv needs wavelength_nm")
        self.wavelengths = wl.wavelength_nm.to_numpy(float)
        if (
            len(wl) != cfg.nir_dim
            or not np.isfinite(self.wavelengths).all()
            or not (np.diff(self.wavelengths) > 0).all()
        ):
            raise ValueError("Invalid wavelength order")
        if self.wavelengths.min() < 1550 - 1e-6 or self.wavelengths.max() > 1950 + 1e-6:
            raise ValueError("Wavelengths outside 1550–1950 nm")
        self.batch = {b: np.where(frame.batch.to_numpy() == b)[0] for b in ["B1", "B2", "B3"]}
        b1 = self.batch["B1"]
        tv, te = train_test_split(
            b1, test_size=0.2, stratify=self.y[b1], random_state=cfg.split_seed
        )
        tr, va = train_test_split(
            tv, test_size=0.25, stratify=self.y[tv], random_state=cfg.split_seed
        )
        self.splits = {"train": np.asarray(tr), "val": np.asarray(va), "test": np.asarray(te)}
        self.target_splits = {b: self.half_split(b) for b in ["B2", "B3"]}
        voc = pd.read_csv(required[2], dtype={"sample_id": str})
        if list(voc.columns) != ["sample_id", *cfg.voc_columns]:
            raise ValueError("VOC requires ID + EXACT four named columns in configured order")
        if voc.sample_id.duplicated().any() or set(voc.sample_id) != set(frame.iloc[b1].sample_id):
            raise ValueError("VOC IDs must match B1 only")
        x = (
            voc.set_index("sample_id")
            .loc[frame.iloc[b1].sample_id, list(cfg.voc_columns)]
            .to_numpy(np.float32)
        )
        if not np.isfinite(x).all():
            raise ValueError("Non-finite VOC")
        self.voc = np.zeros((len(frame), 4), dtype=np.float32)
        self.voc[b1] = x
        self.voc_mean = self.voc[tr].mean(0)
        self.voc_std = self.voc[tr].std(0, ddof=1)
        self.voc_std = np.where(self.voc_std < 1e-8, 1, self.voc_std)
        self.voc[b1] = (self.voc[b1] - self.voc_mean) / self.voc_std
        self._check_counts()
        h = hashlib.sha256()
        for p in [*required, *self.images]:
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        self.data_hash = h.hexdigest()

    def half_split(self, batch):
        a, h = [], []
        for cv in sorted(self.frame.iloc[self.batch[batch]].cultivar.unique()):
            ix = self.batch[batch][self.frame.iloc[self.batch[batch]].cultivar.to_numpy() == cv]
            rng = np.random.default_rng(self.cfg.split_seed + sum(map(ord, cv)))
            ix = rng.permutation(ix)
            cut = len(ix) // 2
            a.extend(ix[:cut])
            h.extend(ix[cut:])
        a, h = np.asarray(sorted(a)), np.asarray(sorted(h))
        if len(set(a) & set(h)):
            raise RuntimeError("Adapt/holdout overlap")
        return a, h

    def _check_counts(self):
        for i in self.batch["B3"]:
            r = self.frame.iloc[i]
            if (
                r.cultivar not in B3_CLASSES
                or r.label != B3_CLASSES[r.cultivar]
                or r.site != B3_SITES[r.cultivar]
            ):
                raise ValueError("B3 cultivar/class/site disagrees with Table S1")
        if not self.cfg.strict_paper_counts:
            return
        expected = {
            "B1": ([47, 69, 23], {"XL": 22, "DT": 24, "BPP": 23, "FD": 23, "HPP": 24, "YTSK": 23}),
            "B2": ([47, 71, 24], {"XL": 24, "DT": 23, "BPP": 24, "FD": 24, "HPP": 23, "YTSK": 24}),
            "B3": ([300, 100, 200], {k: 100 for k in B3_CLASSES}),
        }
        for b, (yc, cvs) in expected.items():
            ix = self.batch[b]
            if (
                np.bincount(self.y[ix], minlength=3).tolist() != yc
                or self.frame.iloc[ix].cultivar.value_counts().to_dict() != cvs
            ):
                raise ValueError(f"{b} counts disagree with Table S1")
        if [len(self.splits[k]) for k in ["train", "val", "test"]] != [83, 28, 28]:
            raise ValueError("B1 split !=83/28/28")
        if [len(i) for i in self.target_splits["B2"]] != [70, 72] or [
            len(i) for i in self.target_splits["B3"]
        ] != [300, 300]:
            raise ValueError("Target split counts differ from paper")

    def loader(self, ids, train=False, hide_labels=False, seed=42):
        ds = LeafDataset(self, ids, train, hide_labels)
        gen = torch.Generator().manual_seed(seed)
        sampler = None
        if train and not hide_labels:
            labels = self.y[list(ids)]
            cnt = np.bincount(labels, minlength=3)
            weights = 1 / np.maximum(cnt, 1)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights[labels], dtype=torch.double),
                len(ids),
                replacement=True,
                generator=gen,
            )
        if train and len(ids) < self.cfg.batch_size:
            raise ValueError("Training subset smaller than batch_size")
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size if train else self.cfg.eval_batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            drop_last=train,
            num_workers=self.cfg.workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_seed,
            generator=gen,
        )

    def ids(self, rows):
        return self.frame.iloc[list(rows)].sample_id.tolist()

    def metadata(self):
        return {
            "data_sha256": self.data_hash,
            "synthetic": self.synthetic,
            "b1": {k: self.ids(v) for k, v in self.splits.items()},
            "targets": {
                b: {"adapt": self.ids(a), "holdout": self.ids(h)}
                for b, (a, h) in self.target_splits.items()
            },
            "voc_mean": self.voc_mean.tolist(),
            "voc_std": self.voc_std.tolist(),
            "wavelengths": self.wavelengths.tolist(),
        }
