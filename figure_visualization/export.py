"""从student checkpoint导出融合特征和光谱注意力。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import load_config
from common.data import DataModule
from common.engine import load_model, tensors
from common.runtime import device_for, json_write, sha256


@torch.inference_mode()
def export(checkpoint, dm, cfg, out, b1_scope="test", sample_id=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(cfg)
    model, obj = load_model(checkpoint, dm, cfg, device)
    model.eval()
    if obj["model_task"] in {"teacher", "teacher_swapped"}:
        raise ValueError("Fig6 export uses student features")
    ids = np.concatenate([dm.splits[b1_scope], dm.batch["B2"]]).astype(int)
    features = []
    rows = []
    for batch in dm.loader(ids):
        im, nir, _, y = tensors(batch, device)
        _, _, an, f = model(im, nir)
        features.append(f.cpu().numpy())
        for j, sid in enumerate(batch["sample_id"]):
            rows.append(
                {"sample_id": sid, "label": ["H", "M", "L"][int(y[j])], "domain": batch["batch"][j]}
            )
            if sample_id == sid:
                if an is None:
                    raise ValueError("Selected model has no spectral attention")
                np.savez(
                    out / "spectral_attention.npz",
                    wavelengths=dm.wavelengths,
                    nir_values=nir[j].cpu().numpy(),
                    attn_weights=an[j].flatten().cpu().numpy(),
                    label=np.array(["H", "M", "L"][int(y[j])]),
                    sample_id=np.array(sid),
                )
    meta = pd.DataFrame(rows)
    if sample_id and sample_id not in set(meta.sample_id):
        raise ValueError("Sample not in selected B1/B2 scope")
    np.save(out / "features.npy", np.concatenate(features))
    meta.to_csv(out / "samples.csv", index=False)
    json_write(
        {
            "checkpoint_sha256": sha256(checkpoint),
            "model_task": obj["model_task"],
            "seed": obj["meta"]["seed"],
            "b1_scope": b1_scope,
            "feature_layer": "post-fusion before classifier",
            "sample_id": sample_id,
            "attention_postprocessing": "raw attention weights",
            "synthetic": dm.synthetic,
            "n": len(rows),
        },
        out / "provenance.json",
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample-id")
    p.add_argument("--b1-scope", choices=["train", "val", "test"], default="test")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    export(a.checkpoint, DataModule(cfg), cfg, a.out, a.b1_scope, a.sample_id)


if __name__ == "__main__":
    main()
