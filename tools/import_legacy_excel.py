"""根据样本行映射，将Excel转换为CSV输入文件。

mapping.csv包含manifest字段及nir_file、nir_sheet、nir_row、voc_file、voc_sheet、voc_row。
row为pandas数据表的0基行号。schema.json列出各批次499个NIR列名及四个VOC列名映射。"""

import argparse
import json
from pathlib import Path

import pandas as pd

from common.config import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--wavelengths", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    mapping = Path(a.mapping).resolve()
    m = pd.read_csv(mapping, dtype=str, keep_default_na=False)
    schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
    cache = {}
    spectra = []
    vocs = []

    def table(file, sheet):
        path = Path(file)
        if not path.is_absolute():
            path = mapping.parent / path
        key = (str(path.resolve()), sheet)
        if key not in cache:
            df = pd.read_excel(path, sheet_name=sheet)
            df.columns = df.columns.map(str)
            cache[key] = df
        return cache[key]

    for r in m.itertuples(index=False):
        cols = schema["nir_columns"][r.batch]
        if len(cols) != 499:
            raise ValueError("nir_columns must explicitly name 499 features per batch")
        values = table(r.nir_file, r.nir_sheet).iloc[int(r.nir_row)][list(map(str, cols))]
        spectra.append([r.sample_id, *pd.to_numeric(values, errors="raise").to_list()])
        if r.batch == "B1":
            row = table(r.voc_file, r.voc_sheet).iloc[int(r.voc_row)]
            names = [str(schema["voc_columns"][name]) for name in Config().voc_columns]
            vocs.append([r.sample_id, *pd.to_numeric(row[names], errors="raise").to_list()])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cols = ["sample_id", "plant_id", "batch", "cultivar", "site", "nitrogen", "label", "image_path"]
    # 将图像路径转换为绝对路径。
    m["image_path"] = m.image_path.map(
        lambda s: str((mapping.parent / s).resolve()) if not Path(s).is_absolute() else s
    )
    m[cols].to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(spectra, columns=["sample_id", *[f"band_{i:03d}" for i in range(499)]]).to_csv(
        out / "nir.csv", index=False
    )
    pd.DataFrame(vocs, columns=["sample_id", *Config().voc_columns]).to_csv(
        out / "voc.csv", index=False
    )
    (out / "wavelengths.csv").write_bytes(Path(a.wavelengths).read_bytes())
    print('CSV inputs saved. Run: python run.py --config "**" --check-data')


if __name__ == "__main__":
    main()
