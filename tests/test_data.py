import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from common.data import DataModule
from tools.make_synthetic_inputs import make


class DataTests(unittest.TestCase):
    def test_id_join_split_and_hidden_targets(self):
        with tempfile.TemporaryDirectory() as td:
            config = make(Path(td) / "data")
            cfg = load_config(config)
            dm = DataModule(cfg)
            for b in ["B2", "B3"]:
                a, h = dm.target_splits[b]
                self.assertFalse(set(a) & set(h))
                self.assertEqual(set(a) | set(h), set(dm.batch[b]))
            hidden = next(iter(dm.loader(dm.target_splits["B2"][0], train=True, hide_labels=True)))
            self.assertTrue((hidden["label"] == -1).all())
            self.assertTrue(np.allclose(dm.voc[dm.splits["train"]].mean(0), 0, atol=1e-6))
            # 检查按sample_id连接后的输入与CSV行序无关。
            for name in ["nir.csv", "voc.csv"]:
                path = Path(cfg.data_dir) / name
                f = pd.read_csv(path)
                f.iloc[::-1].to_csv(path, index=False)
            other = DataModule(cfg)
            np.testing.assert_allclose(dm.nir, other.nir, atol=1e-6)
            np.testing.assert_allclose(dm.voc, other.voc, atol=1e-6)

    def test_duplicate_plant_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            config = make(Path(td) / "data")
            cfg = load_config(config)
            path = Path(cfg.data_dir) / "manifest.csv"
            f = pd.read_csv(path)
            f.loc[1, "plant_id"] = f.loc[0, "plant_id"]
            f.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "globally unique"):
                DataModule(cfg)

    def test_loco_requires_complete_configuration(self):
        from common.config import Config
        from table4_protocols.protocols import loco5

        with self.assertRaisesRegex(ValueError, "Incomplete LOCO configuration"):
            loco5("unused", None, Config(loco_epochs=None), 42, "unused")


if __name__ == "__main__":
    unittest.main()
