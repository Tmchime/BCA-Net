"""模型、损失、指标和checkpoint测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common.config import Config, load_config
from common.identifiers import resolve_experiment, result_ids
from common.losses import coral, teacher_targets
from common.metrics import classification
from common.registry import TASKS, build_model
from common.runtime import json_write, prepare_output, read_checkpoint, save_checkpoint


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cfg(self):
        return Config(
            backbone="convnext_atto",
            embed_dim=32,
            pretrained=False,
            image_size=32,
            workers=0,
            strict_paper_counts=False,
        )

    def test_coral_formula_and_gradient(self):
        torch.manual_seed(42)
        x = torch.randn(6, 8, requires_grad=True)
        y = torch.randn(7, 8)
        actual = coral(x, y)
        expected = ((torch.cov(x.T) - torch.cov(y.T)) ** 2).mean() / (4 * 8**2)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_unmixed_teacher_targets(self):
        log = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
        ai = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        p, a, n = teacher_targets(log, ai, ai, 4.0)
        torch.testing.assert_close(p, torch.softmax(log / 4, 1))
        torch.testing.assert_close(a, ai)
        torch.testing.assert_close(n, ai)

    def test_missing_class_stays_error(self):
        m = classification([0, 0, 2, 2], [0, 1, 2, 2], loco=True)
        self.assertAlmostEqual(m["macro_f1"], 5 / 6)
        self.assertEqual(m["predicted_M"], 1)
        self.assertEqual(m["confusion_HML"][0][1], 1)

    def test_baseline_and_bca_shapes(self):
        cfg = self.cfg()
        im = torch.randn(2, 3, 32, 32)
        nir = torch.randn(2, 499)
        for task in TASKS:
            with self.subTest(task=task):
                model = build_model(task, cfg).eval()
                with torch.no_grad():
                    out = (
                        model(im, nir, torch.randn(2, 4)) if TASKS[task].teacher else model(im, nir)
                    )
                self.assertEqual(out[0].shape, (2, 3))
                self.assertTrue(torch.isfinite(out[0]).all())

    def test_modality_mask_is_shared_train_inference(self):
        cfg = self.cfg()
        im = torch.randn(2, 3, 32, 32)
        nir = torch.randn(2, 499)
        for task, swap in [("student_image", "nir"), ("student_nir", "image")]:
            m = build_model(task, cfg).eval()
            with torch.no_grad():
                a = m(im, nir)[0]
                b = (
                    m(im, torch.randn_like(nir))[0]
                    if swap == "nir"
                    else m(torch.randn_like(im), nir)[0]
                )
            torch.testing.assert_close(a, b)

    def test_checkpoint_state_only(self):
        m = build_model("nir_only", self.cfg())
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best.pt"
            save_checkpoint(m, p, {"seed": 42}, model_task="nir_only")
            self.assertEqual(read_checkpoint(p)["meta"]["seed"], 42)

    def test_gradcam_is_actual_model_path(self):
        from figure_visualization.gradcam import compute

        cfg = self.cfg()
        for task in ["image_only", "bca_full"]:
            m = build_model(task, cfg)
            cam, cls = compute(m, torch.randn(1, 3, 32, 32), torch.randn(1, 499))
            self.assertEqual(cam.shape, (32, 32))
            self.assertTrue(np.isfinite(cam).all())
            self.assertIn(cls, [0, 1, 2])

    def test_all_paper_defaults(self):
        c = Config()
        self.assertEqual((c.batch_size, c.eval_batch_size, c.temperature), (6, 256, 4))
        self.assertEqual((c.loco_epochs, c.loco_lr, c.loco_weight_decay), (60, 4e-4, 0.05))
        self.assertEqual(c.embed_dim, 512)

    def test_path_placeholder_rejected(self):
        with self.assertRaisesRegex(ValueError, "Replace.*data_dir"):
            load_config()

    def test_data_card_scope_ids(self):
        self.assertEqual(resolve_experiment("B03"), "bca_full")
        self.assertEqual(resolve_experiment("C14"), "student_image")
        self.assertEqual(result_ids("student_image", "B1-Test"), ["C14"])
        self.assertEqual(result_ids("student_image", "B2-All"), ["C15"])
        self.assertEqual(result_ids("bca_full", "B3-All"), [])
        self.assertEqual(result_ids("partial_B2_B3", "B2-Holdout"), ["C09"])
        self.assertEqual(result_ids("partial_B2", "B2-Holdout"), [])

    def test_unicode_run_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "训练输出"
            meta = {"path_label": "菊花训练"}
            self.assertTrue(prepare_output(path, meta))
            json_write({}, path / "COMPLETED.json")
            self.assertFalse(prepare_output(path, meta, reuse=True))


if __name__ == "__main__":
    unittest.main()
