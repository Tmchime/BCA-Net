"""实验名称、模型及损失项配置。"""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    model: str
    teacher: bool = False
    kd: bool = False
    kl: bool = False
    mse_img: bool = False
    mse_nir: bool = False
    coral: bool = False
    dann: bool = False
    swapped: bool = False
    mode: str = "multi"


TASKS = {
    "teacher": Task("teacher", teacher=True),
    "teacher_swapped": Task("teacher", teacher=True, swapped=True),
    "image_only": Task("image_only"),
    "nir_only": Task("nir_only"),
    "early_concat": Task("early_concat"),
    "late_fusion": Task("late_fusion"),
    "cross_modal_transformer": Task("cross_modal_transformer"),
    "vanilla_kd": Task("early_concat", kd=True, kl=True),
    "deepcoral": Task("early_concat", coral=True),
    "dann": Task("dann", dann=True),
    "bca_source": Task("bca", kd=True, kl=True, mse_img=True, mse_nir=True),
    "bca_full": Task("bca", kd=True, kl=True, mse_img=True, mse_nir=True, coral=True),
    "cross_attention_only": Task("bca"),
    "without_all_kd": Task("bca", coral=True),
    "without_kl": Task("bca", kd=True, mse_img=True, mse_nir=True, coral=True),
    "without_mse_img": Task("bca", kd=True, kl=True, mse_nir=True, coral=True),
    "without_mse_nir": Task("bca", kd=True, kl=True, mse_img=True, coral=True),
    "activation_swap": Task(
        "bca", kd=True, kl=True, mse_img=True, mse_nir=True, coral=True, swapped=True
    ),
    "student_nir": Task("bca", kd=True, kl=True, mse_nir=True, coral=True, mode="nir"),
    "student_image": Task("bca", kd=True, kl=True, mse_img=True, coral=True, mode="image"),
}


def build_model(task, cfg, pretrained=None):
    spec = TASKS[task]
    if spec.model in {"bca", "teacher"}:
        from bca_net.model import BCAStudent, BCATeacher

        if spec.teacher:
            return BCATeacher(cfg, swapped=spec.swapped, pretrained=pretrained)
        return BCAStudent(cfg, swapped=spec.swapped, pretrained=pretrained, mode=spec.mode)
    if pretrained is not None:
        from dataclasses import replace

        cfg = replace(cfg, pretrained=pretrained)
    return importlib.import_module(f"table2_baselines.{spec.model}.model").Model(cfg)
