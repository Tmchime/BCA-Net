"""实验编号"""

EXPERIMENT_IDS = {
    "teacher": ("T00", "C01"),
    "teacher_swapped": ("B08T",),
    "image_only": ("A01",),
    "nir_only": ("A02",),
    "early_concat": ("A03",),
    "late_fusion": ("A04",),
    "cross_modal_transformer": ("A05",),
    "vanilla_kd": ("A06",),
    "deepcoral": ("A07",),
    "dann": ("A08",),
    "bca_full": ("A09", "B03"),
    "cross_attention_only": ("B01",),
    "bca_source": ("B02", "C02", "C03", "C04"),
    "without_all_kd": ("B04",),
    "without_kl": ("B05",),
    "without_mse_img": ("B06",),
    "without_mse_nir": ("B07",),
    "activation_swap": ("B08",),
    "transductive_B2_B3": ("C05", "C06", "C07"),
    "partial_B2_B3": ("C08", "C09", "C10"),
    "loco5": ("C11",),
    "student_nir": ("C12", "C13"),
    "student_image": ("C14", "C15"),
}

ALIASES = {code: task for task, codes in EXPERIMENT_IDS.items() for code in codes}

RESULT_IDS = {
    ("teacher", "B1-Test"): ("T00", "C01"),
    ("teacher_swapped", "B1-Test"): ("B08T",),
    ("bca_source", "B1-Test"): ("B02", "C02"),
    ("bca_source", "B2-All"): ("B02", "C03"),
    ("bca_source", "B3-All"): ("C04",),
    ("transductive_B2_B3", "B1-Test"): ("C05",),
    ("transductive_B2_B3", "B2-All"): ("C06",),
    ("transductive_B2_B3", "B3-All"): ("C07",),
    ("partial_B2_B3", "B1-Test"): ("C08",),
    ("partial_B2_B3", "B2-Holdout"): ("C09",),
    ("partial_B2_B3", "B3-Holdout"): ("C10",),
    ("loco5", "B3-LOCO5"): ("C11",),
    ("student_nir", "B1-Test"): ("C12",),
    ("student_nir", "B2-All"): ("C13",),
    ("student_image", "B1-Test"): ("C14",),
    ("student_image", "B2-All"): ("C15",),
}

for _task, _codes in EXPERIMENT_IDS.items():
    if _codes[0].startswith("A") or _task in {
        "cross_attention_only",
        "without_all_kd",
        "without_kl",
        "without_mse_img",
        "without_mse_nir",
        "activation_swap",
    }:
        for _scope in ("B1-Test", "B2-All"):
            RESULT_IDS[(_task, _scope)] = _codes


def result_ids(task, evaluation):
    return list(RESULT_IDS.get((task, evaluation), ()))


def resolve_experiment(name):
    return ALIASES.get(name, name)
