"""Model list, paths, layer sweep helpers for train_probe_automated."""
import os
from typing import List, Dict
from dataclasses import dataclass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))

ROOT = os.environ.get("RNA_PROBE_DATA_ROOT", _REPO_ROOT)
EMBEDDING_BASE_DIR = os.path.join(ROOT, "data", "embeddings")
DATASET_NAME = "bpRNA"

RESULTS_DIR_NAME = os.environ.get("RESULTS_DIR", "results")
RESULTS_BASE = os.environ.get(
    "RNA_PROBE_RESULTS_BASE",
    os.path.join(ROOT, RESULTS_DIR_NAME),
)

# ---- Model configurations --------------------------------------------------
MODEL_CONFIGS = {
    "rnabert": {
        "embedding_type": "rnabert",
        "base_path": os.path.join(EMBEDDING_BASE_DIR, "RNABERT", DATASET_NAME, "by_layer"),
    },
    "rnafm": {
        "embedding_type": "rnafm",
        "base_path": os.path.join(EMBEDDING_BASE_DIR, "RNAFM", DATASET_NAME, "by_layer"),
    },
    "ernie": {
        "embedding_type": "ernie",
        "base_path": os.path.join(EMBEDDING_BASE_DIR, "RNAErnie", DATASET_NAME, "by_layer"),
    },
    "rinalmo": {
        "embedding_type": "rinalmo",
        "base_path": os.path.join(EMBEDDING_BASE_DIR, "RiNALMo", DATASET_NAME, "by_layer"),
    },
    "roberta": {
        "embedding_type": "roberta",
        "base_path": os.path.join(EMBEDDING_BASE_DIR, "RoBERTa", DATASET_NAME, "by_layer"),
    },
    "onehot": {
        "embedding_type": "onehot",
        "base_path": os.path.join(EMBEDDING_BASE_DIR, "Onehot", DATASET_NAME, "by_layer"),
    },
}

# ---- Experiment hyper-parameter grid ----------------------------------------
K_VALUES = [32, 64, 128]
SEED = 42
THRESHOLD_SWEEP_START = 0.50
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.05


def detect_model_layers(model_name: str) -> List[int]:
    """Return sorted list of available layer indices for *model_name*.

    Scans the embedding directory for ``layer_<N>`` subdirectories.
    For the one-hot baseline, returns ``[0]`` unconditionally.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    if model_name == "onehot":
        return [0]

    base_path = MODEL_CONFIGS[model_name]["base_path"]
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Embedding directory not found: {base_path}")

    layer_dirs = []
    for item in os.listdir(base_path):
        if item.startswith("layer_") and os.path.isdir(os.path.join(base_path, item)):
            try:
                layer_dirs.append(int(item.split("_")[1]))
            except ValueError:
                continue

    if not layer_dirs:
        raise ValueError(f"No layer directories found in {base_path}")

    layer_dirs.sort()
    return layer_dirs


def get_all_experiment_configs(model_names: List[str]) -> List[Dict]:
    """Generate the full Cartesian product of (model, layer, k, seed) configs."""
    configs = []
    for model_name in model_names:
        layers = detect_model_layers(model_name)
        for layer in layers:
            for k in K_VALUES:
                configs.append({
                    "model": model_name,
                    "layer": layer,
                    "k": k,
                    "seed": SEED,
                })
    return configs


@dataclass
class ExperimentConfig:
    """Configuration for a single probe training run."""
    model: str
    layer: int
    k: int
    seed: int

    @property
    def embedding_type(self) -> str:
        return MODEL_CONFIGS[self.model]["embedding_type"]

    @property
    def run_dir_name(self) -> str:
        return f"layer_{self.layer}/k_{self.k}/seed_{self.seed}"

    @property
    def run_dir(self) -> str:
        return os.path.join(RESULTS_BASE, "outputs", self.model, self.run_dir_name)

    def get_checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, "best.pt")

    def __str__(self):
        return f"{self.model}_L{self.layer}_k{self.k}_seed{self.seed}"
