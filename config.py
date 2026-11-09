

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CONDITIONS = ("C0", "C1", "C2", "C3")

CONDITION_LABELS = {
    "C0": "no safeguard",
    "C1": "privacy-aware prompt",
    "C2": "regex PII filter",
    "C3": "pre-send action review",
}


BASE_URL = "https://api.deepseek.com"
GEN_MODEL = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"
ADJUDICATOR_MODEL = "deepseek-v4-pro"

TEMPERATURE = 0.0
MAX_TOKENS_GEN = 1200
MAX_TOKENS_JUDGE = 2500
MAX_TOKENS_ADJUDICATE = 4000
REQUEST_TIMEOUT = 180


@dataclass
class Settings:


    root: Path
    data: Path
    key: Path
    repeats: int = 3
    concurrency: int = 6
    temperature: float = TEMPERATURE
    gen_model: str = GEN_MODEL
    judge_model: str = JUDGE_MODEL
    adjudicator_model: str = ADJUDICATOR_MODEL

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.data = Path(self.data).resolve() if self.data else self.root / "main_data.json"
        self.key = Path(self.key).resolve() if self.key else self.root / "DSAPIKey.txt"
        self.cache_dir = self.root / "cache"
        self.runs_dir = self.root / "runs"
        self.results_dir = self.root / "results"
        for directory in (self.cache_dir, self.runs_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def annotations_path(self) -> Path:
        return self.root / "annotations.json"

    @property
    def generations_path(self) -> Path:
        return self.runs_dir / "generations.jsonl"

    @property
    def judgments_path(self) -> Path:
        return self.runs_dir / "judgments.jsonl"
