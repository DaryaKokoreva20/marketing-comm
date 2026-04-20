import json

from app.core.paths import METADATA_PATH


def read_metadata() -> dict:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(data: dict) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
