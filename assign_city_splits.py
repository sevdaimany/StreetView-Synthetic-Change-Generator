import json
import random
from pathlib import Path


def assign_city_splits(
    data_base_dir: str,
    output_json: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Assigns each city folder to train/val/test without touching files.
    Saves the assignment to a JSON file and returns the dict.

    Expected input structure:
        data_base_dir / city_name / sequence_id / ...

    Output JSON format:
        {
            "train": ["city_a", "city_b", ...],
            "val":   ["city_c", ...],
            "test":  ["city_d", ...]
        }
    """
    data_path = Path(data_base_dir)
    cities = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

    random.seed(seed)
    random.shuffle(cities)

    total = len(cities)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    assignment = {
        "train": cities[:train_end],
        "val":   cities[train_end:val_end],
        "test":  cities[val_end:],
    }

    print(f"Total cities: {total}")
    print(f"  train ({len(assignment['train'])}): {assignment['train']}")
    print(f"  val   ({len(assignment['val'])}):   {assignment['val']}")
    print(f"  test  ({len(assignment['test'])}):  {assignment['test']}")

    with open(output_json, "w") as f:
        json.dump(assignment, f, indent=2)

    print(f"\nSplit saved to: {output_json}")
    return assignment


if __name__ == "__main__":
    SOURCE_DATA = "/mnt/stores/store-DAI/pocs/simany/pipeline_3/pipeline_data"
    OUTPUT_JSON = "city_splits.json"

    assign_city_splits(SOURCE_DATA, OUTPUT_JSON)
