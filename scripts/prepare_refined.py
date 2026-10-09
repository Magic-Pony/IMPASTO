"""Validate the frozen refined configurations without overwriting them."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    configs = [json.loads((ROOT / f"configs/stage{i}_refined.json").read_text())
               for i in (1, 2, 3)]
    stage1, stage2, stage3 = configs
    palette = stage1["paint"]["palette_rgb"]
    if len(palette) != 24 or stage2["palette_rgb"] != palette or stage3["palette_rgb"] != palette:
        raise ValueError("Refined palette mismatch; fix configs explicitly and retrain if needed")
    if stage3["stage1_config"] != stage2["stage1_config"]:
        raise ValueError("Stage-2 and Stage-3 use different simulators")
    if stage3["stage2_config"] != "configs/stage2_refined.json":
        raise ValueError("Stage-3 does not reference the refined dynamics configuration")
    if not (ROOT / stage3["checkpoint"]).is_file():
        raise FileNotFoundError(stage3["checkpoint"])
    print("Refined configurations and checkpoint are present; no files changed.")


if __name__ == "__main__":
    main()
