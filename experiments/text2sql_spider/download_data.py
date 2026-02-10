from __future__ import annotations
import argparse, os, json
from datasets import load_dataset

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    ds = load_dataset("xlangai/spider")  # has train/dev + db paths metadata
    for split in ["train", "validation"]:
        out = os.path.join(args.data_dir, f"{split}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for ex in ds[split]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[ok] wrote Spider splits to {args.data_dir}")

if __name__ == "__main__":
    main()
