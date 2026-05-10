import json, random, re
from datasets import load_dataset

def clean_question(q: str) -> str:
    # MedXpertQA question often contains "Answer Choices: ..."
    # Keep only the prompt part to avoid duplicating options.
    return re.split(r"\bAnswer Choices:\b", q, maxsplit=1)[0].strip()

def options_dict_to_list(opts: dict) -> list[str]:
    # keys are letters: A, B, C... (Text can have up to 10 options; MM usually 5) :contentReference[oaicite:4]{index=4}
    return [opts[k] for k in sorted(opts.keys())]

def label_to_index(label: str) -> int:
    # label is a single letter like "C" :contentReference[oaicite:5]{index=5}
    return ord(label.strip().upper()) - ord("A")

def dump_jsonl(rows, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_medxpertqa(out_path: str, config: str = "Text", split: str = "test",
                    n: int | None = 200, seed: int = 42):
    ds = load_dataset("TsinghuaC3I/MedXpertQA", config, split=split)  # :contentReference[oaicite:6]{index=6}
    idxs = list(range(len(ds)))
    if n is not None and n < len(ds):
        random.Random(seed).shuffle(idxs)
        idxs = idxs[:n]

    rows = []
    for i in idxs:
        ex = ds[i]
        choices = options_dict_to_list(ex["options"])
        rows.append({
            "id": ex["id"],
            "dataset": f"medxpertqa_{config.lower()}_{split}",
            "question": clean_question(ex["question"]),
            "choices": choices,
            "answer_index": label_to_index(ex["label"]),
            "meta": {
                "subset": config,
                "split": split,
                "medical_task": ex.get("medical_task"),
                "body_system": ex.get("body_system"),
                "question_type": ex.get("question_type"),
                # keep raw if you want:
                "raw_question": ex["question"],
            }
        })

    dump_jsonl(rows, out_path)
    print(f"Wrote {len(rows)} -> {out_path}")

if __name__ == "__main__":
    # small dev sample first, then test
    build_medxpertqa("data/medxpertqa_text_dev_200.jsonl", config="Text", split="dev", n=5)
    build_medxpertqa("data/medxpertqa_text_test_50.jsonl", config="Text", split="test", n=50)
