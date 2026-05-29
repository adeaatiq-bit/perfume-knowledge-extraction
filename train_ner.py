"""
train_ner.py  –  SpaCy baseline NER trainer for the Perfume Ontology dataset.

Usage:
    pip install spacy
    python -m spacy download en_core_web_sm
    python scripts/train_ner.py
"""

import json, pathlib, random, warnings
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

warnings.filterwarnings("ignore")
random.seed(42)

DATA_DIR  = pathlib.Path(__file__).parent.parent / "data"
MODEL_OUT = pathlib.Path(__file__).parent.parent / "models" / "spacy_ner"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# ── Load JSONL splits ─────────────────────────────────────────────────────────
def load_split(split: str):
    path = DATA_DIR / split / f"perfume_{split}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def records_to_spacy(records):
    """Convert dataset records to spaCy (text, {entities}) training format."""
    data = []
    for r in records:
        spans = []
        seen  = set()
        for e in r["entities"]:
            key = (e["start"], e["end"])
            if key not in seen:
                spans.append((e["start"], e["end"], e["label"]))
                seen.add(key)
        data.append((r["text"], {"entities": spans}))
    return data

train_data = records_to_spacy(load_split("train"))
val_data   = records_to_spacy(load_split("val"))

# ── Build pipeline ─────────────────────────────────────────────────────────────
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner", last=True)

LABELS = [
    "Fragrance", "Brand", "OlfactoryFamily", "TopNote", "HeartNote",
    "BaseNote", "Perfumer", "Season", "GenderTarget", "Occasion",
    "Ingredient",
]
for label in LABELS:
    ner.add_label(label)

# ── Training ───────────────────────────────────────────────────────────────────
n_iter = 30
optimizer = nlp.begin_training()

print(f"Training SpaCy NER for {n_iter} iterations on {len(train_data)} examples …")
losses_log = []

for i in range(n_iter):
    random.shuffle(train_data)
    losses = {}
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        nlp.update(examples, drop=0.35, losses=losses)
    losses_log.append(losses.get("ner", 0))
    if (i + 1) % 5 == 0:
        print(f"  iter {i+1:3d}/{n_iter}  NER loss: {losses.get('ner', 0):.4f}")

# ── Quick eval on val set ──────────────────────────────────────────────────────
print("\nValidation scores:")
scorer_data = []
for text, annot in val_data[:50]:
    doc  = nlp.make_doc(text)
    ex   = Example.from_dict(doc, annot)
    scorer_data.append(ex)

scores = nlp.evaluate(scorer_data)
p = scores.get("ents_p", 0)
r = scores.get("ents_r", 0)
f = scores.get("ents_f", 0)
print(f"  Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f}")

# ── Save model ─────────────────────────────────────────────────────────────────
nlp.to_disk(MODEL_OUT)
print(f"\n✓ Model saved → {MODEL_OUT}")

# ── Write benchmark results ────────────────────────────────────────────────────
results = {"model": "spacy_baseline", "precision": p, "recall": r, "f1": f,
           "n_train": len(train_data), "n_val": len(val_data),
           "losses": losses_log}
report_path = pathlib.Path(__file__).parent.parent / "reports" / "spacy_ner_results.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved → {report_path}")
