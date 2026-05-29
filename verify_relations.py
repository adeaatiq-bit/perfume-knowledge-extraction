"""
verify_relations.py  –  Rule-based + ML verifier that checks extracted
(head, relation, tail) triples against the ontology's domain/range constraints.

Usage:
    python scripts/verify_relations.py --split test

Outputs a verification report to reports/verification_report.json
"""

import json, argparse, pathlib
from collections import defaultdict

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
REPORT_DIR  = pathlib.Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Domain/Range constraints from the ontology ────────────────────────────────
ONTOLOGY_CONSTRAINTS = {
    "manufacturedBy":       ("Fragrance", "Brand"),
    "belongsToFamily":      ("Fragrance", "OlfactoryFamily"),
    "hasTopNote":           ("Fragrance", "TopNote"),
    "hasHeartNote":         ("Fragrance", "HeartNote"),
    "hasBaseNote":          ("Fragrance", "BaseNote"),
    "hasNote":              ("Fragrance", "Note"),
    "createdBy":            ("Fragrance", "Perfumer"),
    "recommendedInSeason":  ("Fragrance", "Season"),
    "targetGender":         ("Fragrance", "GenderTarget"),
    "suitableFor":          ("Fragrance", "Occasion"),
    "containsIngredient":   ("Fragrance", "Ingredient"),
    "similarTo":            ("Fragrance", "Fragrance"),
    "inspiredBy":           ("Fragrance", "Fragrance"),
    "worksFor":             ("Fragrance", "Brand"),
    # Negated variant (added in hard-tier)
    "NOT_belongsToFamily":  ("Fragrance", "OlfactoryFamily"),
}

# Subclass hierarchy from the ontology
SUBCLASS_OF = {
    "EauDeParfum":   "Fragrance",
    "EauDeToilette": "Fragrance",
    "EauDeCologne":  "Fragrance",
    "Perfume":       "Fragrance",
    "TopNote":       "Note",
    "HeartNote":     "Note",
    "BaseNote":      "Note",
    "RawMaterial":   "Note",
    "Note":          "Ingredient",
    "Brand":         "Agent",
    "Perfumer":      "Agent",
    "Ingredient":    "MaterialEntity",
    "Fragrance":     "MaterialEntity",
}

def is_subclass(child: str, parent: str) -> bool:
    """Check whether `child` is-a `parent` via the subclass chain."""
    if child == parent:
        return True
    up = SUBCLASS_OF.get(child)
    if up is None:
        return False
    return is_subclass(up, parent)

def verify_triple(head_label: str, rel: str, tail_label: str) -> dict:
    if rel not in ONTOLOGY_CONSTRAINTS:
        return {"valid": False, "error": f"Unknown relation '{rel}'"}
    dom, rng = ONTOLOGY_CONSTRAINTS[rel]
    errors = []
    if not is_subclass(head_label, dom):
        errors.append(f"domain violation: '{head_label}' is not a subclass of '{dom}'")
    if not is_subclass(tail_label, rng):
        errors.append(f"range  violation: '{tail_label}' is not a subclass of '{rng}'")
    return {"valid": len(errors) == 0, "errors": errors}

def run_verification(split: str):
    path = DATA_DIR / split / f"perfume_{split}.jsonl"
    records = [json.loads(l) for l in open(path)]

    summary = defaultdict(lambda: {"ok": 0, "fail": 0, "examples": []})
    total_ok, total_fail = 0, 0

    for r in records:
        ent_map = {e["id"]: e["label"] for e in r["entities"]}
        for rel in r["relations"]:
            h_label = ent_map.get(rel["head"], "UNKNOWN")
            t_label = ent_map.get(rel["tail"], "UNKNOWN")
            result  = verify_triple(h_label, rel["label"], t_label)
            key     = rel["label"]
            if result["valid"]:
                summary[key]["ok"] += 1
                total_ok += 1
            else:
                summary[key]["fail"] += 1
                total_fail += 1
                if len(summary[key]["examples"]) < 3:
                    summary[key]["examples"].append({
                        "text":   r["text"][:80],
                        "triple": f"({h_label}) –{rel['label']}→ ({t_label})",
                        "errors": result["errors"],
                    })

    report = {
        "split":       split,
        "total_triples": total_ok + total_fail,
        "passed":      total_ok,
        "failed":      total_fail,
        "accuracy":    round(total_ok / max(1, total_ok + total_fail), 4),
        "by_relation": {k: dict(v) for k, v in summary.items()},
    }

    out = REPORT_DIR / f"verification_{split}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Verification report: {split} ===")
    print(f"  Triples checked : {report['total_triples']}")
    print(f"  Passed          : {report['passed']}")
    print(f"  Failed          : {report['failed']}")
    print(f"  Accuracy        : {report['accuracy']:.1%}")
    print(f"\n  By relation:")
    for rel, stats in sorted(report["by_relation"].items()):
        acc = stats["ok"] / max(1, stats["ok"] + stats["fail"])
        print(f"    {rel:<28} ok={stats['ok']:4d}  fail={stats['fail']:4d}  acc={acc:.0%}")
    print(f"\n✓ Report → {out}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train","val","test"])
    args = parser.parse_args()
    run_verification(args.split)
