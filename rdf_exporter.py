"""
rdf_exporter.py  –  Converts JSONL annotated records to RDF/Turtle,
aligning extracted entities/relations back to the perfume ontology IRI.

Usage:
    pip install rdflib
    python scripts/rdf_exporter.py --split test --output outputs/extracted.ttl
"""

import json, argparse, pathlib, re
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
OUT_DIR  = pathlib.Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Namespaces ─────────────────────────────────────────────────────────────────
ONT  = Namespace("http://www.semanticweb.org/adea/ontologies/2026/4/data.perfume.ont#/")
XSD_ = XSD

CLASS_MAP = {
    "Fragrance":       ONT.Fragrance,
    "Brand":           ONT.Brand,
    "OlfactoryFamily": ONT.OlfactoryFamily,
    "TopNote":         ONT.TopNote,
    "HeartNote":       ONT.HeartNote,
    "BaseNote":        ONT.BaseNote,
    "Perfumer":        ONT.Perfumer,
    "Season":          ONT.Season,
    "GenderTarget":    ONT.GenderTarget,
    "Occasion":        ONT.Occasion,
    "Ingredient":      ONT.Ingredient,
    "Note":            ONT.Note,
    "EauDeParfum":     ONT.EauDeParfum,
    "EauDeToilette":   ONT.EauDeToilette,
    "EauDeCologne":    ONT.EauDeCologne,
    "Perfume":         ONT.Perfume,
}

PROP_MAP = {
    "manufacturedBy":      ONT.manufacturedBy,
    "belongsToFamily":     ONT.belongsToFamily,
    "hasTopNote":          ONT.hasTopNote,
    "hasHeartNote":        ONT.hasHeartNote,
    "hasBaseNote":         ONT.hasBaseNote,
    "hasNote":             ONT.hasNote,
    "createdBy":           ONT.createdBy,
    "recommendedInSeason": ONT.recommendedInSeason,
    "targetGender":        ONT.targetGender,
    "suitableFor":         ONT.suitableFor,
    "containsIngredient":  ONT.containsIngredient,
    "similarTo":           ONT.similarTo,
    "inspiredBy":          ONT.inspiredBy,
    "worksFor":            ONT.worksFor,
}

def slugify(text: str) -> str:
    """Turn entity surface form into a clean IRI fragment."""
    return re.sub(r"\s+", "_", text.strip().replace("/", "-"))

def export(split: str, output: pathlib.Path):
    path = DATA_DIR / split / f"perfume_{split}.jsonl"
    records = [json.loads(l) for l in open(path)]

    g = Graph()
    g.bind("ont",  ONT)
    g.bind("owl",  OWL)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    triples_added = 0

    for r in records:
        ent_map = {e["id"]: e for e in r["entities"]}

        # Declare each entity as an OWL NamedIndividual of its class
        for e in r["entities"]:
            slug = slugify(e["text"])
            uri  = ONT[slug]
            cls  = CLASS_MAP.get(e["label"])
            g.add((uri, RDF.type, OWL.NamedIndividual))
            if cls:
                g.add((uri, RDF.type, cls))
            g.add((uri, RDFS.label, Literal(e["text"], lang="en")))

        # Add relation triples (skip negated relations)
        for rel in r["relations"]:
            if rel["label"].startswith("NOT_"):
                continue
            prop = PROP_MAP.get(rel["label"])
            if prop is None:
                continue
            head_e = ent_map.get(rel["head"])
            tail_e = ent_map.get(rel["tail"])
            if head_e is None or tail_e is None:
                continue
            h_uri = ONT[slugify(head_e["text"])]
            t_uri = ONT[slugify(tail_e["text"])]
            g.add((h_uri, prop, t_uri))
            triples_added += 1

    g.serialize(destination=str(output), format="turtle")
    print(f"✓ Exported {triples_added} triples ({len(records)} records) → {output}")
    return triples_added

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="test", choices=["train","val","test"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out_path = pathlib.Path(args.output) if args.output else \
               OUT_DIR / f"extracted_{args.split}.ttl"
    export(args.split, out_path)
