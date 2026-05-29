"""
generate_dataset.py
-------------------
Generates a synthetic NLP-annotated dataset for the Perfume Ontology.
Outputs JSONL files split into train / val / test with tiered complexity.

Complexity tiers:
  EASY   - single entity, single relation
  MEDIUM - 2 entities, 1-2 relations, one optional modifier
  HARD   - 3+ entities, chained relations, negation/comparison

Each record follows the NER + RE annotation schema:
{
  "id": "...",
  "tier": "easy|medium|hard",
  "text": "...",
  "entities": [{"id":"E0","start":int,"end":int,"label":"CLASS","text":"..."}],
  "relations": [{"head":"E0","tail":"E1","label":"PROPERTY"}]
}
"""

import json, random, pathlib, hashlib

random.seed(42)

# ── Ontology vocabulary (derived from perfume_ontology.ttl) ──────────────────
BASE = "http://www.semanticweb.org/adea/ontologies/2026/4/data.perfume.ont#/"

BRANDS      = ["Chanel", "Dior", "Guerlain", "Hermès", "Creed", "Tom Ford",
               "Yves Saint Laurent", "Givenchy", "Versace", "Prada"]
FRAGRANCES  = {
    "Dior_Sauvage":     {"brand":"Dior",    "family":"Fougere",  "type":"EauDeToilette","top":"Pepper"},
    "Bleu_de_Chanel":   {"brand":"Chanel",  "family":"Woody",    "type":"EauDeParfum", "top":"Mint"},
    "Chanel_Chance":    {"brand":"Chanel",  "family":"Floral",   "type":"EauDeParfum", "top":"Bergamot"},
    "La_Vie_Est_Belle": {"brand":"Lancôme", "family":"Gourmand", "type":"EauDeParfum", "top":"Iris"},
    "Aventus":          {"brand":"Creed",   "family":"Chypre",   "type":"EauDeParfum", "top":"Blackcurrant"},
    "Black_Orchid":     {"brand":"Tom Ford","family":"Floral",   "type":"EauDeParfum", "top":"Truffle"},
    "Libre":            {"brand":"YSL",     "family":"Floral",   "type":"EauDeParfum", "top":"Lavender"},
    "Oud_Wood":         {"brand":"Tom Ford","family":"Woody",    "type":"EauDeParfum", "top":"Oud"},
}
TOP_NOTES   = ["Bergamot", "Lemon", "Pepper", "Mint", "Lavender", "Blackcurrant",
               "Grapefruit", "Orange Blossom", "Iris", "Truffle", "Oud"]
HEART_NOTES = ["Rose", "Jasmine", "Geranium", "Iris", "Patchouli", "Vetiver",
               "Sandalwood", "Cedar", "Neroli", "Ylang-Ylang"]
BASE_NOTES  = ["Musk", "Amber", "Oud", "Vanilla", "Benzoin", "Sandalwood",
               "Patchouli", "Labdanum", "Oakmoss", "Civet"]
FAMILIES    = ["Woody", "Floral", "Oriental", "Chypre", "Fougere",
               "Citrus", "Gourmand", "Aquatic", "Leather"]
PERFUMERS   = ["François Demachy", "Jacques Polge", "Alberto Morillas",
               "Dominique Ropion", "Olivier Polge", "Nathalie Lorson"]
OCCASIONS   = ["evening", "daytime", "office", "casual", "formal", "romantic", "outdoor"]
SEASONS     = ["Summer", "Winter", "Spring", "Autumn"]
GENDERS     = ["masculine", "feminine", "unisex"]
LONGEVITY   = ["4", "6", "8", "10", "12"]
SILLAGE     = ["intimate", "moderate", "strong", "enormous"]

PROPERTIES  = {
    "manufacturedBy":       ("Fragrance","Brand"),
    "belongsToFamily":      ("Fragrance","OlfactoryFamily"),
    "hasTopNote":           ("Fragrance","TopNote"),
    "hasHeartNote":         ("Fragrance","HeartNote"),
    "hasBaseNote":          ("Fragrance","BaseNote"),
    "createdBy":            ("Fragrance","Perfumer"),
    "recommendedInSeason":  ("Fragrance","Season"),
    "targetGender":         ("Fragrance","GenderTarget"),
    "suitableFor":          ("Fragrance","Occasion"),
    "containsIngredient":   ("Fragrance","Ingredient"),
    "similarTo":            ("Fragrance","Fragrance"),
    "inspiredBy":           ("Fragrance","Fragrance"),
}

# ── Template bank ────────────────────────────────────────────────────────────

def make_entity(eid, text, label, start):
    return {"id": eid, "start": start, "end": start + len(text), "label": label, "text": text}

def find_span(text, token):
    idx = text.find(token)
    return idx

# --- EASY templates -----------------------------------------------------------
EASY_TEMPLATES = [
    lambda: _easy_manufactured(),
    lambda: _easy_family(),
    lambda: _easy_top_note(),
    lambda: _easy_heart_note(),
    lambda: _easy_base_note(),
    lambda: _easy_season(),
    lambda: _easy_gender(),
    lambda: _easy_occasion(),
]

def _easy_manufactured():
    name, info = random.choice(list(FRAGRANCES.items()))
    display = name.replace("_", " ")
    brand   = info["brand"]
    text    = f"{display} is manufactured by {brand}."
    E0 = make_entity("E0", display, "Fragrance", find_span(text, display))
    E1 = make_entity("E1", brand,   "Brand",     find_span(text, brand))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"manufacturedBy"}]

def _easy_family():
    name, info = random.choice(list(FRAGRANCES.items()))
    display = name.replace("_", " ")
    family  = info["family"]
    text    = f"{display} belongs to the {family} olfactory family."
    E0 = make_entity("E0", display, "Fragrance",      find_span(text, display))
    E1 = make_entity("E1", family,  "OlfactoryFamily",find_span(text, family))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"belongsToFamily"}]

def _easy_top_note():
    name, info = random.choice(list(FRAGRANCES.items()))
    display = name.replace("_", " ")
    note    = info["top"]
    text    = f"The top note of {display} is {note}."
    E0 = make_entity("E0", display, "Fragrance", find_span(text, display))
    E1 = make_entity("E1", note,    "TopNote",   find_span(text, note))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"hasTopNote"}]

def _easy_heart_note():
    note    = random.choice(HEART_NOTES)
    name, _ = random.choice(list(FRAGRANCES.items()))
    display = name.replace("_", " ")
    text    = f"{display} has a heart note of {note}."
    E0 = make_entity("E0", display, "Fragrance", find_span(text, display))
    E1 = make_entity("E1", note,    "HeartNote", find_span(text, note))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"hasHeartNote"}]

def _easy_base_note():
    note    = random.choice(BASE_NOTES)
    name, _ = random.choice(list(FRAGRANCES.items()))
    display = name.replace("_", " ")
    text    = f"The base note of {display} is {note}."
    E0 = make_entity("E0", display, "Fragrance", find_span(text, display))
    E1 = make_entity("E1", note,    "BaseNote",  find_span(text, note))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"hasBaseNote"}]

def _easy_season():
    name, _  = random.choice(list(FRAGRANCES.items()))
    display  = name.replace("_", " ")
    season   = random.choice(SEASONS)
    text     = f"{display} is recommended for {season}."
    E0 = make_entity("E0", display, "Fragrance", find_span(text, display))
    E1 = make_entity("E1", season,  "Season",    find_span(text, season))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"recommendedInSeason"}]

def _easy_gender():
    name, _  = random.choice(list(FRAGRANCES.items()))
    display  = name.replace("_", " ")
    gender   = random.choice(GENDERS)
    text     = f"{display} is targeted at {gender} customers."
    E0 = make_entity("E0", display, "Fragrance",   find_span(text, display))
    E1 = make_entity("E1", gender,  "GenderTarget", find_span(text, gender))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"targetGender"}]

def _easy_occasion():
    name, _  = random.choice(list(FRAGRANCES.items()))
    display  = name.replace("_", " ")
    occ      = random.choice(OCCASIONS)
    text     = f"{display} is suitable for {occ} wear."
    E0 = make_entity("E0", display, "Fragrance", find_span(text, display))
    E1 = make_entity("E1", occ,     "Occasion",  find_span(text, occ))
    return text, [E0, E1], [{"head":"E0","tail":"E1","label":"suitableFor"}]

# --- MEDIUM templates ---------------------------------------------------------
def _medium_brand_and_family():
    name, info = random.choice(list(FRAGRANCES.items()))
    display = name.replace("_", " ")
    brand   = info["brand"]
    family  = info["family"]
    text    = f"{display}, produced by {brand}, is classified as a {family} fragrance."
    E0 = make_entity("E0", display, "Fragrance",       find_span(text, display))
    E1 = make_entity("E1", brand,   "Brand",           find_span(text, brand))
    E2 = make_entity("E2", family,  "OlfactoryFamily", find_span(text, family))
    return text, [E0,E1,E2], [
        {"head":"E0","tail":"E1","label":"manufacturedBy"},
        {"head":"E0","tail":"E2","label":"belongsToFamily"},
    ]

def _medium_top_and_heart():
    top     = random.choice(TOP_NOTES)
    heart   = random.choice(HEART_NOTES)
    name, _ = random.choice(list(FRAGRANCES.items()))
    display = name.replace("_", " ")
    text    = f"{display} opens with {top} and has a heart of {heart}."
    E0 = make_entity("E0", display, "Fragrance", find_span(text, display))
    E1 = make_entity("E1", top,     "TopNote",   find_span(text, top))
    E2 = make_entity("E2", heart,   "HeartNote", find_span(text, heart))
    return text, [E0,E1,E2], [
        {"head":"E0","tail":"E1","label":"hasTopNote"},
        {"head":"E0","tail":"E2","label":"hasHeartNote"},
    ]

def _medium_perfumer_and_brand():
    name, info = random.choice(list(FRAGRANCES.items()))
    display  = name.replace("_", " ")
    brand    = info["brand"]
    perfumer = random.choice(PERFUMERS)
    text     = f"{display} was created by {perfumer} for {brand}."
    E0 = make_entity("E0", display,  "Fragrance", find_span(text, display))
    E1 = make_entity("E1", perfumer, "Perfumer",  find_span(text, perfumer))
    E2 = make_entity("E2", brand,    "Brand",     find_span(text, brand))
    return text, [E0,E1,E2], [
        {"head":"E0","tail":"E1","label":"createdBy"},
        {"head":"E0","tail":"E2","label":"manufacturedBy"},
    ]

def _medium_season_and_gender():
    name, _  = random.choice(list(FRAGRANCES.items()))
    display  = name.replace("_", " ")
    season   = random.choice(SEASONS)
    gender   = random.choice(GENDERS)
    text     = f"{display} is a {gender} fragrance best worn in {season}."
    E0 = make_entity("E0", display, "Fragrance",   find_span(text, display))
    E1 = make_entity("E1", gender,  "GenderTarget", find_span(text, gender))
    E2 = make_entity("E2", season,  "Season",       find_span(text, season))
    return text, [E0,E1,E2], [
        {"head":"E0","tail":"E1","label":"targetGender"},
        {"head":"E0","tail":"E2","label":"recommendedInSeason"},
    ]

MEDIUM_TEMPLATES = [
    _medium_brand_and_family,
    _medium_top_and_heart,
    _medium_perfumer_and_brand,
    _medium_season_and_gender,
]

# --- HARD templates -----------------------------------------------------------
def _hard_full_profile():
    name, info = random.choice(list(FRAGRANCES.items()))
    display  = name.replace("_", " ")
    brand    = info["brand"]
    family   = info["family"]
    top      = info["top"]
    heart    = random.choice(HEART_NOTES)
    base     = random.choice(BASE_NOTES)
    text = (f"{display} is an {info['type'].replace('EauDe','')} by {brand}. "
            f"It belongs to the {family} family and opens with {top}, "
            f"transitions into a heart of {heart}, and settles on {base}.")
    E0 = make_entity("E0", display, "Fragrance",       find_span(text, display))
    E1 = make_entity("E1", brand,   "Brand",           find_span(text, brand))
    E2 = make_entity("E2", family,  "OlfactoryFamily", find_span(text, family))
    E3 = make_entity("E3", top,     "TopNote",         find_span(text, top))
    E4 = make_entity("E4", heart,   "HeartNote",       find_span(text, heart))
    E5 = make_entity("E5", base,    "BaseNote",        find_span(text, base))
    return text, [E0,E1,E2,E3,E4,E5], [
        {"head":"E0","tail":"E1","label":"manufacturedBy"},
        {"head":"E0","tail":"E2","label":"belongsToFamily"},
        {"head":"E0","tail":"E3","label":"hasTopNote"},
        {"head":"E0","tail":"E4","label":"hasHeartNote"},
        {"head":"E0","tail":"E5","label":"hasBaseNote"},
    ]

def _hard_comparison():
    keys   = random.sample(list(FRAGRANCES.keys()), 2)
    n1, n2 = keys[0].replace("_"," "), keys[1].replace("_"," ")
    i1, i2 = FRAGRANCES[keys[0]], FRAGRANCES[keys[1]]
    text = (f"While {n1} by {i1['brand']} is a {i1['family']} scent, "
            f"{n2} by {i2['brand']} belongs to the {i2['family']} family. "
            f"Both are popular {i1['type'].replace('EauDe','')} choices.")
    E0 = make_entity("E0", n1,           "Fragrance",       find_span(text, n1))
    E1 = make_entity("E1", i1["brand"],  "Brand",           find_span(text, i1["brand"]))
    E2 = make_entity("E2", i1["family"], "OlfactoryFamily", find_span(text, i1["family"]))
    E3 = make_entity("E3", n2,           "Fragrance",       find_span(text, n2))
    E4 = make_entity("E4", i2["brand"],  "Brand",           text.rfind(i2["brand"]))
    E5 = make_entity("E5", i2["family"], "OlfactoryFamily", text.rfind(i2["family"]))
    return text, [E0,E1,E2,E3,E4,E5], [
        {"head":"E0","tail":"E1","label":"manufacturedBy"},
        {"head":"E0","tail":"E2","label":"belongsToFamily"},
        {"head":"E3","tail":"E4","label":"manufacturedBy"},
        {"head":"E3","tail":"E5","label":"belongsToFamily"},
    ]

def _hard_negation():
    name, info = random.choice(list(FRAGRANCES.items()))
    display  = name.replace("_", " ")
    brand    = info["brand"]
    wrong_f  = random.choice([f for f in FAMILIES if f != info["family"]])
    top      = info["top"]
    season   = random.choice(SEASONS)
    text = (f"{display} by {brand} does not belong to the {wrong_f} family. "
            f"Its top note is {top} and it is recommended for {season}.")
    E0 = make_entity("E0", display,  "Fragrance",       find_span(text, display))
    E1 = make_entity("E1", brand,    "Brand",           find_span(text, brand))
    E2 = make_entity("E2", wrong_f,  "OlfactoryFamily", find_span(text, wrong_f))
    E3 = make_entity("E3", top,      "TopNote",         find_span(text, top))
    E4 = make_entity("E4", season,   "Season",          find_span(text, season))
    return text, [E0,E1,E2,E3,E4], [
        {"head":"E0","tail":"E1","label":"manufacturedBy"},
        {"head":"E0","tail":"E2","label":"NOT_belongsToFamily"},   # negated
        {"head":"E0","tail":"E3","label":"hasTopNote"},
        {"head":"E0","tail":"E4","label":"recommendedInSeason"},
    ]

HARD_TEMPLATES = [_hard_full_profile, _hard_comparison, _hard_negation]

# ── Record factory ────────────────────────────────────────────────────────────

def make_record(tier):
    if tier == "easy":
        fn = random.choice(EASY_TEMPLATES)
    elif tier == "medium":
        fn = random.choice(MEDIUM_TEMPLATES)
    else:
        fn = random.choice(HARD_TEMPLATES)

    text, entities, relations = fn()
    uid = hashlib.md5(text.encode()).hexdigest()[:12]
    return {"id": uid, "tier": tier, "text": text,
            "entities": entities, "relations": relations}

# ── Split & write ─────────────────────────────────────────────────────────────
# Distribution: 60% easy/medium/hard each → then split 70/15/15
COUNTS = {"easy": 200, "medium": 200, "hard": 150}  # total = 550

all_records = []
for tier, n in COUNTS.items():
    for _ in range(n):
        all_records.append(make_record(tier))

random.shuffle(all_records)
n     = len(all_records)
train = all_records[:int(n*0.70)]
val   = all_records[int(n*0.70):int(n*0.85)]
test  = all_records[int(n*0.85):]

BASE_DIR = pathlib.Path(__file__).parent.parent / "data"
for split, records in [("train", train), ("val", val), ("test", test)]:
    out = BASE_DIR / split / f"perfume_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {split}: {len(records)} records → {out}")

print("\nDataset generation complete.")
print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
