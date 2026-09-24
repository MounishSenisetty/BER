"""Generate a synthetic business-entity-resolution dataset with the competition's shape.

    python scripts/make_synthetic_data.py --out data --n-train 6000 --n-test 4000

Creates
    data/train/source1.tsv source2.tsv source3.tsv train_ground_truth.tsv
    data/test/source1.tsv  source2.tsv source3.tsv
    data/test_labels/test_ground_truth.tsv      (hidden labels for local hold-out scoring)

Hard cases baked in: chain brands at many addresses, businesses sharing a building, vendor
specific schemas (S2 split address columns, S3 one free-text column), typos, abbreviations,
dropped legal suffixes / postal codes / whole addresses, store numbers, truncated names,
and unmatched distractor records that look like S1 entities.
"""
from __future__ import annotations

import argparse
import os
import random
import string

import pandas as pd

ADJ = ["Golden", "Sunrise", "Blue", "Royal", "Green", "Silver", "Star", "Pacific", "Metro", "Prime", "Apex",
       "Summit", "Harbor", "Cedar", "Maple", "Oak", "River", "Valley", "Liberty", "Eagle", "Northern",
       "Coastal", "Urban", "United", "First", "Premier", "Elite", "Rapid", "Bright", "Global", "Pioneer",
       "Heritage", "Crystal", "Diamond", "Evergreen", "Lotus", "Phoenix", "Sapphire", "Horizon", "Nova"]
NOUN = ["Leaf", "Stone", "Bridge", "Point", "Rock", "Wave", "Peak", "Field", "Ridge", "Crest", "Grove",
        "Bay", "Lake", "Hill", "Garden", "Tower", "Gate", "Path", "Spring", "Sky"]
SURN = ["Johnson", "Smith", "Patel", "Garcia", "Kumar", "Nguyen", "Brown", "Miller", "Sharma", "Lee",
        "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Martin", "Jackson", "White", "Harris", "Clark",
        "Lewis", "Walker", "Hall", "Young", "King", "Wright", "Lopez", "Hill", "Scott", "Adams", "Reddy",
        "Iyer", "Gupta", "Singh", "Chen", "Wang", "Kim", "Rossi", "Muller", "Dubois"]
BTYPE = ["Bakery", "Dental Clinic", "Auto Repair", "Hardware", "Pharmacy", "Restaurant", "Consulting",
         "Services", "Technologies", "Logistics", "Motors", "Salon", "Cafe", "Grocery", "Electronics",
         "Law Office", "Associates", "Builders", "Printing", "Fitness Center", "Insurance", "Realty",
         "Plumbing", "Florist", "Laundry", "Veterinary Hospital", "Management", "International Trading",
         "Medical Center", "Supply Company"]
LEGAL = ["Inc", "LLC", "Ltd", "Corp", "Co", "Pvt Ltd", "Incorporated", "Corporation", "Limited"]
CHAINS = ["QuickMart", "Sunny Burger", "FreshCo Grocery", "Metro Pharmacy", "Speedy Lube", "Bean There Coffee",
          "PizzaPoint", "ValueMart", "CityBank", "FitZone Gym", "QuickPrint", "HomeFix Hardware",
          "Tasty Tacos", "CarePlus Clinic", "PetPals"]
STREET = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Lake", "Hill", "Park", "Market",
          "Church", "Mill", "River", "Sunset", "Highland", "Lincoln", "Jefferson", "Madison", "Franklin",
          "MG", "Station", "Temple", "Gandhi", "Nehru", "Forest", "Spring", "Center", "Union", "Bridge"]
STYPE = [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr"),
         ("Lane", "Ln"), ("Court", "Ct"), ("Place", "Pl"), ("Parkway", "Pkwy"), ("Highway", "Hwy")]
DIRS = [("North", "N"), ("South", "S"), ("East", "E"), ("West", "W")]
CITIES = [("Springfield", "IL", "627"), ("Riverside", "CA", "925"), ("Franklin", "TN", "370"),
          ("Greenville", "SC", "296"), ("Bristol", "CT", "060"), ("Clinton", "IA", "527"),
          ("Madison", "WI", "537"), ("Georgetown", "TX", "786"), ("Salem", "OR", "973"),
          ("Fairview", "NJ", "070"), ("Arlington", "VA", "222"), ("Ashland", "KY", "411"),
          ("Burlington", "VT", "054"), ("Dover", "DE", "199"), ("Oxford", "MS", "386"),
          ("Pune", "MH", "411"), ("Mysore", "KA", "570"), ("Nagpur", "MH", "440")]
WORD_ABBR = {"Services": "Svcs", "International": "Intl", "Management": "Mgmt", "Associates": "Assoc",
             "Center": "Ctr", "Company": "Co", "Medical": "Med", "Technologies": "Tech", "Brothers": "Bros",
             "Supply": "Sup", "Hospital": "Hosp", "&": "and", "and": "&"}
LEGAL_ALT = {"Inc": ["Inc.", "Incorporated", "INC"], "LLC": ["L.L.C.", "llc"], "Ltd": ["Limited", "Ltd."],
             "Corp": ["Corporation", "Corp."], "Co": ["Company", "Co."], "Pvt Ltd": ["Private Limited", "Pvt. Ltd."],
             "Incorporated": ["Inc"], "Corporation": ["Corp"], "Limited": ["Ltd"]}


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    # ---------------------------------------------------------------- clean entities
    def name(self):
        r = self.r
        pat = r.random()
        if pat < 0.3:
            base = f"{r.choice(SURN)} {r.choice(BTYPE)}"
        elif pat < 0.55:
            base = f"{r.choice(ADJ)} {r.choice(NOUN)} {r.choice(BTYPE)}"
        elif pat < 0.7:
            base = f"{r.choice(SURN)} & {r.choice(SURN)} {r.choice(BTYPE)}"
        elif pat < 0.85:
            base = f"{r.choice(ADJ)} {r.choice(BTYPE)}"
        else:
            base = f"{r.choice(SURN)}'s {r.choice(ADJ)} {r.choice(BTYPE)}"
        if r.random() < 0.5:
            base += " " + r.choice(LEGAL)
        return base

    def address(self, city=None):
        r = self.r
        c = city or r.choice(CITIES)
        st, _ = r.choice(STYPE)
        street = r.choice(STREET)
        if r.random() < 0.2:
            street = f"{r.choice(DIRS)[0]} {street}"
        return {"num": str(r.randint(1, 9999)), "street": f"{street} {st}",
                "unit": f"Suite {r.randint(100, 900)}" if r.random() < 0.2 else "",
                "city": c[0], "state": c[1], "zip": c[2] + f"{r.randint(0, 99):02d}"}

    # ---------------------------------------------------------------- noise
    def typo(self, s, n=1):
        r = self.r
        for _ in range(n):
            if len(s) < 4:
                return s
            i = r.randrange(1, len(s) - 1)
            op = r.random()
            if op < 0.25:
                s = s[:i] + s[i + 1:]
            elif op < 0.5:
                s = s[:i] + r.choice(string.ascii_lowercase) + s[i:]
            elif op < 0.75:
                s = s[:i] + r.choice(string.ascii_lowercase) + s[i + 1:]
            else:
                s = s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]
        return s

    def noisy_name(self, name, vendor, chain=False):
        r = self.r
        toks = name.split()
        legal_hit = [l for l in sorted(LEGAL, key=len, reverse=True) if name.endswith(" " + l)]
        if legal_hit:
            l = legal_hit[0]
            core = name[: -len(l) - 1]
            u = r.random()
            name = core if u < 0.5 else (f"{core} {r.choice(LEGAL_ALT.get(l, [l]))}" if u < 0.75 else name)
        toks = name.split()
        if r.random() < 0.2:
            toks = [WORD_ABBR.get(t, t) for t in toks]
        name = " ".join(toks)
        if r.random() < (0.25 if vendor == 2 else 0.35):
            name = self.typo(name, 1 if r.random() < 0.7 else 2)
        if vendor == 3 and r.random() < 0.08 and len(name.split()) >= 3:
            name = " ".join(name.split()[:2])                        # truncated fragment
        if r.random() < 0.08 and len(name.split()) >= 3:
            ts = name.split()
            del ts[r.randrange(len(ts))]
            name = " ".join(ts)
        if chain and r.random() < 0.4:
            name += f" #{r.randint(1, 2999):04d}"
        elif r.random() < 0.05:
            name += r.choice([" - Downtown", " (Main Branch)", " Store"])
        u = r.random()
        if u < 0.25:
            name = name.upper()
        elif u < 0.35:
            name = name.lower()
        return name

    def noisy_addr(self, a, vendor):
        r = self.r
        a = dict(a)
        miss = 0.07 if vendor == 2 else 0.15
        if r.random() < miss:
            return {k: "" for k in a}
        street = a["street"]
        for full, ab in STYPE + DIRS:               # toggle long <-> short forms
            if r.random() < 0.5:
                street = " ".join(ab if w == full else (full if w == ab else w) for w in street.split())
        if r.random() < 0.15:
            street = self.typo(street)
        a["street"] = street
        if r.random() < 0.03:
            a["num"] = self.typo(a["num"] + "0", 1)[:-1] or a["num"]   # corrupt house number (hard positive)
        if r.random() < 0.15:
            a["unit"] = "" if a["unit"] else f"Ste {r.randint(1, 50)}"
        elif a["unit"] and r.random() < 0.5:
            a["unit"] = a["unit"].replace("Suite", r.choice(["Ste", "#", "Unit"]))
        if r.random() < 0.2:
            a["zip"] = ""
        if r.random() < 0.15:
            a["city"], a["state"] = "", ""
        if r.random() < 0.2:
            a = {k: v.upper() for k, v in a.items()}
        return a

    @staticmethod
    def addr_str(a):
        line = " ".join(x for x in [a["num"], a["street"], a["unit"]] if x)
        tail = " ".join(x for x in [a["state"], a["zip"]] if x)
        return ", ".join(x for x in [line, a["city"], tail] if x)


def make_split(g: Gen, n_s1: int, prefix: str):
    r = g.r
    ents = []
    # chains: several locations each (hard negatives for each other)
    n_chain = int(0.15 * n_s1)
    for i in range(n_chain):
        brand = CHAINS[i % len(CHAINS)]
        ents.append({"name": brand + (" " + r.choice(LEGAL) if r.random() < 0.2 else ""),
                     "addr": g.address(), "chain": True})
    while len(ents) < n_s1:
        ents.append({"name": g.name(), "addr": g.address(), "chain": False})
    # co-located businesses (same building, different business)
    for i in r.sample(range(n_chain, n_s1), k=int(0.05 * n_s1)):
        j = r.randrange(n_s1)
        ents[i]["addr"] = dict(ents[j]["addr"], unit=f"Suite {r.randint(100, 900)}")
    r.shuffle(ents)

    s1_ids = [f"S1{prefix}{i:07d}" for i in r.sample(range(10**7), n_s1)]
    rec2, rec3, gt = [], [], {i: [] for i in s1_ids}

    def emit(ent, vendor, owner=None):
        a = g.noisy_addr(ent["addr"], vendor)
        nm = g.noisy_name(ent["name"], vendor, ent["chain"])
        row = (nm, a)
        (rec2 if vendor == 2 else rec3).append((row, owner))

    for sid, ent in zip(s1_ids, ents):
        if r.random() < 0.35:
            continue                                              # singleton
        n2 = (1 + (r.random() < 0.25) + (r.random() < 0.05)) if r.random() < 0.75 else 0
        n3 = (1 + (r.random() < 0.2)) if r.random() < 0.6 else 0
        if n2 + n3 == 0:
            n2 = 1
        for _ in range(n2):
            emit(ent, 2, sid)
        for _ in range(n3):
            emit(ent, 3, sid)

    # distractors: businesses absent from S1 (chains at new addresses, near-duplicate names)
    n_dis = int(0.3 * n_s1)
    for _ in range(n_dis):
        u = r.random()
        if u < 0.35:
            ent = {"name": r.choice(CHAINS), "addr": g.address(), "chain": True}
        elif u < 0.65:
            base = r.choice(ents)
            ent = {"name": base["name"] + " " + r.choice(["Express", "Plus", "II", "Annex", "Outlet"]),
                   "addr": g.address() if r.random() < 0.6 else base["addr"], "chain": False}
        else:
            ent = {"name": g.name(), "addr": g.address(), "chain": False}
        emit(ent, 2 if r.random() < 0.5 else 3, None)

    r.shuffle(rec2)
    r.shuffle(rec3)
    s2_ids = [f"S2{prefix}{i:07d}" for i in r.sample(range(10**7), len(rec2))]
    s3_ids = [f"S3{prefix}{i:07d}" for i in r.sample(range(10**7), len(rec3))]
    for rid, (_, owner) in zip(s2_ids, rec2):
        if owner:
            gt[owner].append(rid)
    for rid, (_, owner) in zip(s3_ids, rec3):
        if owner:
            gt[owner].append(rid)

    s1 = pd.DataFrame({"source1_id": s1_ids, "name": [e["name"] for e in ents],
                       "address": [Gen.addr_str(e["addr"]) for e in ents]})
    s2 = pd.DataFrame({"source2_id": s2_ids, "business_name": [x[0][0] for x in rec2],
                       "street_address": [" ".join(v for v in [x[0][1]["num"], x[0][1]["street"], x[0][1]["unit"]] if v)
                                          for x in rec2],
                       "city": [x[0][1]["city"] for x in rec2], "state": [x[0][1]["state"] for x in rec2],
                       "postal_code": [x[0][1]["zip"] for x in rec2]})
    s3 = pd.DataFrame({"source3_id": s3_ids, "name": [x[0][0] for x in rec3],
                       "full_address": [Gen.addr_str(x[0][1]) for x in rec3]})
    gtdf = pd.DataFrame({"source1_id": s1_ids, "matches": [",".join(gt[i]) for i in s1_ids]})
    return s1, s2, s3, gtdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    g = Gen(a.seed)
    for split, n, pre in [("train", a.n_train, "A"), ("test", a.n_test, "B")]:
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        s1, s2, s3, gt = make_split(g, n, pre)
        s1.to_csv(os.path.join(d, "source1.tsv"), sep="\t", index=False)
        s2.to_csv(os.path.join(d, "source2.tsv"), sep="\t", index=False)
        s3.to_csv(os.path.join(d, "source3.tsv"), sep="\t", index=False)
        if split == "train":
            gt.to_csv(os.path.join(d, "train_ground_truth.tsv"), sep="\t", index=False)
        else:
            os.makedirs(os.path.join(a.out, "test_labels"), exist_ok=True)
            gt.to_csv(os.path.join(a.out, "test_labels", "test_ground_truth.tsv"), sep="\t", index=False)
        print(f"{split}: S1={len(s1)} S2={len(s2)} S3={len(s3)} singletons={(gt['matches'] == '').mean():.1%}")


if __name__ == "__main__":
    main()
