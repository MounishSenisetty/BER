"""Generate a synthetic dataset with the competition's exact schema, for local testing / timing.

    python scripts/make_synthetic_data.py --out data/dataset --n-train 20000 --n-test 15000

Creates
    <out>/train/train_source{1,2,3}.tsv  <out>/train/train_ground_truth.tsv
    <out>/test/test_source{1,2,3}.tsv
    <out>/test_labels/test_ground_truth.tsv          (hidden labels for local hold-out scoring)

Columns: entity_id (S1-/S2-/S3- prefix), business_name, business_address, country. Train has
US + India; test adds France (unseen country). Noise: typos, abbreviations, legal forms dropped /
changed / moved to the front, garbage prefixes, Devanagari-rendered Indian names, component
re-ordering, missing / null address parts, store numbers, chains, co-located businesses and
unmatched distractor records.
"""
from __future__ import annotations

import argparse
import os
import random
import string

import pandas as pd

ADJ = ["Golden", "Sunrise", "Blue", "Royal", "Green", "Silver", "Star", "Pacific", "Metro", "Prime", "Apex",
       "Summit", "Harbor", "Cedar", "Maple", "Liberty", "Eagle", "United", "First", "Premier", "Elite",
       "Rapid", "Bright", "Global", "Pioneer", "Heritage", "Crystal", "Diamond", "Horizon", "Nova"]
SURN = ["Johnson", "Smith", "Garcia", "Nguyen", "Brown", "Miller", "Lee", "Wilson", "Anderson", "Taylor",
        "Moore", "Martin", "Jackson", "White", "Harris", "Clark", "Lewis", "Walker", "Hall", "Young", "King"]
BTYPE_US = ["Bakery", "Dental Clinic", "Auto Repair", "Hardware", "Pharmacy", "Restaurant", "Consulting",
            "Services", "Technologies", "Logistics", "Motors", "Salon", "Cafe", "Grocery", "Electronics",
            "Law Office", "Associates", "Builders", "Printing", "Fitness Center", "Insurance", "Realty"]
LEGAL_US = ["Inc", "LLC", "Corp", "Co", "Incorporated", "Corporation", "Ltd"]
IN_WORDS = ["Ram", "Shri", "Ganesh", "Aditya", "Modern", "Suresh", "Kumar", "Krishna", "Lakshmi", "Balaji",
            "Sai", "Durga", "Om", "Mahalaxmi", "Gupta", "Sharma", "Singh", "Patel", "Verma", "Agarwal", "Jain"]
IN_TYPES = ["Traders", "Marketing", "Properties", "Finance", "Enterprises", "Industries", "Construction",
            "Infratech", "Solutions", "Agro Foods", "Textiles", "Motors", "Electricals", "Jewellers",
            "Medical Store", "Hotel"]
LEGAL_IN = ["Private Limited", "Pvt Ltd", "LLP", "Limited", ""]
DEVA = {"Ram": "राम", "Shri": "श्री", "Ganesh": "गणेश", "Traders": "ट्रेडर्स", "Marketing": "मार्केटिंग",
        "Private": "प्राइवेट", "Limited": "लिमिटेड", "Aditya": "आदित्य", "Properties": "प्रॉपर्टीज",
        "Modern": "मॉडर्न", "Finance": "फाइनेंस", "Suresh": "सुरेश", "Kumar": "कुमार", "Enterprises": "एंटरप्राइजेज",
        "Krishna": "कृष्णा", "Lakshmi": "लक्ष्मी", "Balaji": "बालाजी", "Sai": "साई", "Industries": "इंडस्ट्रीज",
        "Construction": "कंस्ट्रक्शन", "Infratech": "इंफ्राटेक", "Solutions": "सॉल्यूशंस", "Agro": "एग्रो",
        "Foods": "फूड्स", "Textiles": "टेक्सटाइल्स", "Motors": "मोटर्स", "Electricals": "इलेक्ट्रिकल्स",
        "Jewellers": "ज्वेलर्स", "Medical": "मेडिकल", "Store": "स्टोर", "Hotel": "होटल", "Gupta": "गुप्ता",
        "Sharma": "शर्मा", "Singh": "सिंह", "Patel": "पटेल", "Verma": "वर्मा", "Agarwal": "अग्रवाल",
        "Jain": "जैन", "Durga": "दुर्गा", "Om": "ओम", "Mahalaxmi": "महालक्ष्मी", "LLP": "एलएलपी"}
FR_WORDS = ["Atelier", "Boulangerie", "Cabinet", "Maison", "Garage", "Pharmacie", "Studio", "Agence", "Ecole",
            "Fractales", "Amis", "Soleil", "Lumiere", "Horizon", "Provence", "Bretagne", "Martin", "Bernard",
            "Dubois", "Moreau", "Laurent", "Lefebvre", "Petit", "Durand"]
LEGAL_FR = ["SARL", "SAS", "S.A.S", "SCI", "EURL", "SA", ""]
CHAINS = ["QuickMart", "Sunny Burger", "FreshCo Grocery", "Metro Pharmacy", "Speedy Lube", "ValueMart"]
US_CITIES = [("High Point", "NC", "272"), ("Tahlequah", "OK", "744"), ("Phoenix", "AZ", "850"),
             ("Dundalk", "MD", "212"), ("Morganton", "NC", "286"), ("Cleveland", "OH", "441"),
             ("Fort Worth", "TX", "761"), ("Cincinnati", "OH", "452"), ("Tyler", "TX", "757"),
             ("Iowa City", "IA", "522"), ("Salyersville", "KY", "414"), ("Roanoke", "VA", "240")]
US_STATE_FULL = {"NC": "North Carolina", "OK": "Oklahoma", "AZ": "Arizona", "MD": "Maryland", "OH": "Ohio",
                 "TX": "Texas", "IA": "Iowa", "KY": "Kentucky", "VA": "Virginia"}
US_STREETS = ["Westchester", "Ellis", "Montebello", "Cameron", "Elm", "Pierpont", "Fawn", "Ivanhoe",
              "Cotten", "Newton", "Kentucky", "Filly", "Stardust", "Mack", "Oak", "Main"]
US_TYPES = [("Drive", "Dr"), ("Road", "Rd"), ("Avenue", "Ave"), ("Street", "St"), ("Court", "Ct"),
            ("Trail", "Trl"), ("Lane", "Ln")]
IN_CITIES = [("Kolkata", "West Bengal", "700", "पश्चिम बंगाल"), ("New Delhi", "Delhi", "110", "दिल्ली"),
             ("Bhopal", "Madhya Pradesh", "462", "मध्य प्रदेश"), ("Mumbai", "Maharashtra", "400", "महाराष्ट्र"),
             ("Bengaluru", "Karnataka", "560", "ಕರ್ನಾಟಕ"), ("Gurugram", "Haryana", "122", "हरियाणा"),
             ("Jaipur", "Rajasthan", "302", "राजस्थान"), ("Lucknow", "Uttar Pradesh", "226", "उत्तर प्रदेश")]
IN_LOCALITY = ["Lake Town Block A", "Gulmohar Colony", "Udyog Vihar Phase V", "Jayanagar 9th Block",
               "Awas Vikas Colony", "Sector 18", "MG Road", "Civil Lines", "Industrial Area", "Model Town"]
FR_CITIES = [("Bordeaux", "Nouvelle-Aquitaine", "330"), ("Lille", "Hauts-de-France", "590"),
             ("Dunkerque", "Nord", "591"), ("La Teste-de-Buch", "Gironde", "332"), ("Lyon", "Rhone", "690")]
FR_STREETS = [("Rue", "R."), ("Boulevard", "Bd"), ("Avenue", "Av."), ("Place", "Pl."), ("Chemin", "Ch.")]
FR_NAMES = ["de Dieppe", "Pierre Dignac", "du President Roosevelt", "Jean Jaures", "Victor Hugo", "de la Paix"]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def typo(self, s):
        r = self.r
        if len(s) < 5:
            return s
        i = r.randrange(1, len(s) - 1)
        op = r.random()
        if op < 0.3:
            return s[:i] + s[i + 1:]
        if op < 0.6:
            return s[:i] + r.choice(string.ascii_lowercase) + s[i:]
        return s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]

    # --------------------------------------------------------------------- clean entities
    def entity(self, country):
        r = self.r
        if country == "US":
            base = r.choice([f"{r.choice(SURN)} {r.choice(BTYPE_US)}", f"{r.choice(ADJ)} {r.choice(BTYPE_US)}",
                             f"{r.choice(ADJ)} {r.choice(SURN)} {r.choice(BTYPE_US)}"])
            legal = r.choice(LEGAL_US) if r.random() < 0.5 else ""
            c = r.choice(US_CITIES)
            st, _ = r.choice(US_TYPES)
            addr = {"num": str(r.randint(1, 19999)), "street": f"{r.choice(US_STREETS)} {st}",
                    "unit": f"Unit {r.choice('ABCDEFG')}" if r.random() < 0.1 else "",
                    "loc": "", "city": c[0], "state": c[1], "zip": c[2] + f"{r.randint(0, 99):02d}"}
        elif country == "India":
            base = " ".join(r.sample(IN_WORDS, r.choice([1, 2]))) + " " + r.choice(IN_TYPES)
            legal = r.choice(LEGAL_IN)
            c = r.choice(IN_CITIES)
            addr = {"num": r.choice([f"{r.randint(1, 999)}", f"G-{r.randint(1, 9)}/{r.randint(1, 999)}",
                                     f"H.No {r.randint(1, 999)}"]),
                    "street": "", "unit": "", "loc": r.choice(IN_LOCALITY), "city": c[0], "state": c[1],
                    "zip": c[2] + f"{r.randint(0, 999):03d}", "state_native": c[3]}
        else:  # France
            base = f"{r.choice(FR_WORDS)} {r.choice(FR_WORDS)}"
            legal = r.choice(LEGAL_FR)
            c = r.choice(FR_CITIES)
            st, _ = r.choice(FR_STREETS)
            addr = {"num": str(r.randint(1, 250)) + (" bis" if r.random() < 0.05 else ""),
                    "street": f"{st} {r.choice(FR_NAMES)}", "unit": "", "loc": "", "city": c[0],
                    "state": c[1], "zip": c[2] + f"{r.randint(0, 99):02d}"}
        return {"name": (base + " " + legal).strip(), "base": base, "legal": legal, "addr": addr,
                "country": country, "chain": False}

    @staticmethod
    def addr_str(a, order=0):
        line = " ".join(x for x in [a["num"], a["street"]] if x)
        parts = [line, a.get("unit", ""), a.get("loc", ""), a["city"], " ".join(x for x in [a["state"], a["zip"]] if x)]
        parts = [p for p in parts if p]
        if order == 1 and len(parts) > 2:              # "IA, Iowa City, 1064 Newton Rd"
            parts = parts[::-1]
        return ", ".join(parts)

    # --------------------------------------------------------------------- noisy copies
    def noisy(self, e, vendor):
        r = self.r
        base, legal = e["base"], e["legal"]
        # legal form: drop / swap position / change
        u = r.random()
        if legal and u < 0.35:
            legal = ""
        elif legal and u < 0.5:
            name = f"{legal} {base}"
            legal = None
        if legal is not None:
            if legal == "Private Limited" and r.random() < 0.5:
                legal = r.choice(["Pvt. Ltd.", "Private (Limited)", "Pvt Ltd"])
            elif legal == "Inc" and r.random() < 0.3:
                legal = "Inc."
            name = f"{base} {legal}".strip()
        u = r.random()
        if u < 0.07:                                  # unrelated trade / brand name (only the address links it)
            syl = ["zeta", "lyra", "novi", "quo", "zeph", "xylo", "lum", "kelo", "riza", "halo", "pyra", "avi", "ecto"]
            name = "".join(r.sample(syl, r.choice([2, 3]))).capitalize()
        elif u < 0.11:                                # acronym of the core name ("WEC", "MC")
            name = "".join(w[0] for w in base.split()).upper()
        elif u < 0.36:
            name = self.typo(name)
        if e["chain"] and r.random() < 0.4:
            name += f" #{r.randint(1, 2999):04d}"
        if r.random() < 0.05:
            name = r.choice(["-- ", "<< ", "** "]) + name
        if e["country"] == "India" and r.random() < 0.3:
            name = " ".join(DEVA.get(w, w) for w in name.replace(".", "").split())
        u = r.random()
        if u < 0.3:
            name = name.upper()
        elif u < 0.4:
            name = name.lower()

        a = dict(e["addr"])
        if r.random() < (0.03 if vendor == 2 else 0.04):
            return name, ""
        if e["country"] == "US":
            st = a["street"]
            for full, ab in US_TYPES:
                if r.random() < 0.6:
                    st = st.replace(full, ab) if full in st else st.replace(" " + ab, " " + full)
            a["street"] = self.typo(st) if r.random() < 0.1 else st
            if r.random() < 0.3:
                a["state"] = US_STATE_FULL.get(a["state"], a["state"])
        elif e["country"] == "France":
            for full, ab in FR_STREETS:
                if r.random() < 0.5 and a["street"].startswith(full):
                    a["street"] = ab + a["street"][len(full):]
        else:
            if r.random() < 0.3:
                a["state"] = a.pop("state_native", a["state"])
            if r.random() < 0.3:
                a["city"] = {"Bengaluru": "Bangalore", "Gurugram": "Gurgaon", "Mumbai": "Bombay"}.get(a["city"], a["city"])
            if r.random() < 0.2:
                a["loc"] = "Near SBI ATM, " + a["loc"]
        if r.random() < 0.2:
            a["zip"] = ""
        if r.random() < 0.1:
            a["state"] = "null"
        if r.random() < 0.15:
            a["city"] = ""
        s = self.addr_str(a, order=1 if r.random() < 0.15 else 0)
        return (s.upper() if r.random() < 0.3 else s, name)[::-1]


def make_split(g: Gen, n_s1: int, countries):
    r = g.r
    ents = []
    for _ in range(n_s1):
        c = r.choices(countries, weights=[0.55, 0.35, 0.10][:len(countries)])[0]
        e = g.entity(c)
        if c == "US" and r.random() < 0.08:
            e["base"], e["legal"], e["chain"] = r.choice(CHAINS), "", True
            e["name"] = e["base"]
        ents.append(e)
    s1_ids = [f"S1-{i}" for i in r.sample(range(10**8, 10**9), n_s1)]
    recs, gt = [], {i: [] for i in s1_ids}
    for sid, e in zip(s1_ids, ents):
        if r.random() < 0.056:                            # real data: 5.6% singletons, ~3.5 matches
            continue
        for vendor in (2, 3):
            for _ in range(r.choice([0, 1, 1, 2, 2, 2, 3])):
                recs.append((vendor, e, sid))
    for _ in range(int(1.2 * n_s1)):                      # distractors not in Source 1
        c = r.choices(countries, weights=[0.55, 0.35, 0.10][:len(countries)])[0]
        e = g.entity(c)
        if r.random() < 0.3:                              # near-duplicate of an existing entity
            b = r.choice(ents)
            e = dict(b, base=b["base"] + " " + r.choice(["Express", "Plus", "Annex", "Outlet", "II"]),
                     addr=g.entity(b["country"])["addr"] if r.random() < 0.6 else b["addr"])
        recs.append((r.choice([2, 3]), e, None))
    r.shuffle(recs)
    rows = {2: [], 3: []}
    for vendor, e, owner in recs:
        name, addr = g.noisy(e, vendor)
        rid = f"S{vendor}-{r.randrange(10**7, 10**9)}"
        rows[vendor].append((rid, name, addr, e["country"]))
        if owner:
            gt[owner].append(rid)
    s1 = pd.DataFrame({"entity_id": s1_ids, "business_name": [e["name"] for e in ents],
                       "business_address": [Gen.addr_str(e["addr"], order=int(r.random() < 0.1)) for e in ents],
                       "country": [e["country"] for e in ents]})
    cols = ["entity_id", "business_name", "business_address", "country"]
    s2 = pd.DataFrame(rows[2], columns=cols).drop_duplicates("entity_id")
    s3 = pd.DataFrame(rows[3], columns=cols).drop_duplicates("entity_id")
    valid = set(s2["entity_id"]) | set(s3["entity_id"])
    gtdf = pd.DataFrame({"source1_entity_id": s1_ids,
                         "matched_entity_ids": [",".join(dict.fromkeys(m for m in gt[i] if m in valid)) for i in s1_ids]})
    return s1, s2, s3, gtdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    g = Gen(a.seed)
    for split, n, countries in [("train", a.n_train, ["US", "India"]), ("test", a.n_test, ["US", "India", "France"])]:
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        s1, s2, s3, gt = make_split(g, n, countries)
        for k, df in [(1, s1), (2, s2), (3, s3)]:
            df.to_csv(os.path.join(d, f"{split}_source{k}.tsv"), sep="\t", index=False)
        gdir = d if split == "train" else os.path.join(a.out, "test_labels")
        os.makedirs(gdir, exist_ok=True)
        gt.to_csv(os.path.join(gdir, f"{split}_ground_truth.tsv"), sep="\t", index=False)
        print(f"{split}: S1={len(s1)} S2={len(s2)} S3={len(s3)} singletons={(gt['matched_entity_ids'] == '').mean():.1%}")


if __name__ == "__main__":
    main()
