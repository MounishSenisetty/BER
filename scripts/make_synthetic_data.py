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


# --hard mode: English words written phonetically in an Indic script ("Software" -> সফটওয়্যার),
# the way the real data renders Indian business names; the rule transliterator cannot undo this
_SCRIPTS = [0x0900, 0x0980, 0x0C80, 0x0C00, 0x0D00]         # Devanagari, Bengali, Kannada, Telugu, Malayalam
_C_OFF = {"kh": 0x16, "gh": 0x18, "ch": 0x1A, "th": 0x25, "dh": 0x27, "ph": 0x2B, "bh": 0x2D, "sh": 0x36,
          "k": 0x15, "g": 0x17, "j": 0x1C, "t": 0x1F, "d": 0x21, "n": 0x28, "p": 0x2A, "b": 0x2C, "m": 0x2E,
          "y": 0x2F, "r": 0x30, "l": 0x32, "v": 0x35, "s": 0x38, "h": 0x39}
_V_IND = {"aa": 0x06, "ii": 0x08, "uu": 0x0A, "ai": 0x10, "au": 0x14, "a": 0x05, "i": 0x07, "u": 0x09,
          "e": 0x0F, "o": 0x13}
_V_MAT = {"aa": 0x3E, "ii": 0x40, "uu": 0x42, "ai": 0x48, "au": 0x4C, "i": 0x3F, "u": 0x41, "e": 0x47, "o": 0x4B}


def _respell(w: str) -> str:
    """Crude English -> Indian phonetic spelling (software -> saftaveyar, service -> sarvis)."""
    w = w.lower()
    for a, b in (("ware", "veyar"), ("tion", "shan"), ("ck", "k"), ("ph", "f"), ("qu", "kv"), ("x", "ks"),
                 ("ee", "ii"), ("oo", "u"), ("ou", "au"), ("w", "v"), ("q", "k"), ("z", "j"), ("f", "ph")):
        w = w.replace(a, b)
    w = "".join("s" if c == "c" and w[i + 1:i + 2] in ("e", "i", "y") else ("k" if c == "c" else c)
                for i, c in enumerate(w))
    if w.endswith("er"):
        w = w[:-2] + "ar"
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]
    return w.replace("y", "i") if not w.startswith("y") else w


def to_script(word: str, block: int) -> str:
    w = _respell(word)
    out, i, prev_cons = [], 0, False
    while i < len(w):
        v = next((x for x in ("aa", "ii", "uu", "ai", "au", "a", "i", "u", "e", "o") if w.startswith(x, i)), None)
        if v:
            if prev_cons:
                if v != "a":
                    out.append(chr(block + _V_MAT[v]))
            else:
                out.append(chr(block + _V_IND[v]))
            prev_cons = False
            i += len(v)
            continue
        c = next((x for x in _C_OFF if w.startswith(x, i)), None)
        if c is None:
            i += 1
            continue
        if prev_cons:
            out.append(chr(block + 0x4D))               # virama: conjunct
        out.append(chr(block + _C_OFF[c]))
        prev_cons = True
        i += len(c)
    return "".join(out)


class Gen:
    def __init__(self, seed, hard=False):
        self.r = random.Random(seed)
        self.hard = hard

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

    def hard_name(self, name, e):
        """Real-data name noise: domains / handles, dropped or appended words, word swaps,
        several typos, phone numbers."""
        r = self.r
        u = r.random()
        words = name.split()
        if u < 0.04:
            return "".join(w.lower() for w in e["base"].split()) + ".com"
        if u < 0.05:
            return "@" + words[0].lower() + str(r.randint(1, 99))
        if u < 0.10:
            return name + " " + r.choice(["Services", "Service", "Center", "Group", "Partners", "Co"])
        if u < 0.15 and len(words) > 2:
            del words[r.randrange(len(words))]
            return " ".join(words)
        if u < 0.19 and len(words) > 1:
            i = r.randrange(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
            return " ".join(words)
        if u < 0.24:
            for _ in range(r.choice([2, 3])):
                name = self.typo(name)
            return name
        if u < 0.26:
            return name + " - " + str(r.randint(6 * 10**9, 10**10 - 1))
        return name

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
        if self.hard:
            name = self.hard_name(name, e)
        if e["country"] == "India" and r.random() < 0.3:
            if self.hard and r.random() < 0.6:
                blk = r.choice(_SCRIPTS)
                name = " ".join(to_script(w, blk) for w in name.replace(".", "").split())
            else:
                name = " ".join(DEVA.get(w, w) for w in name.replace(".", "").split())
        u = r.random()
        if u < 0.3:
            name = name.upper()
        elif u < 0.4:
            name = name.lower()

        a = dict(e["addr"])
        if r.random() < ((0.03 if vendor == 2 else 0.04) * (3 if self.hard else 1)):
            return name, ""
        if self.hard and r.random() < 0.15:          # house number mistyped / replaced
            n = a["num"]
            d = [i for i, ch in enumerate(n) if ch.isdigit()]
            if d and r.random() < 0.6:
                i = r.choice(d)
                a["num"] = n[:i] + str(r.randint(0, 9)) + n[i + 1:]
            else:
                a["num"] = str(r.randint(1, 9999))
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


def make_split(g: Gen, n_s1: int, countries, keep_frac: float = 1.0):
    """keep_frac < 1 (--hard): generate n_s1 / keep_frac entities with records, keep n_s1 of them in
    Source 1; the records of the others become distractors (how the real splits look)."""
    r = g.r
    ents = []
    n_univ = int(round(n_s1 / keep_frac))
    for _ in range(n_univ):
        c = r.choices(countries, weights=[0.55, 0.35, 0.10][:len(countries)])[0]
        e = g.entity(c)
        if c == "US" and r.random() < 0.08:
            e["base"], e["legal"], e["chain"] = r.choice(CHAINS), "", True
            e["name"] = e["base"]
        ents.append(e)
    all_ids = [f"S1-{i}" for i in r.sample(range(10**8, 10**9), n_univ)]
    s1_ids = all_ids[:n_s1]
    recs, gt = [], {i: [] for i in s1_ids}
    for k, (sid, e) in enumerate(zip(all_ids, ents)):
        if k >= n_s1:                                     # entity not in Source 1: its records are distractors
            for vendor in (2, 3):
                for _ in range(r.choice([0, 1, 1, 2, 2, 2, 3])):
                    recs.append((vendor, e, None))
            continue
        if r.random() < 0.056:                            # real data: 5.6% singletons, ~3.5 matches
            continue
        for vendor in (2, 3):
            for _ in range(r.choice([0, 1, 1, 2, 2, 2, 3])):
                recs.append((vendor, e, sid))
    ents = ents[:n_s1]
    for _ in range(int((1.2 if keep_frac >= 1 else 0.3) * n_s1)):     # distractors not in Source 1
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
    ap.add_argument("--hard", action="store_true",
                    help="real-data noise: Indic-script English loanwords, domains, word drops/swaps, house-number "
                         "typos, more empty addresses, and test Source 1 thinned so test has more distractors")
    a = ap.parse_args()
    g = Gen(a.seed, hard=a.hard)
    for split, n, countries in [("train", a.n_train, ["US", "India"]), ("test", a.n_test, ["US", "India", "France"])]:
        keep = (0.74 if split == "train" else 0.60) if a.hard else 1.0
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        s1, s2, s3, gt = make_split(g, n, countries, keep)
        for k, df in [(1, s1), (2, s2), (3, s3)]:
            df.to_csv(os.path.join(d, f"{split}_source{k}.tsv"), sep="\t", index=False)
        gdir = d if split == "train" else os.path.join(a.out, "test_labels")
        os.makedirs(gdir, exist_ok=True)
        gt.to_csv(os.path.join(gdir, f"{split}_ground_truth.tsv"), sep="\t", index=False)
        print(f"{split}: S1={len(s1)} S2={len(s2)} S3={len(s3)} singletons={(gt['matched_entity_ids'] == '').mean():.1%}")


if __name__ == "__main__":
    main()
