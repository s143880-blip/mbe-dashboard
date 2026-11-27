"""
mbe_core.py - Havlena-Odeh material balance calculations for a gas-cap oil reservoir.

No Streamlit code here, so every function can be tested on its own (see tests/).

Internal units:  p psia | Np STB | Gp SCF | Bo bbl/STB | Bg bbl/SCF | Rs SCF/STB | c 1/psi
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field

import numpy as np

FT3_PER_BBL = 5.614583
BBL_PER_FT3 = 1.0 / FT3_PER_BBL

# --------------------------------------------------------------------------- units
UNIT_OPTIONS: dict[str, list[tuple[str, float]]] = {
    "p": [("psia", 1), ("psig", 1), ("bar", 14.50377), ("kPa", 0.1450377), ("MPa", 145.0377)],
    "Np": [("STB", 1), ("MSTB", 1e3), ("MMSTB", 1e6)],
    "Gp": [("SCF", 1), ("MSCF", 1e3), ("MMSCF", 1e6), ("MMMSCF (Bscf)", 1e9)],
    "Rp": [("SCF/STB", 1), ("MSCF/STB", 1e3)],
    "Bo": [("bbl/STB", 1)],
    "Bg": [("bbl/SCF", 1), ("ft3/SCF", BBL_PER_FT3), ("bbl/MSCF", 1e-3), ("ft3/MSCF", BBL_PER_FT3 * 1e-3)],
    "Rs": [("SCF/STB", 1), ("MSCF/STB", 1e3)],
    "Wp": [("STB", 1), ("MSTB", 1e3), ("MMSTB", 1e6)],
    "Bw": [("bbl/STB", 1)],
    "We": [("bbl", 1), ("Mbbl", 1e3), ("MMbbl", 1e6)],
    "c": [("1/psi", 1), ("1e-6 1/psi (microsips)", 1e-6)],
}
UNIT_OPTIONS.update({"pi": UNIT_OPTIONS["p"], "Boi": UNIT_OPTIONS["Bo"], "Bgi": UNIT_OPTIONS["Bg"],
                     "Rsi": UNIT_OPTIONS["Rs"], "cw": UNIT_OPTIONS["c"], "cf": UNIT_OPTIONS["c"]})


def unit_labels(kind: str) -> list[str]:
    return [u for u, _ in UNIT_OPTIONS.get(kind, [("fraction", 1)])]


def unit_factor(kind: str, label: str) -> float:
    for u, f in UNIT_OPTIONS.get(kind, []):
        if u == label:
            return f
    return 1.0


def detect_unit(kind: str, text: str) -> str | None:
    """Recognise a unit written in a header or next to a value. Returns an option label or None."""
    t = str(text or "").lower().replace(" ", "")
    t = t.replace("³", "3")
    t = re.sub(r"cu\.?ft|cubicfeet|cubicfoot|ft\^3", "ft3", t)
    t = re.sub(r"[−–—]", "-", t)
    t = re.sub(r"⁻¹|\^-1|-1|1/", "inv", t)
    if not t:
        return None
    k = {"pi": "p", "Boi": "Bo", "Bgi": "Bg", "Rsi": "Rs", "cw": "c", "cf": "c"}.get(kind, kind)
    if k == "p":
        if "psig" in t: return "psig"
        if "psi" in t: return "psia"
        if "mpa" in t: return "MPa"
        if "kpa" in t: return "kPa"
        if "bar" in t: return "bar"
    elif k in ("Np", "Wp"):
        if re.search(r"mmstb|mmbbl|mmbo", t): return "MMSTB"
        if re.search(r"mstb|mbbl|mbo", t): return "MSTB"
        if re.search(r"stb|bbl", t): return "STB"
    elif k == "We":
        if re.search(r"mmbbl|mmrb", t): return "MMbbl"
        if re.search(r"mbbl|mrb", t): return "Mbbl"
        if re.search(r"bbl|rb", t): return "bbl"
    elif k == "Gp":
        if re.search(r"mmmscf|bscf|bcf|mmmcf", t): return "MMMSCF (Bscf)"
        if re.search(r"mmscf|mmcf", t): return "MMSCF"
        if re.search(r"mscf|mcf", t): return "MSCF"
        if re.search(r"scf|cf|ft3", t): return "SCF"
    elif k == "Bg":
        if re.search(r"ft3/m(scf|cf)", t): return "ft3/MSCF"
        if re.search(r"(bbl|rb)/m(scf|cf)", t): return "bbl/MSCF"
        if "ft3" in t: return "ft3/SCF"
        if re.search(r"bbl|rb", t): return "bbl/SCF"
    elif k in ("Rs", "Rp"):
        if re.search(r"mscf/stb|mcf/stb|mscf/bbl", t): return "MSCF/STB"
        if re.search(r"scf|cf", t): return "SCF/STB"
    elif k in ("Bo", "Bw"):
        if re.search(r"bbl|rb", t): return "bbl/STB"
    elif k == "c":
        if re.search(r"micro|μ|µ", t): return "1e-6 1/psi (microsips)"
        if re.search(r"psi|inv", t): return "1/psi"
    return None


# --------------------------------------------------------------------------- text helpers
def norm_name(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str("" if s is None else s).lower())


def split_header(h) -> tuple[str, str]:
    """'Gp (MMMSCF)' -> ('Gp', 'MMMSCF');  'Gp, MMscf' -> ('Gp', 'MMscf')"""
    s = str("" if h is None else h).strip()
    m = re.match(r"^(.*?)\s*[\(\[]([^\)\]]*)[\)\]]", s)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"^([^,]+),\s*(.+)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return s, ""


_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)(\s?[eE]\s?[-+]?\d+)?")


def parse_num(v) -> float:
    """Number from a cell: 2740, '2740 psia', '4.00E–06 psi-1', '1,234.5'. NaN if not a number."""
    if isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) if math.isfinite(float(v)) else math.nan
    if v is None:
        return math.nan
    s = re.sub(r"[−–—]", "-", str(v).strip()).replace(",", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[x×]\s*10\s*\^?\s*([-+]?\d+)", r"e\1", s, flags=re.I)
    m = _NUM_RE.match(s)
    if not m:
        return math.nan
    rest = s[m.end():].strip()
    if rest[:1].isdigit() or rest[:1] == ".":
        return math.nan
    return float(m.group(0).replace(" ", ""))


def unit_text_of_cell(v) -> str:
    if not isinstance(v, str):
        return ""
    s = re.sub(r"[−–—]", "-", v.strip())
    m = _NUM_RE.match(s)
    return s[m.end():].strip() if m else ""


def is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == ""


def col_letter(i: int) -> str:
    s, i = "", i + 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


# --------------------------------------------------------------------------- names
COL_ALIASES = {
    "p": ["p", "pressure", "pr", "pavg", "pbar", "averagepressure", "reservoirpressure",
          "averagereservoirpressure", "avgpressure", "pres"],
    "Np": ["np", "cumulativeoil", "cumulativeoilproduced", "cumulativeoilproduction", "cumoil", "oilproduced"],
    "Gp": ["gp", "cumulativegas", "cumulativegasproduced", "cumulativegasproduction", "cumgas", "gasproduced"],
    "Rp": ["rp", "cumulativegor", "cumulativegasoilratio", "producinggor"],
    "Bo": ["bo", "oilfvf", "oilformationvolumefactor"],
    "Bg": ["bg", "gasfvf", "gasformationvolumefactor"],
    "Rs": ["rs", "solutiongor", "solutiongasoilratio", "gasoilratio"],
    "Wp": ["wp", "cumulativewater", "cumulativewaterproduced", "waterproduced"],
    "Bw": ["bw", "waterfvf"],
    "We": ["we", "waterinflux", "cumulativewaterinflux"],
}
PARAM_ALIASES = {
    "pi": ["pi", "pini", "pinitial", "initialpressure", "initialreservoirpressure", "originalpressure"],
    "Boi": ["boi", "initialbo", "initialoilfvf", "initialoilformationvolumefactor"],
    "Bgi": ["bgi", "initialbg", "initialgasfvf", "initialgasformationvolumefactor"],
    "Rsi": ["rsi", "initialrs", "initialsolutiongor", "initialsolutiongasoilratio"],
    "Swc": ["swc", "swi", "connatewatersaturation", "initialwatersaturation", "irreduciblewatersaturation"],
    "cw": ["cw", "watercompressibility"],
    "cf": ["cf", "formationcompressibility", "rockcompressibility", "porecompressibility",
           "porevolumecompressibility"],
    "m": ["m", "gascapratio", "gascaptooilratio", "mratio"],
}
REQUIRED_COLS = ["p", "Np", "Bo", "Bg", "Rs"]


def match_key(text, aliases: dict) -> str | None:
    """Match a header/label to a key. Also accepts 'Initial pressure pi' (short symbol as last word)."""
    if not isinstance(text, str):
        return None
    name = split_header(text)[0]
    n = norm_name(name)
    if not n:
        return None
    for k, al in aliases.items():
        if n in al:
            return k
    words = [w for w in re.split(r"[\s,:=\-]+", name.strip()) if w]
    if len(words) > 1:
        last = norm_name(words[-1])
        if len(last) <= 4:
            for k, al in aliases.items():
                if last in al:
                    return k
    return None


# --------------------------------------------------------------------------- reading files
Sheet = tuple  # (name, rows: list[list])


def read_workbook(data: bytes, filename: str) -> list[Sheet]:
    """Read .xlsx/.xlsm or .csv bytes into [(sheet_name, rows)]."""
    name = filename.lower()
    if name.endswith(".csv"):
        text = data.decode("utf-8-sig", errors="replace")
        first = text.splitlines()[0] if text else ""
        delim = ";" if first.count(";") > first.count(",") else ","
        rows = []
        for r in csv.reader(io.StringIO(text), delimiter=delim):
            rows.append([_csv_cell(c) for c in r])
        return [("CSV", rows)]
    if name.endswith(".xls"):
        raise ValueError("Old .xls format is not supported. Open the file in Excel and use "
                         "File > Save As > Excel Workbook (.xlsx).")
    import openpyxl
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError("This is not a valid .xlsx file - " + str(e)) from e
    sheets = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        sheets.append((ws.title, rows))
    wb.close()
    if not sheets:
        raise ValueError("The workbook has no readable sheets.")
    return sheets


def _csv_cell(c: str):
    t = c.strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return t


# --------------------------------------------------------------------------- finding the data
@dataclass
class Table:
    sheet: str
    header_row: int
    cols: dict        # key -> column index
    units: dict       # key -> unit text from header


def find_production_table(sheets: list[Sheet]) -> Table | None:
    for sname, rows in sheets:
        for r, row in enumerate(rows):
            cols, units = {}, {}
            for c, cell in enumerate(row or []):
                k = match_key(cell, COL_ALIASES)
                if k and k not in cols:
                    cols[k] = c
                    units[k] = split_header(cell)[1]
            if all(k in cols for k in REQUIRED_COLS) and ("Gp" in cols or "Rp" in cols):
                return Table(sname, r, cols, units)
    return None


def diagnose_missing(sheets: list[Sheet]) -> tuple[str | None, list[str], list[str]]:
    best_found, best_sheet = set(), None
    for sname, rows in sheets:
        for row in rows:
            found = {match_key(c, COL_ALIASES) for c in (row or [])} - {None}
            if len(found) > len(best_found):
                best_found, best_sheet = found, sname
    need = REQUIRED_COLS + ["Gp"]
    missing = [k for k in need if k not in best_found and not (k == "Gp" and "Rp" in best_found)]
    return best_sheet, sorted(best_found), missing


def find_parameters(sheets: list[Sheet], table: Table | None) -> dict:
    """Find 'label | value | unit' pairs anywhere in the workbook."""
    out = {}
    for sname, rows in sheets:
        for r, row in enumerate(rows):
            if table and sname == table.sheet and r == table.header_row:
                continue
            row = list(row or [])
            for c, cell in enumerate(row):
                if not isinstance(cell, str):
                    continue
                k = match_key(cell, PARAM_ALIASES)
                if not k or k in out:
                    continue
                for j in range(c + 1, min(c + 4, len(row))):
                    v = parse_num(row[j])
                    if not math.isnan(v):
                        unit_text = split_header(cell)[1] + " " + unit_text_of_cell(row[j])
                        for u in range(j + 1, min(j + 3, len(row))):
                            if isinstance(row[u], str) and math.isnan(parse_num(row[u])):
                                unit_text += " " + row[u]
                        out[k] = {"raw": v, "unit_text": unit_text.strip(),
                                  "where": f"{sname}!{col_letter(j)}{r + 1}"}
                        break
    return out


@dataclass
class Dataset:
    table: Table
    col_units: dict           # key -> {label, detected, header_text, factor, assumed}
    rows: list                # list of dicts in internal units, plus 'excel_row'
    skipped: list
    found: dict               # parameters found in the file
    messages: list = field(default_factory=list)


def msg(level: str, text: str) -> dict:
    return {"level": level, "text": text}


def extract_dataset(sheets: list[Sheet], unit_overrides: dict | None = None) -> Dataset:
    unit_overrides = unit_overrides or {}
    messages = []
    table = find_production_table(sheets)
    if table is None:
        sheet, found, missing = diagnose_missing(sheets)
        names = {"p": "pressure p", "Np": "Np", "Gp": "Gp (or Rp)", "Bo": "Bo", "Bg": "Bg", "Rs": "Rs"}
        if found:
            detail = (f'Best match was on sheet "{sheet}", but these columns are missing: '
                      + ", ".join(names[k] for k in missing) + ".")
        else:
            detail = "No recognisable column headers (p, Np, Gp, Bo, Bg, Rs) were found in any sheet."
        raise ValueError("Could not find the production/PVT table. " + detail +
                         ' Each column needs a header cell such as "Np (MMSTB)" in one row.')

    col_units = {}
    for k in table.cols:
        detected = detect_unit(k, table.units[k])
        chosen = unit_overrides.get(k) or detected or unit_labels(k)[0]
        assumed = not unit_overrides.get(k) and not detected
        col_units[k] = {"label": chosen, "detected": detected, "header_text": table.units[k],
                        "factor": unit_factor(k, chosen), "assumed": assumed}
        if assumed and k not in ("Bo", "Bw"):
            messages.append(msg("warn", f'No unit found in the "{k}" header - assumed {chosen}. '
                                         "Change it under Column units if this is wrong."))

    rows_all = dict(sheets)[table.sheet]
    rows = []
    for r in range(table.header_row + 1, len(rows_all)):
        row = list(rows_all[r] or [])
        cells = {k: (row[c] if c < len(row) else None) for k, c in table.cols.items()}
        if all(is_blank(v) for v in cells.values()):
            if rows:
                break
            continue
        rec = {"excel_row": r + 1, "bad": []}
        for k, v in cells.items():
            if is_blank(v):
                if k not in ("Wp", "Bw", "We", "Rp", "Gp"):
                    rec["bad"].append(f"{k} is empty")
                continue
            x = parse_num(v)
            if math.isnan(x):
                rec["bad"].append(f'{k} = "{v}" is not a number')
                continue
            rec[k] = x * col_units[k]["factor"]
        if rows and len(rec["bad"]) >= len(table.cols) - 1:
            break           # a text row after the table (e.g. "Initial parameters") ends it
        rows.append(rec)

    for rec in rows:
        if "Gp" not in rec and "Rp" in rec and "Np" in rec:
            rec["Gp"] = rec["Rp"] * rec["Np"]
        if "Gp" not in rec and rec.get("Np") == 0:
            rec["Gp"] = 0.0                     # nothing produced yet -> no gas produced
        if "Gp" not in rec and not rec["bad"]:
            rec["bad"].append("Gp is empty")
        rec.setdefault("Wp", 0.0)
        rec.setdefault("Bw", 1.0)
        rec.setdefault("We", 0.0)
        if not rec["bad"]:
            if rec["p"] <= 0: rec["bad"].append("pressure must be > 0")
            if rec["Np"] < 0 or rec["Gp"] < 0: rec["bad"].append("Np and Gp cannot be negative")
            if rec["Bo"] <= 0 or rec["Bg"] <= 0: rec["bad"].append("Bo and Bg must be > 0")
            if rec["Rs"] < 0: rec["bad"].append("Rs cannot be negative")
    skipped = [r for r in rows if r["bad"]]
    for r in skipped:
        messages.append(msg("warn", f"Excel row {r['excel_row']} skipped: {'; '.join(r['bad'])}."))
    good = [r for r in rows if not r["bad"]]
    if not good:
        raise ValueError(f'The table on sheet "{table.sheet}" has headers but no valid numeric rows under them.')
    return Dataset(table, col_units, good, skipped, find_parameters(sheets, table), messages)


# --------------------------------------------------------------------------- initial parameters
PARAM_DEFAULT_UNIT = {"pi": "psia", "Boi": "bbl/STB", "Swc": "fraction", "cw": "1/psi", "cf": "1/psi",
                      "m": "fraction"}


def resolve_params(ds: Dataset, edits: dict | None = None, edit_units: dict | None = None):
    """Priority: user edit > value in file > production table row at p = pi.
    Returns (P, src, units, messages); P is in internal units."""
    edits, edit_units = edits or {}, edit_units or {}
    P, src, units, msgs = {}, {}, {}, []
    default_unit = dict(PARAM_DEFAULT_UNIT, Bgi=ds.col_units["Bg"]["label"], Rsi=ds.col_units["Rs"]["label"])

    def take(k):
        if k in edits and edits[k] not in (None, "") and not math.isnan(parse_num(edits[k])):
            lbl = edit_units.get(k) or default_unit[k]
            P[k], units[k], src[k] = parse_num(edits[k]) * unit_factor(k, lbl), lbl, "entered by user"
            return True
        f = ds.found.get(k)
        if f:
            lbl = edit_units.get(k) or detect_unit(k, f["unit_text"]) or default_unit[k]
            P[k], units[k], src[k] = f["raw"] * unit_factor(k, lbl), lbl, "file " + f["where"]
            return True
        return False

    rows = ds.rows
    init_row = next((r for r in rows if r["Np"] == 0), None)
    if not take("pi"):
        units["pi"] = "psia"
        if init_row:
            P["pi"], src["pi"] = init_row["p"], f"production table (row with Np = 0, Excel row {init_row['excel_row']})"
        else:
            P["pi"], src["pi"] = max(r["p"] for r in rows), "highest pressure in table (no Np = 0 row)"
            msgs.append(msg("warn", "Initial pressure not given and no Np = 0 row found - "
                                    "using the highest pressure in the table."))
    at_pi = next((r for r in rows if abs(r["p"] - P["pi"]) < 0.5), None)
    for k, col in (("Boi", "Bo"), ("Bgi", "Bg"), ("Rsi", "Rs")):
        if take(k):
            continue
        row = at_pi or init_row
        if row:
            P[k], units[k], src[k] = row[col], default_unit[k], f"production table (p = {row['p']:.0f} psia row)"
        else:
            msgs.append(msg("error", f"{k} is missing: give it in the parameters list or include a row at p = pi."))
    for k in ("Swc", "cw", "cf"):
        if not take(k):
            P[k], units[k], src[k] = 0.0, default_unit[k], "not given - set to 0"
    if take("m"):
        units["m"] = "fraction"

    if 1 < P["Swc"] <= 100:
        P["Swc"] /= 100
        msgs.append(msg("info", "Swc was larger than 1, so it was read as a percentage."))
    if not (0 <= P["Swc"] < 1):
        msgs.append(msg("error", "Swc must be between 0 and 1."))
    if any(src[k].startswith("not") for k in ("Swc", "cw", "cf")):
        msgs.append(msg("info", "Swc, cw or cf not given - set to 0 (the Ef,w term is ignored for missing values)."))
    if P["cw"] > 1e-3 or P["cf"] > 1e-3:
        msgs.append(msg("warn", "cw or cf looks too large for 1/psi (typical values are 1e-6 to 1e-5). Check the unit."))
    if init_row and abs(init_row["p"] - P["pi"]) > 0.5:
        msgs.append(msg("warn", f"Initial pressure pi = {P['pi']:.1f} psia does not match the Np = 0 row "
                                f"({init_row['p']:g} psia). Check pi - Ef,w and the initial PVT depend on it."))
    if at_pi and "Boi" in P and abs(at_pi["Bo"] - P["Boi"]) / P["Boi"] > 0.002:
        msgs.append(msg("warn", f"Boi ({P['Boi']:.4f}) differs from Bo in the table at p = pi ({at_pi['Bo']:.4f})."))
    if at_pi and "Bgi" in P and abs(at_pi["Bg"] - P["Bgi"]) / P["Bgi"] > 0.002:
        msgs.append(msg("warn", "Bgi differs from Bg in the table at p = pi - check the Bg column unit and the Bgi unit."))
    return P, src, units, msgs


# --------------------------------------------------------------------------- regression
def linfit(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx, sxy, syy = ((x - mx) ** 2).sum(), ((x - mx) * (y - my)).sum(), ((y - my) ** 2).sum()
    slope = sxy / sxx if sxx > 0 else math.nan
    intercept = my - slope * mx
    ssr = ((y - (intercept + slope * x)) ** 2).sum()
    r2 = 1 - ssr / syy if syy > 0 else 1.0
    se_s = se_i = math.nan
    if n > 2 and sxx > 0:
        s2 = ssr / (n - 2)
        se_s, se_i = math.sqrt(s2 / sxx), math.sqrt(s2 * (1 / n + mx * mx / sxx))
    return {"slope": slope, "intercept": intercept, "r2": r2, "se_slope": se_s, "se_intercept": se_i, "n": n}


def originfit(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = (x * y).sum() / (x * x).sum()
    ssr = ((y - b * x) ** 2).sum()
    syy = ((y - y.mean()) ** 2).sum()
    return {"slope": b, "r2": 1 - ssr / syy if syy > 0 else 1.0, "n": len(x)}


# --------------------------------------------------------------------------- Havlena-Odeh analysis
def analyze(ds: Dataset, P: dict, method: str = "unknown-m", m_known: float | None = None,
            include_efw: bool = True, excluded: set | None = None) -> dict:
    """method 'unknown-m': (F-We)/(Eo+Efw) vs (Eg+Efw)/(Eo+Efw) -> intercept N, slope mN
       method 'known-m'  : (F-We) vs Et through origin -> slope N"""
    excluded = excluded or set()
    msgs = []
    pi, Boi, Bgi, Rsi, Swc, cw, cf = (P[k] for k in ("pi", "Boi", "Bgi", "Rsi", "Swc", "cw", "cf"))
    rows = []
    for r in ds.rows:
        F = r["Np"] * r["Bo"] + (r["Gp"] - r["Np"] * r["Rs"]) * r["Bg"] + r["Wp"] * r["Bw"]
        Rp = r["Gp"] / r["Np"] if r["Np"] > 0 else math.nan
        Eo = (r["Bo"] - Boi) + (Rsi - r["Rs"]) * r["Bg"]
        Eg = Boi * (r["Bg"] / Bgi - 1)
        Efw = Boi * (cw * Swc + cf) * (pi - r["p"]) / (1 - Swc) if include_efw else 0.0
        Fnet = F - r["We"]
        den = Eo + Efw
        x = (Eg + Efw) / den if den != 0 else math.nan
        y = Fnet / den if den != 0 else math.nan
        if r["Np"] == 0 or abs(pi - r["p"]) < 0.5:
            status = "initial point (Eo = 0)"
        elif not den > 0:
            status = "Eo + Ef,w <= 0"
        elif not Fnet > 0:
            status = "F <= 0"
        elif r["excel_row"] in excluded:
            status = "excluded by user"
        else:
            status = "used"
        rows.append(dict(r, Rp=Rp, F=F, Fnet=Fnet, Eo=Eo, Eg=Eg, Efw=Efw, x=x, y=y, status=status))
    for r in rows:
        if r["status"] in ("Eo + Ef,w <= 0", "F <= 0"):
            msgs.append(msg("warn", f"Excel row {r['excel_row']} (p = {r['p']:.0f}) not used: {r['status']}. "
                                    "Check pi, Boi, Rsi and units."))
    if any(r["p"] > pi + 0.5 for r in rows):
        msgs.append(msg("warn", "Some pressures are above pi - check the initial pressure."))
    used = [r for r in rows if r["status"] == "used"]
    res = {"rows": rows, "used": used, "messages": msgs, "method": method,
           "N": math.nan, "m": math.nan, "mN": math.nan, "G": math.nan, "r2": math.nan, "fit": None}

    if method == "known-m":
        if m_known is None or not (m_known >= 0):
            msgs.append(msg("error", "Enter a valid known m (>= 0)."))
            return res
        if not used:
            msgs.append(msg("error", "No usable data points."))
            return res
        Et = [r["Eo"] + m_known * r["Eg"] + (1 + m_known) * r["Efw"] for r in used]
        fit = originfit(Et, [r["Fnet"] for r in used])
        res.update(N=fit["slope"], m=m_known, r2=fit["r2"], fit=fit)
    else:
        if len(used) < 2:
            msgs.append(msg("error", f"Need at least 2 usable data points below pi - found {len(used)}."))
            return res
        fit = linfit([r["x"] for r in used], [r["y"] for r in used])
        if not math.isfinite(fit["slope"]):
            msgs.append(msg("error", "All points have the same x value - the straight line cannot be fitted."))
            return res
        res.update(fit=fit, N=fit["intercept"], m=fit["slope"] / fit["intercept"], r2=fit["r2"])
        if len(used) == 2:
            msgs.append(msg("info", "Only 2 points - the line passes exactly through them (R² = 1 means nothing here)."))
        if res["N"] <= 0:
            msgs.append(msg("error", "The fitted intercept N is not positive - the data do not follow a gas-cap "
                                     "straight line (possible water influx or wrong inputs)."))
        if res["m"] < 0:
            msgs.append(msg("warn", "Fitted m is negative - not physical. The reservoir may not have a gas cap, "
                                    "or the data/units are wrong."))
        if res["r2"] < 0.9 and len(used) > 2:
            msgs.append(msg("warn", f"R² = {res['r2']:.3f} - the points are not on a good straight line. "
                                    "Consider water influx, bad data points, or unit errors."))
    res["mN"] = res["m"] * res["N"]
    res["G"] = res["m"] * res["N"] * Boi / Bgi          # SCF
    for r in rows:
        r["Et"] = r["Eo"] + res["m"] * r["Eg"] + (1 + res["m"]) * r["Efw"]
    rps = [r["Rp"] for r in used if math.isfinite(r["Rp"])]
    if rps:
        med = float(np.median(rps))
        if med > 50000 or med < 1:
            msgs.append(msg("warn", f"Cumulative GOR Rp ≈ {med:.1f} SCF/STB looks unrealistic - check the Np and Gp units."))
    if float(np.median([r["Bg"] for r in ds.rows])) > 0.02:
        msgs.append(msg("warn", "Bg in bbl/SCF looks too large - is the Bg column actually in ft3/SCF?"))
    return res


# --------------------------------------------------------------------------- plot specs (plain dicts for plotly)
def havlena_odeh_figure(res: dict, include_efw: bool) -> dict:
    pts = [r for r in res["rows"] if r["status"] in ("used", "excluded by user")]
    used = [r for r in pts if r["status"] == "used"]
    off = [r for r in pts if r["status"] != "used"]
    xs = [r["x"] for r in pts] or [0, 1]
    x0, x1 = min(xs), max(xs)
    pad = (x1 - x0) * 0.08 or 0.1
    lx = [x0 - pad, x1 + pad]
    ly = [(res["N"] + res["mN"] * v) / 1e6 for v in lx]
    hover = "p = %{customdata} psia<br>x = %{x:.4f}<br>y = %{y:.2f} MMSTB<extra></extra>"
    data = [
        {"type": "scatter", "mode": "markers", "name": "data point (used)", "x": [r["x"] for r in used],
         "y": [r["y"] / 1e6 for r in used], "customdata": [round(r["p"]) for r in used], "hovertemplate": hover,
         "marker": {"size": 10, "color": "#257de0", "line": {"width": 2, "color": "white"}}},
        {"type": "scatter", "mode": "lines", "name": "best-fit line", "x": lx, "y": ly,
         "line": {"color": "#f47230", "width": 2}, "hoverinfo": "skip"},
    ]
    if off:
        data.insert(1, {"type": "scatter", "mode": "markers", "name": "excluded", "x": [r["x"] for r in off],
                        "y": [r["y"] / 1e6 for r in off], "customdata": [round(r["p"]) for r in off],
                        "hovertemplate": hover,
                        "marker": {"size": 10, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#a9adb3"}}})
    xl = "(Eg + Ef,w) / (Eo + Ef,w)" if include_efw else "Eg / Eo"
    yl = "F / (Eo + Ef,w)  [MMSTB]" if include_efw else "F / Eo  [MMSTB]"
    return {"data": data, "layout": _layout("Havlena–Odeh straight line", xl, yl)}


def check_figure(res: dict) -> dict:
    pts = [r for r in res["rows"] if r["status"] in ("used", "excluded by user")]
    used = [r for r in pts if r["status"] == "used"]
    off = [r for r in pts if r["status"] != "used"]
    xmax = max([r["Et"] for r in pts] or [1]) * 1.08
    hover = "p = %{customdata} psia<br>Et = %{x:.5f} rb/STB<br>F = %{y:.3f} MM rb<extra></extra>"
    data = [
        {"type": "scatter", "mode": "markers", "name": "data point (used)", "x": [r["Et"] for r in used],
         "y": [r["Fnet"] / 1e6 for r in used], "customdata": [round(r["p"]) for r in used], "hovertemplate": hover,
         "marker": {"size": 10, "color": "#257de0", "line": {"width": 2, "color": "white"}}},
        {"type": "scatter", "mode": "lines", "name": "slope = N", "x": [0, xmax], "y": [0, res["N"] * xmax / 1e6],
         "line": {"color": "#f47230", "width": 2}, "hoverinfo": "skip"},
    ]
    if off:
        data.insert(1, {"type": "scatter", "mode": "markers", "name": "excluded", "x": [r["Et"] for r in off],
                        "y": [r["Fnet"] / 1e6 for r in off], "customdata": [round(r["p"]) for r in off],
                        "hovertemplate": hover,
                        "marker": {"size": 10, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#a9adb3"}}})
    return {"data": data, "layout": _layout("Check plot: F vs Et", "Et = Eo + m·Eg + (1+m)·Ef,w  [rb/STB]",
                                            "F − We  [MM rb]", rangemode="tozero")}


def _layout(title, xl, yl, rangemode="normal"):
    return {"title": {"text": title}, "height": 420, "margin": {"l": 60, "r": 20, "t": 50, "b": 55},
            "xaxis": {"title": {"text": xl}, "rangemode": rangemode, "zeroline": True},
            "yaxis": {"title": {"text": yl}, "rangemode": rangemode, "zeroline": True},
            "legend": {"orientation": "h", "y": -0.22}, "hovermode": "closest"}


# --------------------------------------------------------------------------- tables & files
def results_table(res: dict) -> list[dict]:
    """Rows for display / download (display units)."""
    out = []
    for r in res["rows"]:
        out.append({
            "Use": r["status"] == "used", "Excel row": r["excel_row"], "p (psia)": r["p"],
            "Np (MMSTB)": r["Np"] / 1e6, "Gp (MMSCF)": r["Gp"] / 1e6, "Rp (SCF/STB)": r["Rp"],
            "Bo (bbl/STB)": r["Bo"], "Bg (bbl/SCF)": r["Bg"], "Rs (SCF/STB)": r["Rs"],
            "F - We (MM rb)": r["Fnet"] / 1e6, "Eo (rb/STB)": r["Eo"], "Eg (rb/STB)": r["Eg"],
            "Ef,w (rb/STB)": r["Efw"], "x": r["x"] if r["status"] in ("used", "excluded by user") else math.nan,
            "y (MMSTB)": r["y"] / 1e6 if r["status"] in ("used", "excluded by user") else math.nan,
            "Status": r["status"],
        })
    return out


EXAMPLE_SHEETS: list[Sheet] = [
    ("Production", [
        ["Pressure (psia)", "Np (MMSTB)", "Gp (MMMSCF)", "Bo (bbl/STB)", "Rs (SCF/STB)", "Bg (cuft/SCF)"],
        [3300, 0, 0, 1.2511, 510, 0.00488505],
        [3150, 3.295, 3.460, 1.2353, 477, 0.00516580],
        [3000, 5.903, 6.257, 1.2222, 450, 0.00539040],
        [2850, 8.852, 10.268, 1.2122, 425, 0.00567115],
        [2700, 11.503, 14.206, 1.2022, 401, 0.00600805],
        [2550, 14.513, 18.359, 1.1922, 375, 0.00634495],
        [2400, 17.730, 23.049, 1.1822, 352, 0.00673800]]),
    ("Initial", [
        ["Property", "Value", "Unit"],
        ["Initial pressure pi", 3300, "psia"],
        ["Boi", 1.2511, "bbl/STB"],
        ["Bgi", 0.00488505, "cuft/SCF"],
        ["Rsi", 510, "SCF/STB"],
        ["Swc", 0.05, "fraction"],
        ["cw", 3e-6, "1/psi"],
        ["cf", 4e-6, "1/psi"],
        ["Porosity", 0.25, "fraction (not used by Havlena-Odeh)"]]),
]
TEMPLATE_SHEETS: list[Sheet] = [
    ("Production", [["Pressure (psia)", "Np (MMSTB)", "Gp (MMMSCF)", "Bo (bbl/STB)", "Rs (SCF/STB)", "Bg (cuft/SCF)",
                     "Wp (STB) - optional", "Bw (bbl/STB) - optional", "We (bbl) - optional"]]),
    ("Initial", [["Property", "Value", "Unit"], ["Initial pressure pi", None, "psia"], ["Boi", None, "bbl/STB"],
                 ["Bgi", None, "cuft/SCF"], ["Rsi", None, "SCF/STB"], ["Swc", None, "fraction"],
                 ["cw", None, "1/psi"], ["cf", None, "1/psi"]]),
]


def write_xlsx(sheets: list[Sheet]) -> bytes:
    import openpyxl
    from openpyxl.styles import Font
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name[:31])
        for row in rows:
            ws.append([None if (isinstance(v, float) and not math.isfinite(v)) else v for v in row])
        if rows:
            for cell in ws[1]:
                cell.font = Font(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 45)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def results_workbook(res: dict, P: dict, src: dict, file_name: str, include_efw: bool) -> bytes:
    summary = [
        ["Havlena-Odeh material balance results"], ["Input file", file_name],
        ["Method", "Known m: F vs Et through origin" if res["method"] == "known-m"
         else "Unknown m: F/Eo vs Eg/Eo straight line"],
        ["Ef,w included", "yes" if include_efw else "no"], [],
        ["Result", "Value", "Unit"], ["N", res["N"], "STB"], ["m", res["m"], "fraction"], ["mN", res["mN"], "STB"],
        ["G", res["G"], "SCF"], ["R squared", res["r2"], ""], ["Points used", len(res["used"]), ""], [],
        ["Parameter used", "Value", "Unit", "Source"],
        ["pi", P["pi"], "psia", src["pi"]], ["Boi", P["Boi"], "bbl/STB", src["Boi"]],
        ["Bgi", P["Bgi"], "bbl/SCF", src["Bgi"]], ["Rsi", P["Rsi"], "SCF/STB", src["Rsi"]],
        ["Swc", P["Swc"], "fraction", src["Swc"]], ["cw", P["cw"], "1/psi", src["cw"]], ["cf", P["cf"], "1/psi", src["cf"]],
    ]
    tbl = results_table(res)
    calc = [list(tbl[0].keys())] + [list(r.values()) for r in tbl] if tbl else [[]]
    return write_xlsx([("Results", summary), ("Calculations", calc)])
