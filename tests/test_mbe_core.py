"""
Tests for mbe_core.py.  Run with:   python -m pytest -v

Each numerical result from the app is compared with an independent NumPy
calculation written directly from the Havlena-Odeh equations (reference()).
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mbe_core as mc
from tests.build_test_data import build

DATA = build(Path(tempfile.mkdtemp()) / "data")     # test input files are created fresh for every run
K = 5.614583  # ft3 per bbl

# HW2 data in field units (as given in the assignment)
P_ = [3300, 3150, 3000, 2850, 2700, 2550, 2400]
NP = [0, 3.295, 5.903, 8.852, 11.503, 14.513, 17.730]          # MMSTB
GP = [0, 3.460, 6.257, 10.268, 14.206, 18.359, 23.049]          # MMMSCF
BO = [1.2511, 1.2353, 1.2222, 1.2122, 1.2022, 1.1922, 1.1822]
RS = [510, 477, 450, 425, 401, 375, 352]
BG = [0.00488505, 0.0051658, 0.0053904, 0.00567115, 0.00600805, 0.00634495, 0.006738]  # ft3/SCF


def reference(idx, pi=3300.0, swc=0.05, cw=3e-6, cf=4e-6):
    """Independent Havlena-Odeh calculation (F/(Eo+Efw) vs (Eg+Efw)/(Eo+Efw))."""
    p = np.array(P_, float)[idx]
    Np = np.array(NP)[idx] * 1e6
    Gp = np.array(GP)[idx] * 1e9
    Bo = np.array(BO)[idx]
    Rs = np.array(RS, float)[idx]
    Bg = np.array(BG)[idx] / K
    Boi, Bgi, Rsi = BO[0], BG[0] / K, RS[0]
    F = Np * (Bo + (np.divide(Gp, Np, out=np.zeros_like(Gp), where=Np > 0) - Rs) * Bg)
    Eo = Bo - Boi + (Rsi - Rs) * Bg
    Eg = Boi * (Bg / Bgi - 1)
    Efw = Boi * (cw * swc + cf) * (pi - p) / (1 - swc)
    k = Np > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = ((Eg + Efw) / (Eo + Efw))[k], (F / (Eo + Efw))[k]
    slope, N = np.polyfit(x, y, 1)
    m = slope / N
    return {"N": N, "m": m, "G": m * N * Boi / Bgi, "r2": np.corrcoef(x, y)[0, 1] ** 2}


ALL = list(range(7))


def run(fname, **kw):
    sheets = mc.read_workbook((DATA / fname).read_bytes(), fname)
    ds = mc.extract_dataset(sheets, kw.pop("unit_overrides", None))
    P, src, units, pmsgs = mc.resolve_params(ds, kw.pop("edits", None), kw.pop("edit_units", None))
    res = mc.analyze(ds, P, **kw)
    return ds, P, src, pmsgs, res


def close(a, b, rel=2e-3):
    return abs(a - b) <= rel * abs(b)


# ------------------------------------------------------------------ main results
def test_hw2_matches_reference():
    ds, P, src, pmsgs, res = run("HW2_input.xlsx")
    ref = reference(ALL)
    assert close(res["N"], ref["N"]) and close(res["m"], ref["m"]) and close(res["G"], ref["G"])
    assert abs(res["r2"] - ref["r2"]) < 2e-3
    assert round(res["N"] / 1e6, 1) == 112.0 and round(res["m"], 3) == 0.501 and round(res["G"] / 1e9, 1) == 80.7
    assert src["pi"].startswith("file") and ds.col_units["Bg"]["label"] == "ft3/SCF"
    assert ds.col_units["Gp"]["label"] == "MMMSCF (Bscf)"
    assert len(res["used"]) == 6
    assert not [m for m in ds.messages + pmsgs + res["messages"] if m["level"] in ("warn", "error")]


def test_synthetic_exact_answer():
    """File built from N = 50 MMSTB, m = 0.3 (one sheet, parameters below the table, other units)."""
    ds, P, src, pmsgs, res = run("test_synthetic_known_answer.xlsx")
    assert abs(res["N"] - 50e6) / 50e6 < 1e-6
    assert abs(res["m"] - 0.3) < 1e-6
    assert res["r2"] > 0.999999
    assert P["pi"] == 3300 and abs(P["cw"] - 3e-6) < 1e-12      # "3300 psia", "3.00E–06 psi-1" in one cell


def test_csv_with_rp_and_other_units():
    ds, P, src, pmsgs, res = run("test_rp_csv.csv")
    ref = reference(ALL)
    assert close(res["N"], ref["N"]) and close(res["m"], ref["m"])
    assert ds.col_units["Np"]["label"] == "MSTB" and ds.col_units["Bg"]["label"] == "bbl/MSCF"
    assert not ds.skipped                      # Np = 0 row with empty Rp is kept (Gp = 0)


def test_bad_cells_are_skipped():
    ds, P, src, pmsgs, res = run("test_bad_cells.xlsx")
    ref = reference([0, 1, 3, 5, 6])
    assert close(res["N"], ref["N"]) and close(res["m"], ref["m"])
    assert len(ds.skipped) == 2
    text = " ".join(m["text"] for m in ds.messages)
    assert '"abc" is not a number' in text and "Rs is empty" in text


def test_no_parameter_list_uses_table_row():
    ds, P, src, pmsgs, res = run("test_no_initial_params.xlsx")
    ref = reference(ALL, swc=0, cw=0, cf=0)
    assert close(res["N"], ref["N"]) and close(res["m"], ref["m"])
    assert src["pi"].startswith("production table") and src["Swc"].startswith("not given")


def test_missing_column_gives_clear_error():
    with pytest.raises(ValueError, match="missing: Bg"):
        run("test_missing_column.xlsx")


# ------------------------------------------------------------------ options
def test_known_m_method_gives_same_N():
    ref = reference(ALL)
    *_, res = run("HW2_input.xlsx", method="known-m", m_known=ref["m"])
    assert abs(res["N"] - ref["N"]) / ref["N"] < 0.01


def test_known_m_requires_value():
    *_, res = run("HW2_input.xlsx", method="known-m", m_known=None)
    assert math.isnan(res["N"]) and any(m["level"] == "error" for m in res["messages"])


def test_without_efw_matches_lecture_form():
    *_, res = run("HW2_input.xlsx", include_efw=False)
    ref = reference(ALL, swc=0, cw=0, cf=0)
    assert close(res["N"], ref["N"]) and round(res["N"] / 1e6, 1) == 108.7


def test_excluding_rows():
    *_, res = run("HW2_input.xlsx", excluded={3, 4})
    assert len(res["used"]) == 4
    ref = reference([0, 3, 4, 5, 6])
    assert close(res["N"], ref["N"])


def test_pi_edit_triggers_warning():
    *_, pmsgs, res = run("HW2_input.xlsx", edits={"pi": "2740"}, edit_units={"pi": "psia"})
    assert any("does not match" in m["text"] for m in pmsgs)


def test_wrong_bg_unit_is_flagged():
    *_, pmsgs, res = run("HW2_input.xlsx", unit_overrides={"Bg": "bbl/SCF"})
    assert any("Bgi differs" in m["text"] for m in pmsgs)


def test_too_few_points():
    sheets = [("S", [["p (psia)", "Np (MMSTB)", "Gp (MMMSCF)", "Bo", "Rs (SCF/STB)", "Bg (cuft/SCF)"],
                     [3300, 0, 0, 1.2511, 510, 0.00488505], [3150, 3.295, 3.46, 1.2353, 477, 0.0051658]])]
    ds = mc.extract_dataset(sheets)
    P, *_ = mc.resolve_params(ds)
    res = mc.analyze(ds, P)
    assert any("at least 2" in m["text"] for m in res["messages"])


# ------------------------------------------------------------------ file handling & helpers
def test_old_xls_rejected():
    with pytest.raises(ValueError, match="xlsx"):
        mc.read_workbook(b"xxxx", "old.xls")


def test_junk_file_rejected():
    with pytest.raises(ValueError, match="not a valid"):
        mc.read_workbook(b"hello", "junk.xlsx")


def test_template_and_results_roundtrip():
    sheets = mc.read_workbook(mc.write_xlsx(mc.EXAMPLE_SHEETS), "x.xlsx")
    ds = mc.extract_dataset(sheets)
    P, src, units, _ = mc.resolve_params(ds)
    res = mc.analyze(ds, P)
    assert close(res["N"], reference(ALL)["N"])
    out = mc.read_workbook(mc.results_workbook(res, P, src, "x.xlsx", True), "r.xlsx")
    assert [s[0] for s in out] == ["Results", "Calculations"]
    tmpl = mc.read_workbook(mc.write_xlsx(mc.TEMPLATE_SHEETS), "t.xlsx")
    assert mc.find_production_table(tmpl) is not None


@pytest.mark.parametrize("cell,val", [(2740, 2740.0), ("2740 psia", 2740.0), ("4.00E–06 psi-1", 4e-6),
                                      ("1,234.5", 1234.5), ("4 x 10^-6", 4e-6), ("abc", math.nan), (None, math.nan)])
def test_parse_num(cell, val):
    got = mc.parse_num(cell)
    assert (math.isnan(got) and math.isnan(val)) or abs(got - val) < 1e-12


@pytest.mark.parametrize("kind,text,label", [("Bg", "cuft/SCF", "ft3/SCF"), ("Bg", "rb/Mscf", "bbl/MSCF"),
                                             ("Gp", "MMMSCF", "MMMSCF (Bscf)"), ("Gp", "MMscf", "MMSCF"),
                                             ("Np", "MMSTB", "MMSTB"), ("cf", "psi-1", "1/psi"),
                                             ("pi", "psia", "psia")])
def test_detect_unit(kind, text, label):
    assert mc.detect_unit(kind, text) == label


def test_figures_are_valid_plotly_dicts():
    *_, res = run("HW2_input.xlsx", excluded={5})
    for fig in (mc.havlena_odeh_figure(res, True), mc.check_figure(res)):
        assert {"data", "layout"} <= fig.keys()
        assert all(len(t["x"]) == len(t["y"]) for t in fig["data"])
        assert any(t["name"] == "excluded" for t in fig["data"])
