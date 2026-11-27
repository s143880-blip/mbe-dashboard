"""Creates the test input files (Excel/CSV) in a folder. Called automatically by the tests."""
import csv
from pathlib import Path

import numpy as np
import openpyxl

K = 5.614583
p = [3300, 3150, 3000, 2850, 2700, 2550, 2400]
Np = [0, 3.295, 5.903, 8.852, 11.503, 14.513, 17.730]
Gp = [0, 3.460, 6.257, 10.268, 14.206, 18.359, 23.049]
Bo = [1.2511, 1.2353, 1.2222, 1.2122, 1.2022, 1.1922, 1.1822]
Rs = [510, 477, 450, 425, 401, 375, 352]
Bg = [0.00488505, 0.0051658, 0.0053904, 0.00567115, 0.00600805, 0.00634495, 0.006738]
INIT = [("Property", "Value", "Unit"), ("Initial pressure pi", 3300, "psia"), ("Boi", 1.2511, "bbl/STB"),
        ("Bgi", 0.00488505, "cuft/SCF"), ("Rsi", 510, "SCF/STB"), ("Swc", 0.05, "fraction"),
        ("cw", 3e-6, "psi-1"), ("cf", 4e-6, "psi-1"), ("Porosity", 0.25, "fraction (not used)")]
HDR = ("Pressure (psia)", "Np (MMSTB)", "Gp (MMMSCF)", "Bo (bbl/STB)", "Rs (SCF/STB)", "Bg (cuft/SCF)")
PROD = [HDR] + list(zip(p, Np, Gp, Bo, Rs, Bg))


def _save(path, sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))
    wb.save(path)


def build(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    # 1. the HW2 data, two sheets
    _save(folder / "HW2_input.xlsx", [("Production", PROD), ("Initial", INIT)])

    # 2. synthetic data made from a chosen answer N = 50 MMSTB, m = 0.3 (one sheet, params below table)
    N0, m0 = 50e6, 0.3
    pa, Boa, Rsa = np.array(p, float), np.array(Bo), np.array(Rs, float)
    Bga = np.array(Bg) / K
    Eo = Boa - Boa[0] + (Rsa[0] - Rsa) * Bga
    Eg = Boa[0] * (Bga / Bga[0] - 1)
    Efw = Boa[0] * (3e-6 * 0.05 + 4e-6) * (3300 - pa) / 0.95
    F = N0 * (Eo + m0 * Eg + (1 + m0) * Efw)
    NpS = np.array(Np) * 1e6 * 0.45
    GpS = np.zeros_like(NpS)
    k = NpS > 0
    GpS[k] = (F[k] - NpS[k] * Boa[k]) / Bga[k] + NpS[k] * Rsa[k]
    rows = [("Average reservoir pressure p (psia)", "Cumulative oil Np (STB)", "Cumulative gas Gp (SCF)",
             "Oil FVF Bo (bbl/STB)", "Solution GOR Rs (SCF/STB)", "Gas FVF Bg (bbl/SCF)")]
    rows += [tuple(float(v) for v in t) for t in zip(pa, NpS, GpS, Boa, Rsa, Bga)]
    rows += [(), ("Initial reservoir parameters",), ("Initial pressure pi", "3300 psia"),
             ("Initial oil FVF Boi", 1.2511, "bbl/STB"), ("Initial gas FVF Bgi", float(Bga[0]), "bbl/SCF"),
             ("Initial solution GOR Rsi", 510, "SCF/STB"), ("Connate water saturation Swc", 0.05),
             ("Water compressibility cw", "3.00E–06 psi-1"), ("Formation compressibility cf", "4.00E–06 psi-1")]
    _save(folder / "test_synthetic_known_answer.xlsx", [("Data", rows)])

    # 3. CSV with Rp instead of Gp, Np in MSTB, Bg in rb/Mscf
    with open(folder / "test_rp_csv.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p (psia)", "Np (MSTB)", "Rp (SCF/STB)", "Bo (bbl/STB)", "Rs (SCF/STB)", "Bg (rb/Mscf)"])
        for i in range(7):
            w.writerow([p[i], Np[i] * 1e3, (Gp[i] * 1e9 / (Np[i] * 1e6) if Np[i] else ""), Bo[i], Rs[i],
                        Bg[i] / K * 1000])
        w.writerow([])
        for r in INIT[1:]:
            w.writerow(r)

    # 4. Bg column missing
    _save(folder / "test_missing_column.xlsx", [("Production", [r[:5] for r in PROD]), ("Initial", INIT)])

    # 5. a text cell and an empty cell
    bad = [list(r) for r in PROD]
    bad[3][3] = "abc"
    bad[5][4] = None
    _save(folder / "test_bad_cells.xlsx", [("Production", bad), ("Initial", INIT)])

    # 6. no parameter list at all
    _save(folder / "test_no_initial_params.xlsx", [("Production", PROD)])
    return folder


if __name__ == "__main__":
    print("written to", build(Path("test_files")))
