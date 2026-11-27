# Havlena–Odeh Material Balance Dashboard

**PNGE 4412 Reservoir Engineering · HW 2, Question 2 · Sultan Qaboos University · Fall 2026**

An interactive Streamlit dashboard. It reads an Excel file of production and PVT data for a **gas-cap oil reservoir** and uses the **Havlena–Odeh straight-line method** to calculate:

- **N**: stock-tank original oil in place
- **m**: initial gas-cap to oil volume ratio
- **G**: gas-cap gas initially in place

**Live app:** https://ahmed-mbe-dashboard.streamlit.app

---

## How to use

1. Open the live app link. The built-in HW2 data is shown until you upload a file.
2. In the sidebar, click **Browse files** and upload an Excel file (`.xlsx`) or a CSV.
   To prepare a new file, click **Download blank template (.xlsx)** and fill it in.
3. N, m, G and R² appear at the top, with the Havlena–Odeh plot and a check plot.
4. In the sidebar, check **Column units** and **Initial parameters**. Each value shows where it was read from (e.g. `file Initial!B2`). You can edit any of them to test the effect.
5. Choose the **Method**:
   - *m unknown* (default): finds both N and m.
   - *m known*: enter m and find N only.
6. In the **Calculation table**, untick **Use** to leave a point out of the fit.
7. Download the results (`.xlsx`) or the calculation table (`.csv`).

## Input file format

**Production table.** It needs one header row, with units in brackets. The column order does not matter.

| Column | Required | Accepted units |
|---|---|---|
| p | yes | psia, psig, bar, kPa, MPa |
| Np | yes | STB, MSTB, MMSTB |
| Gp | yes (or Rp) | SCF, MSCF, MMSCF, MMMSCF/Bscf |
| Bo | yes | bbl/STB |
| Bg | yes | bbl/SCF, ft³/SCF (cuft/SCF), bbl/MSCF, ft³/MSCF |
| Rs | yes | SCF/STB, MSCF/STB |
| Wp, Bw, We | optional | produced water and a known water influx |

**Initial parameters.** These go in a `Property | Value | Unit` list, either on a second sheet or below the table: `pi, Boi, Bgi, Rsi, Swc, cw, cf`.

- If Boi, Bgi or Rsi is missing, it is taken from the table row at p = pi.
- If Swc, cw or cf is missing, it is set to 0.

To get an example file, click **Download HW2 example data (.xlsx)** in the app's sidebar.

## Method

```
F   = Np[Bo + (Rp − Rs)Bg] + WpBw            Rp = Gp/Np
Eo  = (Bo − Boi) + (Rsi − Rs)Bg
Eg  = Boi (Bg/Bgi − 1)
Efw = Boi (cw·Swc + cf)(pi − p) / (1 − Swc)

(F − We)/(Eo + Efw) = N + mN · (Eg + Efw)/(Eo + Efw)   →  intercept = N, slope = mN
G = m·N·Boi / Bgi
```

## HW2 result (pi = 3300 psia, with Ef,w)

| N | m | G | R² |
|---|---|---|---|
| 112.0 MMSTB | 0.501 | 80.7 Bscf | 0.963 |

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `mbe_core.py` | All calculations, file reading and checks |
| `tests/test_mbe_core.py` | 31 automated tests |
| `tests/build_test_data.py` | Creates the test input files: HW2, a synthetic case with a known answer, CSV, bad cells, a missing column, no parameters. Run `python -m tests.build_test_data` to save them to a `test_files` folder. |

## Testing

```bash
pip install -r requirements.txt pytest
python -m pytest -v
```

The tests compare every result with an independent NumPy calculation. They also include a synthetic file built from N = 50 MMSTB and m = 0.3, which the app recovers exactly. Other tests check:

- unit conversion, CSV input, and skipping bad cells;
- missing columns and missing parameters;
- the known-m method and excluding points;
- warnings for a wrong pi or a wrong Bg unit;
- rejection of invalid files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
