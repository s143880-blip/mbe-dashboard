"""
Havlena-Odeh Material Balance Dashboard (Streamlit)
PNGE 4412 Reservoir Engineering - HW2, Question 2

Run locally:   streamlit run app.py
All calculations are in mbe_core.py (tested in tests/).
"""
import hashlib
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import mbe_core as mc

st.set_page_config(page_title="Havlena–Odeh MBE Dashboard", page_icon="🛢️", layout="wide")

PARAM_INFO = [
    ("pi", "Initial pressure pi"), ("Boi", "Initial oil FVF Boi"), ("Bgi", "Initial gas FVF Bgi"),
    ("Rsi", "Initial solution GOR Rsi"), ("Swc", "Connate water saturation Swc"),
    ("cw", "Water compressibility cw"), ("cf", "Formation compressibility cf"),
]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@st.cache_data(show_spinner=False)
def load_sheets(data: bytes, name: str):
    return mc.read_workbook(data, name)


def fmt(v, d=3):
    if v is None or not math.isfinite(v):
        return "–"
    return f"{v:,.{d}f}"


def show_messages(messages):
    if not messages:
        st.success("All input checks passed.", icon="✅")
        return
    order = {"error": 0, "warn": 1, "info": 2}
    for m in sorted(messages, key=lambda m: order[m["level"]]):
        if m["level"] == "error":
            st.error(m["text"], icon="⛔")
        elif m["level"] == "warn":
            st.warning(m["text"], icon="⚠️")
        else:
            st.info(m["text"], icon="ℹ️")


# =============================================================== header
st.title("Havlena–Odeh Material Balance Dashboard")
st.caption("Gas-cap oil reservoir · finds **N** (stock-tank oil initially in place), **m** (gas-cap ratio) and "
           "**G** (gas-cap gas initially in place) · PNGE 4412 Reservoir Engineering · HW 2")

if "reset" not in st.session_state:
    st.session_state.reset = 0

# =============================================================== sidebar: input file
with st.sidebar:
    st.header("1 · Input data")
    upload = st.file_uploader("Upload an Excel file (.xlsx) or CSV", type=["xlsx", "xlsm", "csv"])
    st.download_button("Download blank template (.xlsx)", mc.write_xlsx(mc.TEMPLATE_SHEETS),
                       file_name="MBE_input_template.xlsx", mime=XLSX_MIME)
    st.download_button("Download HW2 example data (.xlsx)", mc.write_xlsx(mc.EXAMPLE_SHEETS),
                       file_name="HW2_input_data.xlsx", mime=XLSX_MIME)
    st.caption("The file needs a table with headers p, Np, Gp, Bo, Bg, Rs (units in brackets) and a "
               "Property | Value | Unit list with pi, Boi, Bgi, Rsi, Swc, cw, cf. See *How to use* at the bottom.")

if upload is None:
    sheets, file_name, fid = mc.EXAMPLE_SHEETS, "HW2 example data (built-in)", "example"
    st.info("No file uploaded yet – showing the built-in **HW2 example data**. "
            "Upload your own Excel file in the sidebar to analyse it.", icon="📄")
else:
    raw = upload.getvalue()
    file_name, fid = upload.name, hashlib.md5(raw).hexdigest()[:10]
    try:
        sheets = load_sheets(raw, upload.name)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read this file: {e}", icon="⛔")
        st.stop()

key = f"{fid}_{st.session_state.reset}"          # new file or reset -> fresh widgets

# first pass: find the table and the units written in the headers
try:
    ds0 = mc.extract_dataset(sheets)
except ValueError as e:
    st.error(str(e), icon="⛔")
    st.markdown("Check the file against the required format (see **How to use** below), "
                "or download the blank template from the sidebar and copy your data into it.")
    st.stop()

# =============================================================== sidebar: units, parameters, method
with st.sidebar:
    st.header("2 · Column units")
    st.caption("Read from the column headers – change one if it is wrong.")
    overrides = {}
    for k in ["p", "Np", "Gp", "Rp", "Bo", "Bg", "Rs", "Wp", "Bw", "We"]:
        cu = ds0.col_units.get(k)
        if not cu:
            continue
        options = mc.unit_labels(k)
        help_txt = (f'Header says "{cu["header_text"]}"' if cu["header_text"] else "No unit in the header")
        choice = st.selectbox(f"{k}", options, index=options.index(cu["label"]), key=f"u_{k}_{key}",
                              help=help_txt, disabled=len(options) < 2)
        if choice != cu["label"]:
            overrides[k] = choice

ds = mc.extract_dataset(sheets, overrides)
P0, src0, units0, _ = mc.resolve_params(ds)

with st.sidebar:
    st.header("3 · Initial parameters")
    st.caption("Values found in the file. Edit any of them to test the effect.")
    edits, edit_units = {}, {}
    for k, label in PARAM_INFO:
        unit = units0.get(k, "fraction")
        found = ds.found.get(k)
        if found and src0[k].startswith("file"):
            shown = found["raw"]
        else:
            shown = P0.get(k, math.nan) / (mc.unit_factor(k, unit) if k in mc.UNIT_OPTIONS else 1)
        shown_txt = f"{shown:.8g}" if math.isfinite(shown) else ""
        c1, c2 = st.columns([3, 2])
        txt = c1.text_input(label, value=shown_txt, key=f"pv_{k}_{key}")
        opts = mc.unit_labels(k) if k in mc.UNIT_OPTIONS else ["fraction"]
        u = c2.selectbox("unit", opts, index=opts.index(unit) if unit in opts else 0, key=f"pu_{k}_{key}",
                         label_visibility="hidden", disabled=len(opts) < 2)
        if txt.strip() != shown_txt or u != unit:
            edits[k], edit_units[k] = txt.strip(), u
        st.caption(f"source: {src0.get(k, 'missing')}" if k not in edits else "source: entered by you")
    if st.button("Reset to file values"):
        st.session_state.reset += 1
        st.rerun()

    st.header("4 · Method")
    method_label = st.radio("Gas-cap size m", ["m unknown – find N and m", "m known – find N only"], index=0,
                            help="m unknown: plot F/Eo vs Eg/Eo (intercept = N, slope = mN). "
                                 "m known: plot F vs Et through the origin (slope = N).")
    method = "known-m" if method_label.startswith("m known") else "unknown-m"
    m_known = None
    if method == "known-m":
        default_m = f"{P0['m']:g}" if "m" in P0 else ""
        m_known = mc.parse_num(st.text_input("Known m", value=default_m, key=f"mk_{key}"))
        m_known = None if math.isnan(m_known) else m_known
    include_efw = st.checkbox("Include connate-water & rock expansion (Ef,w)", value=True)
    st.caption("Water influx is assumed zero unless the file has a We column (then F − We is used).")

P, src, units, pmsgs = mc.resolve_params(ds, edits, edit_units)
if any(m["level"] == "error" for m in pmsgs):
    st.subheader(f"File: {file_name}")
    show_messages(ds.messages + pmsgs)
    st.stop()

# =============================================================== layout: results on top, table below
st.markdown(f"**File:** {file_name} · sheet “{ds.table.sheet}” · {len(ds.rows)} data rows"
            + (f" · {len(ds.skipped)} rows skipped" if ds.skipped else ""))
top = st.container()
st.subheader("Calculation table")
st.caption("Untick **Use** to leave a point out of the straight-line fit. "
           "The row at p = pi (Np = 0) is never used because Eo = 0 there.")

res0 = mc.analyze(ds, P, method, m_known, include_efw)        # no exclusions -> stable table
base = pd.DataFrame(mc.results_table(res0))
usable = set(base.loc[base["Status"] == "used", "Excel row"])
edited = st.data_editor(
    base, hide_index=True, key=f"tbl_{key}",
    disabled=[c for c in base.columns if c != "Use"],
    column_config={
        "Use": st.column_config.CheckboxColumn("Use", help="Include this point in the fit"),
        "p (psia)": st.column_config.NumberColumn(format="%.0f"),
        "Np (MMSTB)": st.column_config.NumberColumn(format="%.3f"),
        "Gp (MMSCF)": st.column_config.NumberColumn(format="%.0f"),
        "Rp (SCF/STB)": st.column_config.NumberColumn(format="%.0f"),
        "Bo (bbl/STB)": st.column_config.NumberColumn(format="%.4f"),
        "Bg (bbl/SCF)": st.column_config.NumberColumn(format="%.6f"),
        "Rs (SCF/STB)": st.column_config.NumberColumn(format="%.0f"),
        "F - We (MM rb)": st.column_config.NumberColumn(format="%.3f"),
        "Eo (rb/STB)": st.column_config.NumberColumn(format="%.5f"),
        "Eg (rb/STB)": st.column_config.NumberColumn(format="%.5f"),
        "Ef,w (rb/STB)": st.column_config.NumberColumn(format="%.5f"),
        "x": st.column_config.NumberColumn(format="%.4f"),
        "y (MMSTB)": st.column_config.NumberColumn(format="%.2f"),
    },
)
excluded = {int(r) for r, use in zip(edited["Excel row"], edited["Use"]) if not use and r in usable}
res = mc.analyze(ds, P, method, m_known, include_efw, excluded)

with top:
    ok = math.isfinite(res["N"])
    c1, c2, c3, c4 = st.columns(4)
    if ok:
        se = res["fit"].get("se_intercept") if res["fit"] else None
        c1.metric("Original oil in place, N", f"{fmt(res['N'] / 1e6, 2)} MMSTB",
                  help=f"{res['N']:,.0f} STB" + (f" · ± {se / 1e6:.2f} MMSTB (1 standard error)"
                                                  if se and math.isfinite(se) else ""))
        c2.metric("Gas-cap ratio, m", fmt(res["m"], 3),
                  help="entered (known m)" if method == "known-m" else f"mN = slope = {res['mN'] / 1e6:.2f} MMSTB")
        c3.metric("Gas-cap gas in place, G", f"{fmt(res['G'] / 1e9, 2)} Bscf",
                  help=f"{res['G'] / 1e6:,.0f} MMSCF · G = m·N·Boi/Bgi")
        c4.metric("Fit quality, R²", fmt(res["r2"], 4), help=f"{len(res['used'])} points used")
    else:
        st.error("No result – see the messages below.", icon="⛔")

    with st.expander("Checks & messages", expanded=True):
        show_messages(ds.messages + pmsgs + res["messages"])

    if ok:
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(go.Figure(mc.havlena_odeh_figure(res, include_efw)))
            st.caption(f"y = {res['N'] / 1e6:.2f} + {res['mN'] / 1e6:.2f}·x (MMSTB) → intercept = N, slope = mN"
                       + (" (line implied by the known m)" if method == "known-m" else ""))
        with g2:
            st.plotly_chart(go.Figure(mc.check_figure(res)))
            st.caption(f"Check: with m = {res['m']:.3f} the points should lie on a straight line through the "
                       f"origin with slope N = {res['N'] / 1e6:.2f} MMSTB.")

        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Download results (.xlsx)",
                           mc.results_workbook(res, P, src, file_name, include_efw),
                           file_name="MBE_results.xlsx", mime=XLSX_MIME)
        d2.download_button("Download table (.csv)", pd.DataFrame(mc.results_table(res)).to_csv(index=False),
                           file_name="MBE_calculations.csv", mime="text/csv")

# =============================================================== help
with st.expander("How to use this dashboard & equations"):
    st.markdown("""
1. **Upload** an Excel file (.xlsx) or CSV in the sidebar, or use **Download blank template** and fill it in.
   Without a file the built-in HW2 data is shown.
2. The file needs a table with one header row containing **p, Np, Gp, Bo, Bg, Rs** with units in brackets,
   e.g. `Np (MMSTB)`, `Bg (cuft/SCF)`. Optional columns: `Wp`, `Bw`, `We`, or `Rp` instead of `Gp`.
3. The initial parameters go in a **Property | Value | Unit** list, on a second sheet or below the table:
   `pi, Boi, Bgi, Rsi, Swc, cw, cf`. If Boi, Bgi or Rsi are missing they are taken from the row at p = pi;
   missing Swc, cw, cf are set to 0.
4. Check **Column units** and **Initial parameters** in the sidebar – each value shows where it came from.
5. Read **N, m, G** at the top. Hover over points for their values; untick rows in the table to test a point.
6. Download the results (.xlsx) or the calculation table (.csv).
""")
    st.latex(r"F = N_p\left[B_o + (R_p - R_s)B_g\right] + W_pB_w,\qquad R_p = G_p/N_p")
    st.latex(r"E_o = (B_o - B_{oi}) + (R_{si} - R_s)B_g,\qquad E_g = B_{oi}\left(\frac{B_g}{B_{gi}} - 1\right),"
             r"\qquad E_{f,w} = B_{oi}\frac{c_wS_{wc} + c_f}{1 - S_{wc}}(p_i - p)")
    st.latex(r"\frac{F - W_e}{E_o + E_{f,w}} = N + mN\,\frac{E_g + E_{f,w}}{E_o + E_{f,w}}"
             r"\qquad\Rightarrow\qquad \text{intercept} = N,\ \text{slope} = mN")
    st.latex(r"G = \frac{m\,N\,B_{oi}}{B_{gi}}\qquad (1\ \text{bbl} = 5.6146\ \text{ft}^3)")
