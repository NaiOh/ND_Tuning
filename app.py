"""
MetrixND Binary Tuning Assistant — Streamlit app.

Upload a MetrixND model export (.xlsx with Data/Coef/MStat/Err sheets) and get
a ranked list of binary-variable recommendations: which existing binaries look
safe to drop, which missing calendar months are worth adding, which individual
data points look like data-quality issues, and any regressor whose sign looks
off.

Deploy on Streamlit Community Cloud by pointing it at this file as the main
module (see README.md).
"""

import io

import pandas as pd
import streamlit as st

import analyzer as az

st.set_page_config(page_title="MetrixND Binary Tuning Assistant", layout="wide")

st.title("MetrixND Binary Tuning Assistant")
st.caption(
    "Upload a MetrixND model export and get a ranked, Add: Yes/No list of "
    "binary-variable suggestions — same process used for manual model reviews, "
    "automated."
)

with st.sidebar:
    st.header("Settings")
    p_thresh = st.slider(
        "Significance cutoff (p-value) for flagging existing binaries as removable",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        help="Existing binaries in the real MetrixND fit with a reported p-value above this are flagged for removal.",
    )
    bic_pref = st.radio(
        "Selection rule for adding missing months",
        options=["BIC-parsimony (recommended)", "AIC-only (more aggressive)"],
        index=0,
        help="BIC-parsimony only adds a month if it earns its keep against the extra parameter penalty. "
             "AIC-only is more permissive and will suggest more additions.",
    ) == "BIC-parsimony (recommended)"
    st.divider()
    st.warning(az.AR_CAVEAT, icon="⚠️")

uploaded = st.file_uploader("Upload MetrixND .xlsx export", type=["xlsx"])

if uploaded is None:
    st.info(
        "Waiting for a file. Expected sheets: **Data** (Year, Month, target, then regressor "
        "columns), **Coef** (Variable/Coefficient/StdErr/T-Stat/P-Value), **MStat** (model "
        "diagnostics), **Err** (Year, Month, Actual, Pred, Resid, %Resid, StdResid)."
    )
    st.stop()

try:
    file_bytes = io.BytesIO(uploaded.getvalue())
    file_bytes.name = uploaded.name
    pw = az.load_workbook(file_bytes)
except az.ParseError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Couldn't read this file as a MetrixND export: {e}")
    st.stop()

st.success(f"Loaded **{pw.file_name}** — target variable **{pw.target_name}**, {len(pw.data_df)} observations.")

# ---------------------------------------------------------------------------
# Section 1: current model diagnostics
# ---------------------------------------------------------------------------
st.header("1. Current model diagnostics (as reported by MetrixND)")

m = pw.mstat


def _fmt(key, digits=4):
    v = m.get(key)
    if v is None or not isinstance(v, (int, float)):
        return "—"
    return f"{v:.{digits}f}"


def _pass_fail(key, good_if, digits=4):
    v = m.get(key)
    if v is None or not isinstance(v, (int, float)):
        return "—"
    label = f"{v:.{digits}f}"
    return f"{label} ✅" if good_if(v) else f"{label} ⚠️"


c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", _fmt("R2"))
c1.metric("Adj R²", _fmt("AdjR2"))
c2.metric("AIC", _fmt("AIC"))
c2.metric("BIC", _fmt("BIC"))
c3.metric("Durbin-Watson", _fmt("DW", 3))
c3.metric("MAPE", f"{m.get('MAPE', float('nan')) * 100:.2f}%" if isinstance(m.get("MAPE"), (int, float)) else "—")
c4.metric("Ljung-Box p", _pass_fail("LjungBox_p", lambda v: v > 0.05))
c4.metric("Jarque-Bera p", _pass_fail("JarqueBera_p", lambda v: v > 0.05))

st.caption(
    "✅ = passes the conventional 0.05 threshold (no significant residual autocorrelation / "
    "residuals look normal). ⚠️ = fails — worth investigating before trusting the fit."
)

# ---------------------------------------------------------------------------
# Section 2: current coefficients
# ---------------------------------------------------------------------------
st.header("2. Current coefficients")
if not pw.coef_df.empty:
    show_cols = [c for c in ["Variable", "Coefficient", "StdErr", "TStat", "PValue", "Definition"] if c in pw.coef_df.columns]
    styled = pw.coef_df[show_cols].copy()
    if "PValue" in styled.columns:
        styled["Significant (p<0.05)"] = styled["PValue"].apply(lambda p: "Yes" if pd.notna(p) and p < 0.05 else "No")
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("No Coef sheet data found.")

# ---------------------------------------------------------------------------
# Section 3: recommendations
# ---------------------------------------------------------------------------
st.header("3. Ranked recommendations")

recs, extras = az.build_recommendations(pw, p_thresh=p_thresh, bic_pref=bic_pref)

if not recs:
    st.info("No missing months, no removable binaries, and no sign-check flags — this model looks tight already.")
else:
    rec_rows = []
    for r in recs:
        rec_rows.append({
            "Rank": r.rank,
            "Category": r.category,
            "Variable": r.variable,
            "Add: Yes/No": r.add_yes_no,
            "ΔAIC": f"{r.delta_aic:+.2f}" if r.delta_aic is not None else "—",
            "ΔBIC": f"{r.delta_bic:+.2f}" if r.delta_bic is not None else "—",
            "ΔMAPE (pp)": f"{r.delta_mape:+.3f}" if r.delta_mape is not None else "—",
            "p-value": f"{r.pvalue:.4f}" if r.pvalue is not None else "—",
            "Rationale": r.rationale,
        })
    rec_df = pd.DataFrame(rec_rows)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    csv_bytes = rec_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recommendations as CSV",
        data=csv_bytes,
        file_name=f"{pw.target_name}_recommendations.csv",
        mime="text/csv",
    )

st.caption(
    "\"Add month\" rows come from a plain OLS proxy this app fits itself (candidates aren't in the "
    "real MetrixND model yet, so there's no reported p-value to read). \"Remove existing\" and "
    "\"Investigate\" rows are read directly from MetrixND's own Coef sheet. See the AR/MA caveat "
    "in the sidebar before trusting the proxy numbers."
)

# ---------------------------------------------------------------------------
# Section 4: supporting detail
# ---------------------------------------------------------------------------
st.header("4. Supporting detail")

tab1, tab2, tab3, tab4 = st.tabs(
    ["By-month residual bias", "Forward-selection trace", "Outlier residuals", "Sign checks"]
)

with tab1:
    st.write(
        "Mean residual by calendar month. Months already coded as bare seasonal dummies "
        "should sit at (or near) zero — the missing months are where a real gap would show up."
    )
    mb = extras["month_bias"]
    if not mb.empty:
        st.bar_chart(mb.set_index("MonthName")["MeanResid"])
        st.dataframe(mb, use_container_width=True, hide_index=True)
    else:
        st.info("No Err sheet data found.")

with tab2:
    st.write(
        "Step-by-step trace of the greedy forward selection over missing months. "
        "`Improves` = True means that candidate would have improved the chosen criterion "
        "(BIC or AIC) at that step; only the best-improving candidate is accepted per step."
    )
    ft = extras["forward_trace"]
    if not ft.empty:
        st.dataframe(ft, use_container_width=True, hide_index=True)
        accepted = extras["accepted_months"]
        if accepted:
            st.success(f"Accepted, in order: {', '.join(accepted)}")
        else:
            st.info("No missing month improved the fit enough to be accepted.")
    else:
        st.info("No missing calendar months to test — every month already has a bare seasonal dummy.")

with tab3:
    st.write(
        "Largest-magnitude standardized residuals. A single extreme point (rather than a "
        "whole month or year trending the same way) is usually a data-quality question — "
        "verify the source data before patching it with a one-off dummy."
    )
    st.dataframe(extras["outliers"], use_container_width=True, hide_index=True)

with tab4:
    st.write(
        "Continuous (non-binary) regressors — e.g. weather-driver variables — with a "
        "significant negative coefficient in the real MetrixND fit. Not necessarily wrong, "
        "but worth a sanity check against what the variable is supposed to represent."
    )
    if extras["sign_flags"]:
        for f in extras["sign_flags"]:
            st.warning(f)
    else:
        st.info("No sign-check flags.")

with st.expander("Raw parsed data (debug)"):
    st.write("Regressor columns detected:", pw.regressor_cols)
    st.write("Bare seasonal month dummies found:", pw.month_dummy_cols)
    st.write("Missing months:", [az.MONTH_ABBR[m - 1] for m in pw.missing_months])
    st.dataframe(pw.data_df.head(20), use_container_width=True)
