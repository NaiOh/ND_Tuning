# ND Binary Tuning Assistant

A Streamlit app that automates the model-review process used throughout this
project: upload a ND xlsx export, get back a ranked, **Add: Yes/No** list
of binary-variable suggestions — which existing binaries look safe to drop,
which missing calendar months are worth adding, which individual residuals
look like data-quality issues rather than real seasonal effects, and any
regressor whose coefficient sign looks off.

## What it expects

A `.xlsx` file exported from ND with (at least) these sheets:

- **Data** — columns `Year, Month, <target>, <regressor 1>, <regressor 2>, ...`
- **Coef** — `Variable, Coefficient, StdErr, T-Stat, P-Value, ...` (as ND reports it)
- **MStat** — model diagnostics (R², AIC, BIC, Durbin-Watson, Ljung-Box, Jarque-Bera, MAPE, ...)
- **Err** — `Year, Month, Actual, Pred, Resid, %Resid, StdResid`

This matches the standard ND model export format used for ResUPC,
ComUPC, and the SalesShr models.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`),
and upload an `.xlsx` file.

## Deploying on Streamlit Community Cloud

1. Create a new GitHub repository (public or private) and push these three
   files to it: `app.py`, `analyzer.py`, `requirements.txt` (and this
   `README.md` if you want).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick the repository/branch, and set **Main file path**
   to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically — no
   other configuration is needed.
5. Once deployed, share the app URL — anyone with the link can upload a file
   and get the report back; nothing is stored server-side beyond the current
   session.

## Important limitation — read before trusting results

ND's real model can include **AR(1)/MA(1)** autocorrelation correction.
This app has no access to that internal fitting process — it only sees the
Data/Coef/MStat/Err sheets you export. For any **candidate** variable not
already in the model (e.g. a missing month being tested), the app fits a
plain OLS proxy to estimate AIC/BIC/significance. That proxy can be
**misleading when AR(1) is close to 1** (near unit-root): the real model may
already be absorbing a persistent shift through the AR term, while the
static OLS proxy sees the same shift and (wrongly) attributes it to whatever
binary happens to correlate with it.

Practical rule of thumb: treat every "Add: Yes" suggestion as a hypothesis to
test by actually adding it in ND and re-fitting — not as a final
answer. The app's job is to narrow down what to try, not to replace
re-fitting the real model.

For variables **already in the model** (existing binaries flagged for
removal, and the sign-check flags), the app reads the real reported
coefficients/p-values straight from the Coef sheet, so those are as
trustworthy as ND's own fit.

## Files

- `app.py` — Streamlit UI (upload, diagnostics, coefficients, recommendations, charts)
- `analyzer.py` — parsing + statistics (pure Python/pandas/numpy, no Streamlit
  dependency — can be imported and tested independently)
- `requirements.txt` — pinned minimum versions for Streamlit Cloud
