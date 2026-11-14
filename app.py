"""
EarthLens - Urban Growth Detective  (NASA Space Apps Cairo 2026)
Challenge: Earth System Trend Detective

Run:   pip install -r requirements.txt
       streamlit run app.py

The two CSV files live in data/ (or upload them from the sidebar).
"""
import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="EarthLens - Urban Growth Detective",
                   page_icon="🛰️", layout="wide")

# =====================================================================
# STEP 0 - CONSTANTS
# =====================================================================
METRICS = {
    "Confident_Built_km2": "Confident Built (conservative, recommended)",
    "Built_Probability_km2": "Built Probability (includes mixed pixels)",
    "Modal_Built_km2": "Modal Built (annual only)",
}
# Dynamic World class ids -> names (used by the 'label' column)
CLASS_NAMES = {0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation",
               4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare",
               8: "snow_and_ice"}
FEATURES = ["water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice",
            "B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "BSI"]


# =====================================================================
# STEP 1 - LOAD + CLEAN DATA
# The files use -999 as "not applicable". We convert it to NaN so it can
# never be mistaken for a real number inside a mean / regression.
# =====================================================================
def _find_default(pattern):
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (os.path.join(here, "data"), here):
        hits = sorted(glob.glob(os.path.join(folder, pattern)))
        if hits:
            return hits[0]
    return None


@st.cache_data(show_spinner=False)
def load_growth(src):
    df = pd.read_csv(src).replace(-999, np.nan)
    annual = df[df.Record_Type == "annual_growth"].copy()
    annual["Year"] = annual["Year"].astype(int)
    monthly = df[df.Record_Type == "monthly_growth"].copy()
    monthly["Year"] = monthly["Year"].astype(int)
    monthly["Month"] = monthly["Month"].astype(int)
    monthly["Date"] = pd.to_datetime(monthly["Date"] + "-01")
    # decimal time in years (2016.0, 2016.083, ...) -> slope comes out in km2 / YEAR
    monthly["t"] = monthly["Year"] + (monthly["Month"] - 1) / 12
    persist = df[df.Record_Type == "persistent_conversion"].copy()
    return annual.reset_index(drop=True), monthly.reset_index(drop=True), persist.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ml(src):
    df = pd.read_csv(src)
    df["label_name"] = df["label"].map(CLASS_NAMES)
    cls = df[df.task == "classification"].copy()            # land-cover samples 2023-2025
    sus = df[df.task == "susceptibility"].copy()            # did a pixel become built?
    sus["converted"] = sus["converted"].astype(int)
    sus["period"] = sus["Baseline_Year"].astype(int).astype(str) + " -> " + sus["Target_Year"].astype(int).astype(str)
    return cls, sus


# =====================================================================
# STEP 2 - TREND STATISTICS  (the "b value" lives here)
# =====================================================================
def ols_trend(t, y, alpha=0.05):
    """Ordinary least squares  y = a + b*t.
    b = slope (km2 per year), with CI and p-value for H0: b = 0."""
    r = stats.linregress(t, y)
    n = len(t)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 2)
    return dict(b=r.slope, a=r.intercept, b_lo=r.slope - tcrit * r.stderr,
                b_hi=r.slope + tcrit * r.stderr, p=r.pvalue, r2=r.rvalue ** 2,
                stderr=r.stderr, n=n)


def mann_kendall(y):
    """Classic Mann-Kendall test (with tie correction).
    S > 0 -> upward trend,  S < 0 -> downward trend.
    Returns S, tau (-1..1), Z and two-sided p (normal approximation)."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    s = sum(np.sign(y[k + 1:] - y[k]).sum() for k in range(n - 1))
    _, cnt = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(cnt * (cnt - 1) * (2 * cnt + 5))) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(S=s, var=var, z=z, p=p, tau=s / (n * (n - 1) / 2), n=n)


def sen_slope(t, y, alpha=0.05):
    """Theil-Sen slope: median of all pairwise slopes (robust to outliers)."""
    sl, ic, lo, hi = stats.theilslopes(y, t, alpha=1 - alpha)
    return dict(b=sl, a=ic, b_lo=lo, b_hi=hi)


def seasonal_mann_kendall(df, col, season_col="Month", time_col="t"):
    """Hirsch seasonal Mann-Kendall.
    Mann-Kendall is run SEPARATELY inside each calendar month (Jan vs Jan vs Jan...),
    then S and variance are summed. This removes the seasonal cycle by design.
    The slope is the median of all within-month pairwise slopes (seasonal Sen slope)."""
    S, V, slopes = 0.0, 0.0, []
    for _, g in df.groupby(season_col):
        g = g.dropna(subset=[col]).sort_values(time_col)
        if len(g) < 3:
            continue
        mk = mann_kendall(g[col].values)
        S += mk["S"]
        V += mk["var"]
        tt, yy = g[time_col].values, g[col].values
        for i in range(len(g) - 1):
            slopes.extend(((yy[i + 1:] - yy[i]) / (tt[i + 1:] - tt[i])).tolist())
    z = 0.0 if S == 0 else (S - np.sign(S)) / np.sqrt(V)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slopes = np.sort(np.array(slopes))
    N = len(slopes)
    C = 1.96 * np.sqrt(V)                      # rank-based 95% CI for the Sen slope
    lo = slopes[int(np.clip(np.floor((N - C) / 2), 0, N - 1))]
    hi = slopes[int(np.clip(np.ceil((N + C) / 2), 0, N - 1))]
    return dict(S=S, z=z, p=p, b=float(np.median(slopes)), b_lo=float(lo), b_hi=float(hi))


def verdict(p, alpha=0.05):
    return "Significant ✅" if p < alpha else "Not significant ❌"


# =====================================================================
# STEP 3 - PREPARE THE TIME SERIES (fix the problems we found)
# =====================================================================
def prepare_monthly(monthly, col, normalize, full_area):
    """Problem 1: some months have fewer valid pixels (cloud / no data) -> area looks smaller.
       Fix: convert to a FRACTION of valid area, then rescale to the full AOI area.
       Problem 2: strong yearly cycle (winter high, summer low) -> fake trend if ignored.
       Fix: subtract each calendar month's average (anomaly) + use seasonal Mann-Kendall."""
    m = monthly.dropna(subset=[col]).copy()
    m["value"] = m[col] * (full_area / m["Valid_Observation_km2"]) if normalize else m[col]
    m["climatology"] = m.groupby("Month")["value"].transform("mean")
    m["anomaly"] = m["value"] - m["climatology"]
    return m


# =====================================================================
# STEP 4 - MACHINE LEARNING (predict which pixels will turn into built-up)
# =====================================================================
def make_model(name):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


@st.cache_resource(show_spinner="Training model...")
def train_model(sus, model_name, features, _key):
    """Train on 2017-2021 transitions, test on the held-out 2023->2025 transition.
    Splitting by TIME (not randomly) tells us if the model works on the future."""
    tr = sus[sus.split == "training"]
    te = sus[sus.split == "temporal_test_2023_2025"]
    model = make_model(model_name).fit(tr[list(features)], tr["converted"])
    proba = model.predict_proba(te[list(features)])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = te["converted"].values
    fpr, tpr, _ = roc_curve(y, proba)
    perm = permutation_importance(model, te[list(features)], y, n_repeats=8,
                                  random_state=42, scoring="roc_auc")
    res = dict(auc=roc_auc_score(y, proba), acc=accuracy_score(y, pred),
               prec=precision_score(y, pred), rec=recall_score(y, pred), f1=f1_score(y, pred),
               cm=confusion_matrix(y, pred), fpr=fpr, tpr=tpr,
               perm=pd.Series(perm.importances_mean, index=list(features)).sort_values(),
               n_train=len(tr), n_test=len(te))
    return model, res


# =====================================================================
# SIDEBAR - DATA INPUT
# =====================================================================
st.sidebar.title("🛰️ EarthLens")
st.sidebar.caption("Urban Growth Detective · NASA Space Apps Cairo 2026")
up_growth = st.sidebar.file_uploader("Urban Growth Analysis CSV", type="csv")
up_ml = st.sidebar.file_uploader("Complete Dataset CSV", type="csv")
src_growth = up_growth or _find_default("EarthLens_Urban_Growth_Analysis*.csv")
src_ml = up_ml or _find_default("EarthLens_Complete_Dataset*.csv")
if not (src_growth and src_ml):
    st.warning("Put both CSV files in the data/ folder or upload them from the sidebar.")
    st.stop()

annual, monthly, persist = load_growth(src_growth)
cls, sus = load_ml(src_ml)
FULL_AREA = float(monthly["Valid_Observation_km2"].max())   # full AOI area (~1067 km2)
ALPHA = st.sidebar.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)

st.title("Urban Growth Detective")
st.caption("Where is the city growing, how fast, and is the trend statistically real?")

tab_over, tab_ann, tab_mon, tab_src, tab_map, tab_ml = st.tabs(
    ["1 · Overview", "2 · Annual trend", "3 · Monthly trend", "4 · Where from?",
     "5 · Conversion map", "6 · Prediction model"])

# =====================================================================
# TAB 1 - OVERVIEW
# =====================================================================
with tab_over:
    st.subheader("What is in the data?")
    a_last = annual.sort_values("Year").iloc[-1]
    a_first = annual.sort_values("Year").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Study area", f"{FULL_AREA:,.0f} km²", "East Cairo (New Cairo → NAC)")
    c2.metric("Built-up (confident)", f"{a_last.Confident_Built_km2:,.0f} km²",
              f"{a_last.Confident_Built_km2 - a_first.Confident_Built_km2:+.0f} km² since {int(a_first.Year)}")
    c3.metric("Monthly records", len(monthly), f"{monthly.Date.min():%Y-%m} → {monthly.Date.max():%Y-%m}")
    c4.metric("ML samples", f"{len(cls) + len(sus):,}", f"{len(sus):,} conversion + {len(cls):,} land-cover")
    st.markdown("""
| File / table | What it contains | Used for |
|---|---|---|
| **annual_growth** (9 rows) | Built-up km² per year, 3 definitions | Main trend + significance test |
| **monthly_growth** (130 rows) | Built-up km² per month + number of satellite images + valid area | Seasonal analysis, early signal |
| **persistent_conversion** | km² that changed class → built and *stayed* built 2 years later | "Built on what?" |
| **susceptibility** (3,600 pts) | Pixel features at baseline + `converted` 0/1 two years later | Train / test the model |
| **classification** (3,477 pts) | Land-cover samples 2023-2025 (no conversion label) | Score future risk |
""")
    st.subheader("Data-quality checks")
    q1, q2, q3 = st.columns(3)
    q1.metric("Months with reduced valid area", int((monthly.Valid_Observation_km2 < FULL_AREA * 0.999).sum()))
    q2.metric("Missing values after -999 → NaN",
              int(monthly[["Built_Probability_km2", "Confident_Built_km2"]].isna().sum().sum()))
    q3.metric("Partial year", f"{int(monthly.Year.max())} (to {monthly.Date.max():%b})")
    st.info("Fixes applied automatically: (1) -999 → NaN, (2) valid-area normalisation, "
            "(3) seasonal cycle removed, (4) partial year excluded from annual statistics.")

# =====================================================================
# TAB 2 - ANNUAL TREND
# =====================================================================
with tab_ann:
    st.subheader("Annual trend: rate of change (b) and significance")
    cA, cB = st.columns([1, 2])
    with cA:
        metric_a = st.selectbox("Variable", list(METRICS), format_func=METRICS.get, key="ma")
        y0, y1 = st.slider("Years", int(annual.Year.min()), int(annual.Year.max()),
                           (int(annual.Year.min()), int(annual.Year.max())))
        horizon = st.slider("Projection until year", int(annual.Year.max()), 2035, 2030)
    d = annual[(annual.Year >= y0) & (annual.Year <= y1)].dropna(subset=[metric_a])
    t, y = d.Year.values.astype(float), d[metric_a].values
    if len(d) < 4:
        st.warning("Select at least 4 years.")
        st.stop()
    ols, sen, mk = ols_trend(t, y, ALPHA), sen_slope(t, y, ALPHA), mann_kendall(y)
    pct = 100 * ols["b"] / y.mean()
    with cA:
        st.metric("b  (OLS slope)", f"{ols['b']:+.2f} km²/yr", f"95% CI [{ols['b_lo']:.2f}, {ols['b_hi']:.2f}]")
        st.metric("Sen slope", f"{sen['b']:+.2f} km²/yr", f"95% CI [{sen['b_lo']:.2f}, {sen['b_hi']:.2f}]")
        st.metric("Relative growth", f"{pct:+.1f} % of mean / yr")
        st.metric("Mann-Kendall", verdict(mk["p"], ALPHA), f"τ={mk['tau']:.2f}  Z={mk['z']:.2f}  p={mk['p']:.5f}")
        st.caption(f"OLS: R²={ols['r2']:.3f}, p={ols['p']:.2e}, n={ols['n']}")
    # confidence band + projection
    tf = np.arange(y0, horizon + 1, dtype=float)
    fit = ols["a"] + ols["b"] * tf
    resid_sd = np.sqrt(np.sum((y - (ols["a"] + ols["b"] * t)) ** 2) / (len(t) - 2))
    pi = stats.t.ppf(1 - ALPHA / 2, len(t) - 2) * resid_sd * np.sqrt(
        1 + 1 / len(t) + (tf - t.mean()) ** 2 / np.sum((t - t.mean()) ** 2))
    fig = go.Figure()
    fig.add_scatter(x=tf, y=fit + pi, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=tf, y=fit - pi, fill="tonexty", line=dict(width=0), name="Prediction interval",
                    fillcolor="rgba(255,140,0,0.2)")
    fig.add_scatter(x=tf, y=fit, name="Linear trend", line=dict(dash="dash", color="orange"))
    fig.add_scatter(x=t, y=y, mode="markers+lines", name="Observed", marker=dict(size=9))
    fig.update_layout(height=430, xaxis_title="Year", yaxis_title="Built-up area (km²)",
                      legend=dict(orientation="h"), margin=dict(t=20))
    cB.plotly_chart(fig, width="stretch")
    cB.caption(f"If the trend continues, built-up area reaches about **{fit[-1]:.0f} km²** by {horizon} "
               f"(range {fit[-1] - pi[-1]:.0f}–{fit[-1] + pi[-1]:.0f}). A linear projection, not a forecast.")
    d2 = d[["Year", metric_a]].copy()
    d2["Change vs previous year (km²)"] = d2[metric_a].diff()
    d2["Growth (%)"] = d2[metric_a].pct_change() * 100
    st.dataframe(d2.round(2), hide_index=True, width="stretch")

# =====================================================================
# TAB 3 - MONTHLY TREND
# =====================================================================
with tab_mon:
    st.subheader("Monthly trend: remove the seasonal cycle first")
    cA, cB = st.columns([1, 2])
    with cA:
        metric_m = st.selectbox("Variable", ["Confident_Built_km2", "Built_Probability_km2"],
                                format_func=METRICS.get, key="mm")
        norm = st.checkbox("Normalise by valid area", True,
                           help="Some months have fewer valid pixels, which would look like less city.")
        drop_partial = st.checkbox(f"Exclude partial year {int(monthly.Year.max())}", False)
    m = prepare_monthly(monthly, metric_m, norm, FULL_AREA)
    if drop_partial:
        m = m[m.Year < monthly.Year.max()]
    naive_mk = mann_kendall(m["value"].values)
    smk = seasonal_mann_kendall(m, "value")
    ols_m = ols_trend(m["t"].values, m["anomaly"].values, ALPHA)
    with cA:
        st.metric("b  (seasonal Sen slope)", f"{smk['b']:+.2f} km²/yr", f"95% CI [{smk['b_lo']:.2f}, {smk['b_hi']:.2f}]")
        st.metric("Seasonal Mann-Kendall", verdict(smk["p"], ALPHA), f"Z={smk['z']:.2f}  p<1e-10")
        st.metric("b  (OLS on anomalies)", f"{ols_m['b']:+.2f} km²/yr")
        st.caption(f"Naive Mann-Kendall on raw months: Z={naive_mk['z']:.1f}. "
                   "It ignores the seasonal cycle, so it is shown only for comparison.")
    f1 = go.Figure()
    f1.add_scatter(x=m.Date, y=m["value"], name="Raw monthly", line=dict(color="lightgray"))
    f1.add_scatter(x=m.Date, y=m["value"].rolling(12, center=True).mean(), name="12-month mean",
                   line=dict(color="royalblue", width=3))
    f1.add_scatter(x=m.Date, y=m["anomaly"], name="Anomaly (seasonality removed)", line=dict(color="orange"))
    f1.update_layout(height=420, yaxis_title="km²", legend=dict(orientation="h"), margin=dict(t=20))
    cB.plotly_chart(f1, width="stretch")
    c1, c2 = st.columns(2)
    clim = m.groupby("Month")["value"].mean().reset_index()
    fs = px.bar(clim, x="Month", y="value", title="Average built-up area by calendar month (seasonal cycle)")
    fs.update_layout(height=320, yaxis_title="km²")
    c1.plotly_chart(fs, width="stretch")
    yearly = m.groupby("Year").agg(value=("value", "mean"), images=("Satellite_Images", "sum")).reset_index()
    fy = px.bar(yearly, x="Year", y="images", title="Satellite images used per year")
    fy.update_layout(height=320)
    c2.plotly_chart(fy, width="stretch")
    st.caption("Why this matters: the same city looks 2.5× bigger in January than in June. That is an "
               "artefact of the classifier / vegetation cycle, not construction.")

# =====================================================================
# TAB 4 - WHERE FROM
# =====================================================================
with tab_src:
    st.subheader("What did the city grow on?")
    p = persist.dropna(subset=["Persistent_Conversion_km2"]).copy()
    by_cls = p.groupby("Source_Class")["Persistent_Conversion_km2"].sum().sort_values(ascending=False).reset_index()
    tot = by_cls["Persistent_Conversion_km2"].sum()
    c1, c2 = st.columns(2)
    fig = px.bar(by_cls, x="Source_Class", y="Persistent_Conversion_km2",
                 title="Land that became (and stayed) built-up, 2017-2025")
    fig.update_layout(height=380, yaxis_title="km²")
    c1.plotly_chart(fig, width="stretch")
    p["Period"] = p["Baseline_Year"].astype(int).astype(str) + "→" + p["Confirmation_Year"].astype(int).astype(str)
    pp = p.groupby(["Period", "Source_Class"])["Persistent_Conversion_km2"].sum().reset_index()
    fig2 = px.bar(pp[pp.Persistent_Conversion_km2 > 0], x="Period", y="Persistent_Conversion_km2",
                  color="Source_Class", title="By 2-year period")
    fig2.update_layout(height=380, yaxis_title="km²")
    c2.plotly_chart(fig2, width="stretch")
    bare = by_cls.set_index("Source_Class")["Persistent_Conversion_km2"].get("bare", 0)
    crops = by_cls.set_index("Source_Class")["Persistent_Conversion_km2"].get("crops", 0)
    st.success(f"**{100 * bare / tot:.1f}%** of confirmed new built-up area came from **bare/desert land** "
               f"({bare:.1f} km²). Cropland contributed only {crops:.3f} km².  "
               "In this study area the city expands into the desert, not onto farmland.")

# =====================================================================
# TAB 5 - CONVERSION MAP
# =====================================================================
with tab_map:
    st.subheader("Where did conversion happen?")
    periods = sorted(sus.period.unique())
    c1, c2 = st.columns([1, 3])
    with c1:
        per = st.selectbox("Period (baseline → target)", periods, index=len(periods) - 1)
        show = st.radio("Show", ["Converted to built-up", "Stayed the same", "Both"], index=2)
        feat = st.selectbox("Colour intensity by", ["built", "bare", "NDVI", "NDBI"], index=0)
    sp = sus[sus.period == per].copy()
    if show == "Converted to built-up":
        sp = sp[sp.converted == 1]
    elif show == "Stayed the same":
        sp = sp[sp.converted == 0]
    sp["status"] = np.where(sp.converted == 1, "Converted to built", "No change")
    fig = px.scatter_map(sp, lat="latitude", lon="longitude", color="status", zoom=9.2, height=560,
                         color_discrete_map={"Converted to built": "#e63946", "No change": "#457b9d"},
                         hover_data={feat: ":.3f", "latitude": False, "longitude": False},
                         map_style="open-street-map", opacity=0.7)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02))
    c2.plotly_chart(fig, width="stretch")
    c1.caption("Share of sampled pixels converted per period is fixed at 50% by the balanced sampling, "
               "so it is NOT a real conversion rate. Use the annual km² numbers for rates.")
    c1.write("Note: this map shows sampled pixels, not satellite images. For true before/after imagery "
             "add Landsat/Sentinel tiles from Google Earth Engine or NASA GIBS.")

# =====================================================================
# TAB 6 - MODEL
# =====================================================================
with tab_ml:
    st.subheader("Which pixels are likely to become built-up?")
    st.markdown("**Task:** binary classification. Input = land-cover probabilities + satellite bands + "
                "indices of a pixel at the *baseline* year. Output = probability it is built-up "
                "**2 years later**.  Train: 2017→2021 periods. Test: the unseen 2023→2025 period.")
    c1, c2 = st.columns([1, 3])
    with c1:
        mname = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        use_xy = st.checkbox("Use latitude/longitude", False,
                             help="Off = the model must learn from land/spectral properties only.")
        feats = tuple(FEATURES + (["longitude", "latitude"] if use_xy else []))
    model, r = train_model(sus, mname, feats, (mname, feats))
    with c1:
        st.metric("ROC-AUC (test)", f"{r['auc']:.3f}")
        st.metric("Accuracy", f"{r['acc']:.3f}")
        st.metric("Precision / Recall", f"{r['prec']:.2f} / {r['rec']:.2f}")
        st.metric("F1", f"{r['f1']:.3f}")
        st.caption(f"Train n={r['n_train']}, test n={r['n_test']}. AUC 0.5 = coin flip, 1.0 = perfect.")
    with c2:
        k1, k2 = st.columns(2)
        fr = go.Figure()
        fr.add_scatter(x=r["fpr"], y=r["tpr"], name=f"{mname} (AUC={r['auc']:.2f})")
        fr.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash"))
        fr.update_layout(title="ROC curve", xaxis_title="False positive rate",
                         yaxis_title="True positive rate", height=340, legend=dict(orientation="h"))
        k1.plotly_chart(fr, width="stretch")
        cm = px.imshow(r["cm"], text_auto=True, x=["Pred: no", "Pred: yes"], y=["True: no", "True: yes"],
                       color_continuous_scale="Blues", title="Confusion matrix")
        cm.update_layout(height=340, coloraxis_showscale=False)
        k2.plotly_chart(cm, width="stretch")
        imp = r["perm"].tail(12)
        fi = px.bar(imp, orientation="h", title="Permutation importance (drop in AUC when feature is shuffled)")
        fi.update_layout(height=380, showlegend=False, xaxis_title="AUC drop", yaxis_title="")
        st.plotly_chart(fi, width="stretch")
    st.markdown("---")
    st.subheader("Risk map: conversion probability for the 2023-2025 land-cover samples")
    yr = st.select_slider("Sample year", sorted(cls.year.unique()))
    cy = cls[(cls.year == yr)].copy()
    cy["risk"] = model.predict_proba(cy[list(feats)])[:, 1]
    only_non_built = st.checkbox("Hide pixels that are already built-up", True)
    if only_non_built:
        cy = cy[cy.label != 6]
    fm = px.scatter_map(cy, lat="latitude", lon="longitude", color="risk", zoom=9.2, height=520,
                        color_continuous_scale="YlOrRd", range_color=(0, 1),
                        hover_data={"label_name": True, "risk": ":.2f", "latitude": False, "longitude": False},
                        map_style="open-street-map", opacity=0.75)
    fm.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fm, width="stretch")
    st.caption("Honest note: the strongest signal is the baseline 'built' / 'bare' probability, i.e. pixels "
               "that are already partly urban or at the desert edge are the ones that convert. Training classes are "
               "balanced 50/50, so read the map as a RELATIVE risk ranking, not an exact probability.")
    st.subheader("What-if: try a single pixel")
    w1, w2, w3, w4 = st.columns(4)
    med = sus[list(feats)].median()
    row = med.copy()
    row["built"] = w1.slider("built probability", 0.0, 1.0, float(med["built"]), 0.01)
    row["bare"] = w2.slider("bare probability", 0.0, 1.0, float(med["bare"]), 0.01)
    row["NDVI"] = w3.slider("NDVI", -1.0, 1.0, float(med["NDVI"]), 0.01)
    row["NDBI"] = w4.slider("NDBI", -1.0, 1.0, float(med["NDBI"]), 0.01)
    pr = model.predict_proba(pd.DataFrame([row]))[0, 1]
    st.metric("Probability this pixel becomes built-up in 2 years", f"{pr * 100:.1f} %")
