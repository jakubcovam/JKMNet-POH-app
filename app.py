# streamlit run app.py

import os
import re
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= CONFIG =================
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scenarios"
)

OUTPUTS_SUBDIR = "outputs"
METRICS_SUBDIR = os.path.join("outputs", "metrics")
SETS = ["calib", "valid"]

# =========================================
st.set_page_config(page_title="ANN pro nádrže", layout="wide")
st.title("Přítoky do nádrží povodí Ohře - predikce pomocí ANN")

# =========================================
# Generic helpers
# =========================================
def list_ids(base_dir: str):
    return sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])

def outputs_dir_for_id(id_name: str):
    return os.path.join(BASE_DIR, id_name, OUTPUTS_SUBDIR)

def metrics_dir_for_id(id_name: str):
    return os.path.join(BASE_DIR, id_name, METRICS_SUBDIR)

# =========================================
# METRICS helpers (boxplots)
# =========================================
def parse_run_metric(fname: str, dataset: str):
    m = re.match(rf"{dataset}_metrics_run(\d+)_([A-Za-z0-9]+)\.csv$", fname)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)

def available_metrics(id_name: str):
    mdir = metrics_dir_for_id(id_name)
    metrics = set()
    for ds in SETS:
        for p in glob.glob(os.path.join(mdir, f"{ds}_metrics_run*_*.csv")):
            _, metric = parse_run_metric(os.path.basename(p), ds)
            if metric:
                metrics.add(metric)
    return sorted(metrics)

def load_metric_long(id_name: str, dataset: str, metric: str) -> pd.DataFrame:
    mdir = metrics_dir_for_id(id_name)
    pattern = os.path.join(mdir, f"{dataset}_metrics_run*_{metric}.csv")
    rows = []

    for p in sorted(glob.glob(pattern)):
        run, _ = parse_run_metric(os.path.basename(p), dataset)
        if run is None:
            continue

        df = pd.read_csv(p)
        if df.shape[0] != 1:
            continue

        horizons = [c for c in df.columns if str(c).startswith("h")]
        for h in horizons:
            rows.append({"set": dataset, "run": run, "horizon": h, "value": float(df.loc[0, h])})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["hnum"] = out["horizon"].str.extract(r"h(\d+)").astype(float)
        out = out.sort_values(["set", "hnum", "run"]).drop(columns=["hnum"])
    return out

def metrics_boxplots_two_panels(df_all: pd.DataFrame, metric: str, same_y: bool) -> go.Figure:
    horizons = sorted(df_all["horizon"].unique(), key=lambda x: int(re.findall(r"\d+", x)[0]))

    # Never use shared_yaxes — it merges axes onto one side and drops gridlines
    # on the other panel. Manual range sync below handles the same_y case instead.
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["calib", "valid"],
        shared_yaxes=False,
    )

    colors = {"calib": "#1f77b4", "valid": "#ff7f0e"}

    for col_idx, ds in enumerate(SETS, start=1):
        df_ds = df_all[df_all["set"] == ds]
        for h in horizons:
            vals = df_ds.loc[df_ds["horizon"] == h, "value"].dropna().values
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=h,
                    marker_color=colors[ds],
                    showlegend=False,
                    boxpoints="outliers",
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(
        title_text=f"{metric} napříč spuštěními — kalibrace vs validace",
        title_font=dict(size=25),
        height=500,
        template="plotly_white",
        font=dict(size=20),
    )
    fig.update_xaxes(title_text="Horizon", title_font=dict(size=25), tickfont=dict(size=20))

    # Style BOTH y-axes explicitly — col=1 only would leave the valid panel
    # with the default (smaller) font.
    for col_idx in (1, 2):
        fig.update_yaxes(
            title_text=metric,
            title_font=dict(size=25),
            tickfont=dict(size=20),
            row=1, col=col_idx,
        )

    # Manual range sync: compute a shared range from all values + a small pad.
    # Keeps both axes fully independent (gridlines on both) while showing the
    # same scale when the checkbox is ticked.
    if same_y:
        all_vals = df_all["value"].dropna()
        spread = all_vals.max() - all_vals.min()
        pad = spread * 0.05 if spread > 0 else 0.5
        shared_range = [all_vals.min() - pad, all_vals.max() + pad]
        for col_idx in (1, 2):
            fig.update_yaxes(range=shared_range, row=1, col=col_idx)

    # enlarge subplot title annotations
    for ann in fig.layout.annotations:
        ann.font = dict(size=20)

    return fig

# =========================================
# TIME SERIES helpers (real + ensemble preds)
# =========================================
def find_horizons_from_real(id_name: str, dataset: str):
    real_path = os.path.join(outputs_dir_for_id(id_name), f"{dataset}_real.csv")
    if not os.path.exists(real_path):
        return []
    df = pd.read_csv(real_path)
    horizons = [c for c in df.columns if str(c).startswith("h")]
    try:
        horizons = sorted(horizons, key=lambda x: int(re.findall(r"\d+", x)[0]))
    except Exception:
        horizons = list(horizons)
    return horizons

def load_real_series(id_name: str, dataset: str, horizon: str) -> pd.Series:
    path = os.path.join(outputs_dir_for_id(id_name), f"{dataset}_real.csv")
    df = pd.read_csv(path)
    if horizon not in df.columns:
        raise ValueError(f"{horizon} not found in {path}")
    return df[horizon]

def load_pred_ensemble(id_name: str, dataset: str, horizon: str) -> pd.DataFrame:
    """
    Loads all runs: <dataset>_pred_1.csv, <dataset>_pred_2.csv, ...
    Returns DataFrame shape (T, n_runs) with columns run_1, run_2, ...
    """
    outdir = outputs_dir_for_id(id_name)
    pattern = os.path.join(outdir, f"{dataset}_pred_*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        return pd.DataFrame()

    cols = {}
    for p in paths:
        m = re.search(r"_pred_(\d+)\.csv$", os.path.basename(p))
        if not m:
            continue
        run = int(m.group(1))
        df = pd.read_csv(p)
        if horizon not in df.columns:
            continue
        cols[f"run_{run}"] = df[horizon].astype(float).reset_index(drop=True)

    if not cols:
        return pd.DataFrame()

    ens = pd.DataFrame(cols)
    ens = ens.reindex(sorted(ens.columns, key=lambda c: int(c.split("_")[1])), axis=1)
    return ens

def build_timeseries_figure(
    id_name: str,
    horizon: str,
    show_members: bool,
    show_band: bool,
) -> go.Figure:
    """
    Returns a single Plotly figure with two vertically stacked subplots
    (calib on top, valid on bottom). Both subplots share their x-axis range
    independently (no shared_xaxes so zooming one doesn't lock the other).
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"calib — horizon {horizon}", f"valid — horizon {horizon}"],
        vertical_spacing=0.22,
    )

    MEMBER_COLOR = "rgba(180,180,180,0.55)"
    MEAN_COLOR   = "#e07b25"
    BAND_COLOR   = "rgba(224,123,37,0.20)"
    REAL_COLOR   = "#111111"

    for row_idx, ds in enumerate(SETS, start=1):
        real_path = os.path.join(outputs_dir_for_id(id_name), f"{ds}_real.csv")

        # ---- missing real file → blank panel
        if not os.path.exists(real_path):
            fig.add_annotation(
                text=f"{ds}: missing real file",
                xref="paper", yref="paper",
                x=0.5, y=(0.85 if row_idx == 1 else 0.15),
                showarrow=False, font=dict(size=20, color="red"),
            )
            continue

        real = load_real_series(id_name, ds, horizon)
        ens  = load_pred_ensemble(id_name, ds, horizon)
        x    = np.arange(len(real))

        # ---- ensemble members (light grey, grouped under one legend entry)
        if show_members and not ens.empty:
            first_member = True
            for c in ens.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=ens[c].values,
                        mode="lines",
                        line=dict(color=MEMBER_COLOR, width=0.8),
                        name="Runs" if first_member else None,
                        legendgroup="members",
                        showlegend=(first_member and row_idx == 1),
                        hoverinfo="skip",
                    ),
                    row=row_idx, col=1,
                )
                first_member = False

        # ---- ±2 std band + mean
        if not ens.empty:
            mean = ens.mean(axis=1).values
            std2 = 2.0 * ens.std(axis=1).values

            if show_band:
                # upper boundary — legend entry lives here so give it a
                # visible line color; that color is what shows in the swatch.
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean + std2,
                        mode="lines",
                        line=dict(width=1.5, color=MEAN_COLOR),
                        name="Pred ± 2 std",
                        legendgroup="band",
                        showlegend=(row_idx == 1),
                        hoverinfo="skip",
                    ),
                    row=row_idx, col=1,
                )
                # lower boundary — this trace owns the fill; it inherits the
                # legend group but is hidden from the legend.
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean - std2,
                        mode="lines",
                        line=dict(width=1.5, color=MEAN_COLOR),
                        fill="tonexty",
                        fillcolor=BAND_COLOR,
                        name="Pred ± 2 std",
                        legendgroup="band",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_idx, col=1,
                )

            # mean line
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    line=dict(color=MEAN_COLOR, width=2.5),
                    name="Pred mean",
                    legendgroup="mean",
                    showlegend=(row_idx == 1),
                    hovertemplate="mean: %{y:.3f}<extra></extra>",
                ),
                row=row_idx, col=1,
            )

        # ---- real data
        fig.add_trace(
            go.Scatter(
                x=x,
                y=real.values,
                mode="lines",
                line=dict(color=REAL_COLOR, width=1.2),
                name="Real",
                legendgroup="real",
                showlegend=(row_idx == 1),
                hovertemplate="real: %{y:.3f}<extra></extra>",
            ),
            row=row_idx, col=1,
        )

        fig.update_yaxes(
            title_text="Přítok do VN",
            title_font=dict(size=25),
            tickfont=dict(size=20),
            row=row_idx, col=1,
        )
        fig.update_xaxes(
            title_text="Time index",
            title_font=dict(size=25),
            tickfont=dict(size=20),
            row=row_idx, col=1,
        )

    # enlarge subplot title annotations (created by make_subplots)
    for ann in fig.layout.annotations:
        ann.font = dict(size=20)

    fig.update_layout(
        title_text=f"Realné vs ensemble predikce — {id_name} — {horizon}",
        title_font=dict(size=25),
        height=750,
        template="plotly_white",
        font=dict(size=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=20),
        ),
        hovermode="x unified",
    )

    return fig


# =========================================
# Sidebar UI
# =========================================
with st.sidebar:
    st.header("Nastavení")

    ids = list_ids(BASE_DIR)
    if not ids:
        st.error("No ID folders found next to app.py.")
        st.stop()

    id_name = st.selectbox("Nádrž", ids)

    view = st.radio(
        "Zobrazení",
        ["Time series (real + ensemble preds)", "Metrics (boxplots)"],
        index=0,
    )

    if view.startswith("Time"):
        hz = find_horizons_from_real(id_name, "calib")
        if not hz:
            hz = find_horizons_from_real(id_name, "valid")
        if not hz:
            st.error("No horizons found (missing calib_real.csv / valid_real.csv or no h-columns).")
            st.stop()

        horizon = st.selectbox("Horizon", hz)
        show_members = st.checkbox("Zobraz každé spuštení (šedá linie)", value=True)
        show_band    = st.checkbox("Ukaž průměr ± 2std", value=True)

    else:
        metrics = available_metrics(id_name)
        if not metrics:
            st.error(f"No metric files found in: {metrics_dir_for_id(id_name)}")
            st.stop()

        metric = st.selectbox("Metrika", metrics)
        same_y = st.checkbox("Použij stejnou y-osu pro calib a valid", value=False)


# =========================================
# Main
# =========================================
if view.startswith("Time"):
    missing = []
    for ds in SETS:
        if not os.path.exists(os.path.join(outputs_dir_for_id(id_name), f"{ds}_real.csv")):
            missing.append(f"{ds}_real.csv")

    if missing:
        st.warning("Missing files: " + ", ".join(missing) + " (that panel will be empty).")

    fig = build_timeseries_figure(id_name, horizon, show_members, show_band)
    st.plotly_chart(fig, width="stretch")

else:
    df_calib = load_metric_long(id_name, "calib", metric)
    df_valid  = load_metric_long(id_name, "valid", metric)
    df_all    = pd.concat([df_calib, df_valid], ignore_index=True)

    if df_all.empty:
        st.error("No metrics loaded for this selection.")
        st.stop()

    fig = metrics_boxplots_two_panels(df_all, metric, same_y=same_y)
    st.plotly_chart(fig, width="stretch")