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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model catalogue: display label → folder prefix (must match your folder names)
MODEL_TYPES = {
    "MLP": {
        "8":        "scenarios_MLP_8",
        "10-10":    "scenarios_MLP_10-10",
        "50-50":    "scenarios_MLP_50-50",
    },
    "LSTM": {
        "30 days (P, Tavg, AET, Q)": "scenarios_LSTM_30",
        "14 days (P, AET, Q)":       "scenarios_LSTM_14",
        "10 days (P, Q)":            "scenarios_LSTM_10",
    },
}

# Flat list of all (label, folder) pairs used in comparison views
ALL_MODELS = [
    (f"MLP {v}", folder)
    for v, folder in MODEL_TYPES["MLP"].items()
] + [
    (f"LSTM {v}", folder)
    for v, folder in MODEL_TYPES["LSTM"].items()
]

# Colour palette — one per model (6 models max)
MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]

# For each known metric: True = higher is better (green), False = lower is better (green).
# Unknown metrics default to False (lower = better) — add new ones here as needed.
METRIC_HIGHER_IS_BETTER = {
    "KGE":   True,
    "NS":    True,   # Nash-Sutcliffe efficiency
    "NSE":   True,   # alias
    "PI":    True,   # persistence index
    "R2":    True,
    "CORR":  True,
    "MSE":   False,
    "RMSE":  False,
    "RSR":   False,  # RMSE/std of observations
    "PBIAS": False,  # % bias — closer to 0 is better
    "MAE":   False,
}

def metric_higher_is_better(metric: str) -> bool:
    return METRIC_HIGHER_IS_BETTER.get(metric.upper(), False)

OUTPUTS_SUBDIR = "outputs"
METRICS_SUBDIR = os.path.join("outputs", "metrics")
SETS           = ["calib", "valid"]

FONT = dict(size=18)
TITLE_FONT = dict(size=22)
AXIS_FONT  = dict(size=18)
TICK_FONT  = dict(size=16)

# =========================================
st.set_page_config(page_title="ANN pro nádrže", layout="wide")
st.title("Přítoky do nádrží povodí Ohře — predikce pomocí ANN")

# =========================================
# Generic helpers
# =========================================
def list_ids(base_dir: str):
    if not os.path.isdir(base_dir):
        return []
    return sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])

def outputs_dir_for_id(base_dir: str, id_name: str):
    return os.path.join(base_dir, id_name, OUTPUTS_SUBDIR)

def metrics_dir_for_id(base_dir: str, id_name: str):
    return os.path.join(base_dir, id_name, METRICS_SUBDIR)

def sort_horizons(horizons):
    try:
        return sorted(horizons, key=lambda x: int(re.findall(r"\d+", x)[0]))
    except Exception:
        return list(horizons)

# =========================================
# METRICS helpers
# =========================================
def parse_run_metric(fname: str, dataset: str):
    m = re.match(rf"{dataset}_metrics_run(\d+)_([A-Za-z0-9]+)\.csv$", fname)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)

def available_metrics(base_dir: str, id_name: str):
    mdir = metrics_dir_for_id(base_dir, id_name)
    metrics = set()
    for ds in SETS:
        for p in glob.glob(os.path.join(mdir, f"{ds}_metrics_run*_*.csv")):
            _, metric = parse_run_metric(os.path.basename(p), ds)
            if metric:
                metrics.add(metric)
    return sorted(metrics)

def load_metric_long(base_dir: str, id_name: str, dataset: str, metric: str) -> pd.DataFrame:
    mdir = metrics_dir_for_id(base_dir, id_name)
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

# =========================================
# TIME SERIES helpers
# =========================================
def find_horizons_from_real(base_dir: str, id_name: str, dataset: str):
    real_path = os.path.join(outputs_dir_for_id(base_dir, id_name), f"{dataset}_real.csv")
    if not os.path.exists(real_path):
        return []
    df = pd.read_csv(real_path)
    return sort_horizons([c for c in df.columns if str(c).startswith("h")])

def load_real_series(base_dir: str, id_name: str, dataset: str, horizon: str) -> pd.Series:
    path = os.path.join(outputs_dir_for_id(base_dir, id_name), f"{dataset}_real.csv")
    df = pd.read_csv(path)
    if horizon not in df.columns:
        raise ValueError(f"{horizon} not found in {path}")
    return df[horizon]

def load_pred_ensemble(base_dir: str, id_name: str, dataset: str, horizon: str) -> pd.DataFrame:
    outdir = outputs_dir_for_id(base_dir, id_name)
    pattern = os.path.join(outdir, f"{dataset}_pred_*.csv")
    cols = {}
    for p in sorted(glob.glob(pattern)):
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
    return ens.reindex(sorted(ens.columns, key=lambda c: int(c.split("_")[1])), axis=1)

def ensemble_mean(base_dir, id_name, dataset, horizon):
    ens = load_pred_ensemble(base_dir, id_name, dataset, horizon)
    if ens.empty:
        return None
    return ens.mean(axis=1).values

# =========================================
# SINGLE-MODEL PLOTS
# =========================================
def metrics_boxplots_two_panels(df_all: pd.DataFrame, metric: str, same_y: bool) -> go.Figure:
    horizons = sort_horizons(df_all["horizon"].unique())
    fig = make_subplots(rows=1, cols=2, subplot_titles=["calib", "valid"], shared_yaxes=False)
    colors = {"calib": "#1f77b4", "valid": "#ff7f0e"}
    for col_idx, ds in enumerate(SETS, start=1):
        df_ds = df_all[df_all["set"] == ds]
        for h in horizons:
            vals = df_ds.loc[df_ds["horizon"] == h, "value"].dropna().values
            fig.add_trace(go.Box(y=vals, name=h, marker_color=colors[ds],
                                 showlegend=False, boxpoints="outliers"), row=1, col=col_idx)
    fig.update_layout(
        title_text=f"{metric} napříč spuštěními — {model_type} {variant_label} — {id_name}",
        title_font=TITLE_FONT, height=500, template="plotly_white", font=FONT,
    )
    fig.update_xaxes(title_text="Horizon", title_font=AXIS_FONT, tickfont=TICK_FONT)
    for col_idx in (1, 2):
        fig.update_yaxes(title_text=metric, title_font=AXIS_FONT, tickfont=TICK_FONT, row=1, col=col_idx)
    if same_y:
        all_vals = df_all["value"].dropna()
        spread = all_vals.max() - all_vals.min()
        pad = spread * 0.05 if spread > 0 else 0.5
        shared_range = [all_vals.min() - pad, all_vals.max() + pad]
        for col_idx in (1, 2):
            fig.update_yaxes(range=shared_range, row=1, col=col_idx)
    for ann in fig.layout.annotations:
        ann.font = FONT
    return fig

def build_timeseries_figure(base_dir, id_name, horizon, show_members, show_band):
    fig = make_subplots(rows=2, cols=1,
        subplot_titles=[f"calib — horizon {horizon}", f"valid — horizon {horizon}"],
        vertical_spacing=0.22)
    MEMBER_COLOR = "rgba(180,180,180,0.55)"
    MEAN_COLOR   = "#e07b25"
    BAND_COLOR   = "rgba(224,123,37,0.20)"
    REAL_COLOR   = "#111111"
    for row_idx, ds in enumerate(SETS, start=1):
        real_path = os.path.join(outputs_dir_for_id(base_dir, id_name), f"{ds}_real.csv")
        if not os.path.exists(real_path):
            fig.add_annotation(text=f"{ds}: missing real file", xref="paper", yref="paper",
                x=0.5, y=(0.85 if row_idx == 1 else 0.15), showarrow=False,
                font=dict(size=18, color="red"))
            continue
        real = load_real_series(base_dir, id_name, ds, horizon)
        ens  = load_pred_ensemble(base_dir, id_name, ds, horizon)
        x    = np.arange(len(real))
        if show_members and not ens.empty:
            first_member = True
            for c in ens.columns:
                fig.add_trace(go.Scatter(x=x, y=ens[c].values, mode="lines",
                    line=dict(color=MEMBER_COLOR, width=0.8),
                    name="Runs" if first_member else None,
                    legendgroup="members", showlegend=(first_member and row_idx == 1),
                    hoverinfo="skip"), row=row_idx, col=1)
                first_member = False
        if not ens.empty:
            mean = ens.mean(axis=1).values
            std2 = 2.0 * ens.std(axis=1).values
            if show_band:
                fig.add_trace(go.Scatter(x=x, y=mean+std2, mode="lines",
                    line=dict(width=1.5, color=MEAN_COLOR), name="Pred ± 2 std",
                    legendgroup="band", showlegend=(row_idx == 1), hoverinfo="skip"),
                    row=row_idx, col=1)
                fig.add_trace(go.Scatter(x=x, y=mean-std2, mode="lines",
                    line=dict(width=1.5, color=MEAN_COLOR), fill="tonexty",
                    fillcolor=BAND_COLOR, name="Pred ± 2 std", legendgroup="band",
                    showlegend=False, hoverinfo="skip"), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=x, y=mean, mode="lines",
                line=dict(color=MEAN_COLOR, width=2.5), name="Pred mean",
                legendgroup="mean", showlegend=(row_idx == 1),
                hovertemplate="mean: %{y:.3f}<extra></extra>"), row=row_idx, col=1)
        fig.add_trace(go.Scatter(x=x, y=real.values, mode="lines",
            line=dict(color=REAL_COLOR, width=1.2), name="Real",
            legendgroup="real", showlegend=(row_idx == 1),
            hovertemplate="real: %{y:.3f}<extra></extra>"), row=row_idx, col=1)
        fig.update_yaxes(title_text="Přítok [m3/s]", title_font=AXIS_FONT, tickfont=TICK_FONT,
                         row=row_idx, col=1)
        fig.update_xaxes(title_text="Time index [h]", title_font=AXIS_FONT, tickfont=TICK_FONT,
                         row=row_idx, col=1)
    for ann in fig.layout.annotations:
        ann.font = FONT
    fig.update_layout(
        title_text=f"Realné vs ensemble predikce — {model_type} {variant_label} — {id_name} — {horizon}",
        title_font=TITLE_FONT, height=750, template="plotly_white", font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=FONT),
        hovermode="x unified")
    return fig

# =========================================
# COMPARISON helpers — load data for all models
# =========================================
def available_models_for_id(id_name: str):
    """Return only models whose folder contains the given id_name sub-folder."""
    result = []
    for label, folder in ALL_MODELS:
        base = os.path.join(_SCRIPT_DIR, folder)
        if os.path.isdir(os.path.join(base, id_name)):
            result.append((label, folder))
    return result

def load_metric_median_per_horizon(base_dir, id_name, dataset, metric):
    """Returns {horizon: median_value} across all runs for one model."""
    df = load_metric_long(base_dir, id_name, dataset, metric)
    if df.empty:
        return {}
    df_ds = df[df["set"] == dataset]
    return df_ds.groupby("horizon")["value"].median().to_dict()

def common_horizons_for_id(models, id_name):
    """Intersection of horizons available across all given models."""
    sets = []
    for label, folder in models:
        base = os.path.join(_SCRIPT_DIR, folder)
        hz = find_horizons_from_real(base, id_name, "calib")
        if not hz:
            hz = find_horizons_from_real(base, id_name, "valid")
        if hz:
            sets.append(set(hz))
    if not sets:
        return []
    common = sets[0].intersection(*sets[1:])
    return sort_horizons(list(common))

# =========================================
# COMPARISON PLOTS
# =========================================

def comparison_metric_line(models, id_name, metric, dataset):
    """
    Line chart: median metric vs horizon, one line per model.
    Perfect for spotting which model degrades fastest with forecast lead time.
    """
    fig = go.Figure()
    for i, (label, folder) in enumerate(models):
        base = os.path.join(_SCRIPT_DIR, folder)
        med = load_metric_median_per_horizon(base, id_name, dataset, metric)
        if not med:
            continue
        horizons = sort_horizons(med.keys())
        hnums    = [int(re.findall(r"\d+", h)[0]) for h in horizons]
        vals     = [med[h] for h in horizons]
        fig.add_trace(go.Scatter(
            x=hnums, y=vals, mode="lines+markers",
            name=label, line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"<b>{label}</b><br>h%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        title_text=f"Porovnání modelů — {metric} (medián) — {id_name} — {dataset}",
        title_font=TITLE_FONT, template="plotly_white", height=480, font=FONT,
        legend=dict(font=FONT),
        xaxis=dict(title="Forecast horizon [h]", title_font=AXIS_FONT, tickfont=TICK_FONT),
        yaxis=dict(title=f"{metric} (medián)", title_font=AXIS_FONT, tickfont=TICK_FONT),
    )
    return fig


def comparison_metric_boxplot(models, id_name, metric, dataset, horizon):
    """
    Side-by-side boxplots for a single horizon across all models.
    Shows both central tendency and spread (ensemble variability).
    """
    fig = go.Figure()
    for i, (label, folder) in enumerate(models):
        base = os.path.join(_SCRIPT_DIR, folder)
        df = load_metric_long(base, id_name, dataset, metric)
        if df.empty:
            continue
        vals = df[(df["set"] == dataset) & (df["horizon"] == horizon)]["value"].dropna().values
        if len(vals) == 0:
            continue
        fig.add_trace(go.Box(
            y=vals, name=label,
            marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            boxpoints="all", jitter=0.3, pointpos=-1.6,
        ))
    fig.update_layout(
        title_text=f"Boxplot: {metric} @ {horizon} — {id_name} — {dataset}",
        title_font=TITLE_FONT, template="plotly_white", height=480, font=FONT,
        xaxis=dict(title="Model", title_font=AXIS_FONT, tickfont=TICK_FONT),
        yaxis=dict(title=metric, title_font=AXIS_FONT, tickfont=TICK_FONT),
        showlegend=False,
    )
    return fig


def comparison_metric_heatmap(models, id_name, metric, dataset):
    """
    Heatmap: rows = models, cols = horizons, cells = median metric.
    At-a-glance overview of every model × horizon combination.
    """
    model_labels = []
    horizon_set  = set()
    data_dict    = {}

    for label, folder in models:
        base = os.path.join(_SCRIPT_DIR, folder)
        med  = load_metric_median_per_horizon(base, id_name, dataset, metric)
        if not med:
            continue
        model_labels.append(label)
        horizon_set.update(med.keys())
        data_dict[label] = med

    if not model_labels:
        return None

    horizons = sort_horizons(list(horizon_set))
    z = []
    for lbl in model_labels:
        row = [data_dict[lbl].get(h, np.nan) for h in horizons]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=horizons,
        y=model_labels,
        colorscale="RdYlGn" if metric_higher_is_better(metric) else "RdYlGn_r",
        text=[[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Model: %{y}<br>Horizon: %{x}<br>Hodnota: %{z:.4f}<extra></extra>",
        colorbar=dict(title=metric, title_font=AXIS_FONT, tickfont=TICK_FONT),
    ))
    fig.update_layout(
        title_text=f"Heatmapa: {metric} (medián) — {id_name} — {dataset}",
        title_font=TITLE_FONT, template="plotly_white",
        height=max(300, 80 * len(model_labels) + 120), font=FONT,
        xaxis=dict(title="Horizon", title_font=AXIS_FONT, tickfont=TICK_FONT),
        yaxis=dict(title="Model", title_font=AXIS_FONT, tickfont=TICK_FONT, autorange="reversed"),
    )
    return fig


def comparison_metric_table(models, id_name, metric):
    """
    Summary table: median metric per (model × horizon) for calib AND valid,
    returned as a styled DataFrame (displayed via st.dataframe).
    """
    rows = []
    for label, folder in models:
        base = os.path.join(_SCRIPT_DIR, folder)
        for ds in SETS:
            med = load_metric_median_per_horizon(base, id_name, ds, metric)
            if not med:
                continue
            horizons = sort_horizons(med.keys())
            for h in horizons:
                rows.append({"Model": label, "Set": ds, "Horizon": h, metric: round(med[h], 4)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["Model", "Set"], columns="Horizon", values=metric, aggfunc="first")
    pivot.columns = sort_horizons(pivot.columns.tolist())
    return pivot


def comparison_scatter(models, id_name, horizon, dataset):
    """
    Real vs Pred mean scatter — one panel per model (up to 3 per row).
    The 1:1 line is the perfect-model reference. Points below = under-prediction.
    """
    valid_models = [(l, f) for l, f in models
                    if ensemble_mean(os.path.join(_SCRIPT_DIR, f), id_name, dataset, horizon) is not None]
    if not valid_models:
        return None

    ncols = min(3, len(valid_models))
    nrows = int(np.ceil(len(valid_models) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[l for l, _ in valid_models],
                        vertical_spacing=0.15, horizontal_spacing=0.10)

    for i, (label, folder) in enumerate(valid_models):
        base  = os.path.join(_SCRIPT_DIR, folder)
        row   = i // ncols + 1
        col   = i % ncols + 1
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        try:
            real = load_real_series(base, id_name, dataset, horizon).values
        except Exception:
            continue
        pred = ensemble_mean(base, id_name, dataset, horizon)
        if pred is None:
            continue
        n = min(len(real), len(pred))
        real, pred = real[:n], pred[:n]

        # 1:1 reference line
        lo, hi = min(real.min(), pred.min()), max(real.max(), pred.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            showlegend=False, hoverinfo="skip"), row=row, col=col)

        # scatter points
        fig.add_trace(go.Scatter(x=real, y=pred, mode="markers",
            marker=dict(color=color, size=4, opacity=0.55),
            name=label, showlegend=False,
            hovertemplate="Real: %{x:.3f}<br>Pred: %{y:.3f}<extra></extra>"),
            row=row, col=col)

        fig.update_xaxes(title_text="Real [m³/s]", title_font=AXIS_FONT,
                         tickfont=TICK_FONT, row=row, col=col)
        fig.update_yaxes(title_text="Pred mean [m³/s]", title_font=AXIS_FONT,
                         tickfont=TICK_FONT, row=row, col=col)

    for ann in fig.layout.annotations:
        ann.font = FONT

    fig.update_layout(
        title_text=f"Real vs Pred mean — {id_name} — {dataset} — {horizon}",
        title_font=TITLE_FONT, template="plotly_white",
        height=420 * nrows, font=FONT,
    )
    return fig


def comparison_residuals(models, id_name, horizon, dataset):
    """
    Residual time series: (Pred mean − Real) vs time index, one line per model.
    Reveals systematic bias, trend drift, and timing errors across models.
    """
    fig = go.Figure()
    any_data = False
    for i, (label, folder) in enumerate(models):
        base = os.path.join(_SCRIPT_DIR, folder)
        try:
            real = load_real_series(base, id_name, dataset, horizon).values
        except Exception:
            continue
        pred = ensemble_mean(base, id_name, dataset, horizon)
        if pred is None:
            continue
        n = min(len(real), len(pred))
        resid = pred[:n] - real[:n]
        fig.add_trace(go.Scatter(
            x=np.arange(n), y=resid, mode="lines",
            line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=1.8),
            name=label,
            hovertemplate=f"<b>{label}</b><br>t=%{{x}}<br>resid=%{{y:.3f}}<extra></extra>",
        ))
        any_data = True
    if not any_data:
        return None
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title_text=f"Rezidua (Pred − Real) — {id_name} — {dataset} — {horizon}",
        title_font=TITLE_FONT, template="plotly_white", height=420, font=FONT,
        xaxis=dict(title="Time index [h]", title_font=AXIS_FONT, tickfont=TICK_FONT),
        yaxis=dict(title="Residual [m³/s]", title_font=AXIS_FONT, tickfont=TICK_FONT),
        legend=dict(font=FONT), hovermode="x unified",
    )
    return fig


def comparison_taylor(models, id_name, horizon, dataset):
    """
    Taylor diagram (polar): angular axis = arccos(correlation), radial = normalised std dev.
    Reference point is always at angle=0, radius=1 (the observed std dev normalised to 1).
    Closer to the reference = better model.
    """
    # Collect stats
    stats = []
    ref_std = None
    for label, folder in models:
        base = os.path.join(_SCRIPT_DIR, folder)
        try:
            real = load_real_series(base, id_name, dataset, horizon).values.astype(float)
        except Exception:
            continue
        pred = ensemble_mean(base, id_name, dataset, horizon)
        if pred is None:
            continue
        n = min(len(real), len(pred))
        real, pred = real[:n], pred[:n]
        mask = np.isfinite(real) & np.isfinite(pred)
        real, pred = real[mask], pred[mask]
        if len(real) < 5:
            continue
        corr   = np.corrcoef(real, pred)[0, 1]
        std_r  = np.std(real, ddof=1)
        std_p  = np.std(pred, ddof=1)
        if ref_std is None:
            ref_std = std_r
        stats.append((label, corr, std_p / ref_std))

    if not stats:
        return None

    # Build polar scatter
    fig = go.Figure()

    # Reference point: angle=0, r=1
    fig.add_trace(go.Scatterpolar(
        r=[1.0], theta=[0],
        mode="markers+text",
        marker=dict(symbol="star", size=16, color="black"),
        text=["Observed"], textposition="top center",
        name="Observed", showlegend=True,
        hovertemplate="Observed (ref)<extra></extra>",
    ))

    for i, (label, corr, norm_std) in enumerate(stats):
        angle_deg = np.degrees(np.arccos(np.clip(corr, -1, 1)))
        fig.add_trace(go.Scatterpolar(
            r=[norm_std], theta=[angle_deg],
            mode="markers+text",
            marker=dict(size=14, color=MODEL_COLORS[i % len(MODEL_COLORS)]),
            text=[label], textposition="top center",
            name=label,
            hovertemplate=f"<b>{label}</b><br>Corr: {corr:.3f}<br>Norm. std: {norm_std:.3f}<extra></extra>",
        ))

    # Add RMSE arcs (dashed circles centred on the reference point r=1, θ=0)
    # We draw them as annotations in a cartesian-to-polar conversion for simplicity.
    # (Full Taylor diagram RMSE arcs require custom shapes — skipped here for clarity.)

    fig.update_layout(
        title_text=f"Taylorův diagram — {id_name} — {dataset} — {horizon}",
        title_font=TITLE_FONT, template="plotly_white", height=520, font=FONT,
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 15, 30, 45, 60, 75, 90],
                ticktext=["r=1.00", "r=0.97", "r=0.87", "r=0.71", "r=0.50", "r=0.26", "r=0.00"],
                direction="counterclockwise",
                rotation=0,
                tickfont=TICK_FONT,
            ),
            radialaxis=dict(
                title="Norm. std dev", title_font=AXIS_FONT, tickfont=TICK_FONT,
                range=[0, max(2.0, max(s[2] for s in stats) * 1.15)],
            ),
        ),
        legend=dict(font=FONT),
    )
    return fig


def comparison_residual_histogram(models, id_name, horizon, dataset):
    """
    Overlapping residual histograms — one per model.
    Shows the error distribution: ideally centred at 0, narrow, symmetric.
    """
    fig = go.Figure()
    any_data = False
    for i, (label, folder) in enumerate(models):
        base = os.path.join(_SCRIPT_DIR, folder)
        try:
            real = load_real_series(base, id_name, dataset, horizon).values.astype(float)
        except Exception:
            continue
        pred = ensemble_mean(base, id_name, dataset, horizon)
        if pred is None:
            continue
        n = min(len(real), len(pred))
        resid = pred[:n] - real[:n]
        fig.add_trace(go.Histogram(
            x=resid, name=label, opacity=0.55,
            marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            nbinsx=40,
            hovertemplate=f"<b>{label}</b><br>Bin: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))
        any_data = True
    if not any_data:
        return None
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1.5)
    fig.update_layout(
        barmode="overlay",
        title_text=f"Rozdělení reziduí — {id_name} — {dataset} — {horizon}",
        title_font=TITLE_FONT, template="plotly_white", height=420, font=FONT,
        xaxis=dict(title="Residual [m³/s]", title_font=AXIS_FONT, tickfont=TICK_FONT),
        yaxis=dict(title="Počet", title_font=AXIS_FONT, tickfont=TICK_FONT),
        legend=dict(font=FONT),
    )
    return fig


# =========================================
# Sidebar UI
# =========================================
with st.sidebar:
    st.header("Nastavení")

    # ---- Model type + variant ----
    model_type    = st.selectbox("Typ modelu", list(MODEL_TYPES.keys()))
    variants      = MODEL_TYPES[model_type]
    variant_label = st.selectbox("Varianta", list(variants.keys()))
    scenario_folder = variants[variant_label]
    BASE_DIR = os.path.join(_SCRIPT_DIR, scenario_folder)

    if not os.path.isdir(BASE_DIR):
        st.error(f"Složka nenalezena: {scenario_folder}/")
        st.stop()

    ids = list_ids(BASE_DIR)
    if not ids:
        st.error(f"Ve složce {scenario_folder}/ nebyly nalezeny žádné podsložky.")
        st.stop()

    id_name = st.selectbox("Nádrž", ids)

    st.divider()

    view = st.radio(
        "Zobrazení",
        [
            "Time series (real + ensemble preds)",
            "Metrics (boxplots)",
            "Porovnání všech modelů",
        ],
        index=0,
    )

    # ---- Per-view controls ----
    if view.startswith("Time"):
        hz = find_horizons_from_real(BASE_DIR, id_name, "calib") or \
             find_horizons_from_real(BASE_DIR, id_name, "valid")
        if not hz:
            st.error("No horizons found.")
            st.stop()
        horizon      = st.selectbox("Horizon", hz)
        show_members = st.checkbox("Zobraz každé spuštení (šedá linie)", value=True)
        show_band    = st.checkbox("Ukaž průměr ± 2std", value=True)

    elif view.startswith("Metrics"):
        metrics = available_metrics(BASE_DIR, id_name)
        if not metrics:
            st.error(f"No metric files found in: {metrics_dir_for_id(BASE_DIR, id_name)}")
            st.stop()
        metric = st.selectbox("Metrika", metrics)
        same_y = st.checkbox("Použij stejnou y-osu pro calib a valid", value=False)

    else:  # Comparison view
        # Which metrics are available in any model?
        all_metrics_found = set()
        for _, folder in ALL_MODELS:
            base = os.path.join(_SCRIPT_DIR, folder)
            all_metrics_found.update(available_metrics(base, id_name))
        all_metrics_found = sorted(all_metrics_found)

        if not all_metrics_found:
            st.warning("Nebyly nalezeny žádné metriky pro tuto nádrž v žádném modelu.")
        cmp_metric  = st.selectbox("Metrika", all_metrics_found) if all_metrics_found else None

        # Horizon selector for scatter / residual plots
        cmp_models  = available_models_for_id(id_name)
        cmp_hz      = common_horizons_for_id(cmp_models, id_name)
        cmp_horizon = st.selectbox("Horizon (pro scatter/rezidua)", cmp_hz) if cmp_hz else None


# =========================================
# Active model banner
# =========================================
st.info(
    f"**Aktivní model:** {model_type} — {variant_label}   |   "
    f"Složka: `{scenario_folder}/`   |   Nádrž: **{id_name}**",
    icon="🤖",
)

# =========================================
# Main — dispatch on view
# =========================================

# ---- Time series ----
if view.startswith("Time"):
    missing = [f"{ds}_real.csv" for ds in SETS
               if not os.path.exists(os.path.join(outputs_dir_for_id(BASE_DIR, id_name), f"{ds}_real.csv"))]
    if missing:
        st.warning("Missing files: " + ", ".join(missing) + " (that panel will be empty).")
    st.plotly_chart(build_timeseries_figure(BASE_DIR, id_name, horizon, show_members, show_band),
                    width='stretch')

# ---- Metrics boxplots ----
elif view.startswith("Metrics"):
    df_calib = load_metric_long(BASE_DIR, id_name, "calib", metric)
    df_valid  = load_metric_long(BASE_DIR, id_name, "valid", metric)
    df_all    = pd.concat([df_calib, df_valid], ignore_index=True)
    if df_all.empty:
        st.error("No metrics loaded for this selection.")
        st.stop()
    st.plotly_chart(metrics_boxplots_two_panels(df_all, metric, same_y=same_y),
                    width='stretch')

# ---- Comparison view ----
else:
    cmp_models = available_models_for_id(id_name)
    if not cmp_models:
        st.error("Žádný model neobsahuje data pro tuto nádrž.")
        st.stop()

    model_names_str = ", ".join(l for l, _ in cmp_models)
    st.caption(f"Dostupné modely pro nádrž **{id_name}**: {model_names_str}")

    # ── 1. Metric line chart ──────────────────────────────────────────────
    if cmp_metric:
        st.subheader(f"① Vývoj {cmp_metric} přes horizonty")
        st.caption("Medián přes všechna spuštění. Ukazuje, jak rychle se přesnost zhoršuje s délkou předpovědi.")
        col1, col2 = st.columns(2)
        with col1:
            fig = comparison_metric_line(cmp_models, id_name, cmp_metric, "calib")
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = comparison_metric_line(cmp_models, id_name, cmp_metric, "valid")
            st.plotly_chart(fig, width='stretch')

        # ── 2. Heatmap ───────────────────────────────────────────────────
        st.subheader(f"② Heatmapa {cmp_metric} — přehled model × horizont")
        st.caption("Každá buňka je medián metriky. Barva umožňuje okamžité srovnání.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**calib**")
            fig = comparison_metric_heatmap(cmp_models, id_name, cmp_metric, "calib")
            if fig:
                st.plotly_chart(fig, width='stretch')
        with col2:
            st.markdown("**valid**")
            fig = comparison_metric_heatmap(cmp_models, id_name, cmp_metric, "valid")
            if fig:
                st.plotly_chart(fig, width='stretch')

        # ── 3. Boxplot side-by-side for chosen horizon ────────────────────
        if cmp_horizon:
            st.subheader(f"③ Boxplot {cmp_metric} @ {cmp_horizon} — rozptyl přes spuštění")
            st.caption("Zobrazuje medián i rozptyl (nestabilitu) každého modelu na zvoleném horizontu.")
            col1, col2 = st.columns(2)
            with col1:
                fig = comparison_metric_boxplot(cmp_models, id_name, cmp_metric, "calib", cmp_horizon)
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = comparison_metric_boxplot(cmp_models, id_name, cmp_metric, "valid", cmp_horizon)
                st.plotly_chart(fig, width='stretch')

        # ── 4. Summary table ──────────────────────────────────────────────
        st.subheader(f"④ Tabulka mediánů — {cmp_metric}")
        st.caption("Přesné hodnoty pro calib i valid, seřazené podle modelu a setu.")
        pivot = comparison_metric_table(cmp_models, id_name, cmp_metric)
        if pivot is not None:
            # Colour gradient per column (lower=better for RMSE/MAE, higher=better for NSE/KGE/R2)
            cmap = "RdYlGn" if metric_higher_is_better(cmp_metric) else "RdYlGn_r"
            st.dataframe(
                pivot.style.background_gradient(cmap=cmap, axis=0).format("{:.4f}"),
                width='stretch',
            )

    # ── 5. Scatter: Real vs Pred mean ─────────────────────────────────────
    if cmp_horizon:
        st.subheader(f"⑤ Scatter: Real vs Pred mean @ {cmp_horizon}")
        st.caption("Bod na úhlopříčce = perfektní predikce. Body pod = podhodnocení, nad = nadhodnocení.")
        col1, col2 = st.columns(2)
        with col1:
            fig = comparison_scatter(cmp_models, id_name, cmp_horizon, "calib")
            if fig:
                st.plotly_chart(fig, width='stretch')
        with col2:
            fig = comparison_scatter(cmp_models, id_name, cmp_horizon, "valid")
            if fig:
                st.plotly_chart(fig, width='stretch')

        # ── 6. Residual time series ───────────────────────────────────────
        st.subheader(f"⑥ Rezidua v čase @ {cmp_horizon}")
        st.caption("Pred mean − Real. Systematická odchylka = bias. Ideálně náhodný šum kolem nuly.")
        col1, col2 = st.columns(2)
        with col1:
            fig = comparison_residuals(cmp_models, id_name, cmp_horizon, "calib")
            if fig:
                st.plotly_chart(fig, width='stretch')
        with col2:
            fig = comparison_residuals(cmp_models, id_name, cmp_horizon, "valid")
            if fig:
                st.plotly_chart(fig, width='stretch')

        # ── 7. Residual histogram ─────────────────────────────────────────
        st.subheader(f"⑦ Histogram reziduí @ {cmp_horizon}")
        st.caption("Ideálně: úzký, symetrický, centrovaný na nule. Šikmé rozdělení = systematický bias.")
        col1, col2 = st.columns(2)
        with col1:
            fig = comparison_residual_histogram(cmp_models, id_name, cmp_horizon, "calib")
            if fig:
                st.plotly_chart(fig, width='stretch')
        with col2:
            fig = comparison_residual_histogram(cmp_models, id_name, cmp_horizon, "valid")
            if fig:
                st.plotly_chart(fig, width='stretch')

        # # ── 8. Taylor diagram ─────────────────────────────────────────────
        # st.subheader(f"⑧ Taylorův diagram @ {cmp_horizon}")
        # st.caption(
        #     "Radiální osa = normovaná std dev (1 = pozorování). "
        #     "Úhlová osa = korelace. Nejlepší model = co nejblíže hvězdičce (referenci)."
        # )
        # col1, col2 = st.columns(2)
        # with col1:
        #     fig = comparison_taylor(cmp_models, id_name, cmp_horizon, "calib")
        #     if fig:
        #         st.plotly_chart(fig, width='stretch')
        # with col2:
        #     fig = comparison_taylor(cmp_models, id_name, cmp_horizon, "valid")
        #     if fig:
        #         st.plotly_chart(fig, width='stretch')