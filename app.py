# streamlit run app.py

import os
import re
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        # keep natural h-order (h1,h2,...)
        out["hnum"] = out["horizon"].str.extract(r"h(\d+)").astype(float)
        out = out.sort_values(["set", "hnum", "run"]).drop(columns=["hnum"])
    return out

def metrics_boxplots_two_panels(df_all: pd.DataFrame, metric: str, same_y: bool):
    # horizon order from data (supports h1..h3 etc.)
    horizons = sorted(df_all["horizon"].unique(), key=lambda x: int(re.findall(r"\d+", x)[0]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharex=True)
    for ax, ds in zip(axes, SETS):
        df_ds = df_all[df_all["set"] == ds]
        data = [df_ds.loc[df_ds["horizon"] == h, "value"].dropna().values for h in horizons]

        ax.boxplot(data, tick_labels=horizons, showfliers=True)
        ax.set_title(ds)
        ax.set_xlabel("Horizon")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"{metric} napříč spuštěními - kalibrace vs validace", y=1.05)

    if same_y:
        ymin = min(ax.get_ylim()[0] for ax in axes)
        ymax = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    return fig

# =========================================
# TIME SERIES helpers (real + ensemble preds)
# =========================================
def find_horizons_from_real(id_name: str, dataset: str):
    """
    Reads headers from <dataset>_real.csv and returns horizon columns found (h1..hN).
    """
    real_path = os.path.join(outputs_dir_for_id(id_name), f"{dataset}_real.csv")
    if not os.path.exists(real_path):
        return []
    df = pd.read_csv(real_path)
    horizons = [c for c in df.columns if str(c).startswith("h")]
    # sort by number if possible
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
    # sort columns by run number
    ens = ens.reindex(sorted(ens.columns, key=lambda c: int(c.split("_")[1])), axis=1)
    return ens

def plot_timeseries_panel(ax, real: pd.Series, ens: pd.DataFrame, title: str,
                          show_members: bool, show_band: bool):
    x = np.arange(len(real))

    # ---- ensemble members (very light grey)
    if show_members and not ens.empty:
        for c in ens.columns:
            ax.plot(
                x,
                ens[c].values,
                color="0.8",
                linewidth=0.8,
                alpha=0.8,
                zorder=1
            )

    # ---- ensemble mean + band
    if not ens.empty:
        mean = ens.mean(axis=1).values
        std = ens.std(axis=1).values
        std2 = 2.0 * std

        if show_band:
            ax.fill_between(
                x,
                mean - std2,
                mean + std2,
                color="tab:orange",
                alpha=0.5,
                zorder=2,
                label="Pred ± 2 std"
            )

        # mean line (dominant)
        ax.plot(
            x,
            mean,
            color="tab:orange",
            linewidth=3.0,
            linestyle="-",
            zorder=4,
            label="Pred mean"
        )

    # ---- real data (black but slightly thinner than mean)
    ax.plot(
        x,
        real.values,
        color="black",
        linewidth=1.0,
        zorder=5,
        label="Real"
    )

    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.grid(alpha=0.3)


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

    view = st.radio("Zobrazení", ["Time series (real + ensemble preds)", "Metrics (boxplots)"], index=0)

    if view.startswith("Time"):
        # horizons from calib_real if exists, else valid_real, else none
        hz = find_horizons_from_real(id_name, "calib")
        if not hz:
            hz = find_horizons_from_real(id_name, "valid")
        if not hz:
            st.error("No horizons found (missing calib_real.csv / valid_real.csv or no h-columns).")
            st.stop()

        horizon = st.selectbox("Horizon", hz)

        show_members = st.checkbox("Zobraz každé spuštení (šedá linie)", value=True)
        show_band = st.checkbox("Ukaž průměr ± 2std", value=True)

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
    # load calib + valid real + ensemble
    missing = []
    for ds in SETS:
        if not os.path.exists(os.path.join(outputs_dir_for_id(id_name), f"{ds}_real.csv")):
            missing.append(f"{ds}_real.csv")

    if missing:
        st.warning("Missing files: " + ", ".join(missing) + " (that panel will be empty if needed).")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    for ax, ds in zip(axes, SETS):
        real_path = os.path.join(outputs_dir_for_id(id_name), f"{ds}_real.csv")
        if not os.path.exists(real_path):
            ax.set_title(f"{ds} (missing real file)")
            ax.axis("off")
            continue

        real = load_real_series(id_name, ds, horizon)
        ens = load_pred_ensemble(id_name, ds, horizon)

        plot_timeseries_panel(
            ax=ax,
            real=real,
            ens=ens,
            title=f"{ds} — horizon {horizon}",
            show_members=show_members,
            show_band=show_band
        )
        ax.set_ylabel("Value")
        ax.legend(loc="best")

    fig.suptitle(f"Realné vs ensemble predikce — {id_name} — {horizon}", y=0.98)
    fig.tight_layout()
    st.pyplot(fig)

else:
    # metrics boxplots
    df_calib = load_metric_long(id_name, "calib", metric)
    df_valid = load_metric_long(id_name, "valid", metric)
    df_all = pd.concat([df_calib, df_valid], ignore_index=True)

    if df_all.empty:
        st.error("No metrics loaded for this selection.")
        st.stop()

    fig = metrics_boxplots_two_panels(df_all, metric, same_y=same_y)
    st.pyplot(fig)
