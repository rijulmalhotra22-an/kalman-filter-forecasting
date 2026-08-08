"""
app.py — Streamlit multi-page analytics dashboard for the Kalman Filter Forecasting System.

Pages:
  1. Model State — Kalman hedge ratio, uncertainty bands, raw vs filtered spread
  2. Performance vs Baseline — equity curves, metrics comparison table
  3. Forecast Health Checks — cointegration tests, z-score distribution, signal frequency
  4. Daily Signal — current z-score, signal indicator, entry/exit levels, trade log
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.pipeline import load_data, CointTestResult
from src.backtest import run_full_backtest

# ──────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kalman Filter Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #e0e0ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #a0a0c0;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.25);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }
    .metric-label {
        color: #8b8bc0;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-good { color: #4ade80; }
    .metric-bad { color: #f87171; }
    .metric-neutral { color: #c4b5fd; }

    /* Signal badges */
    .signal-long {
        background: linear-gradient(135deg, #065f46, #059669);
        color: #d1fae5;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    .signal-short {
        background: linear-gradient(135deg, #7f1d1d, #dc2626);
        color: #fee2e2;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    .signal-flat {
        background: linear-gradient(135deg, #374151, #6b7280);
        color: #e5e7eb;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        letter-spacing: 0.05em;
    }

    /* Section headers */
    .section-header {
        color: #c4b5fd;
        font-size: 1.15rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd;
    }

    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Chart template
# ──────────────────────────────────────────────────────────────────

CHART_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(15,12,41,0.6)",
        plot_bgcolor="rgba(26,26,46,0.8)",
        font=dict(family="Inter, sans-serif", color="#c4b5fd"),
        xaxis=dict(gridcolor="rgba(99,102,241,0.1)", zerolinecolor="rgba(99,102,241,0.2)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.1)", zerolinecolor="rgba(99,102,241,0.2)"),
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(15,12,41,0.7)",
            bordercolor="rgba(99,102,241,0.3)",
            borderwidth=1,
        ),
    )
)

COLORS = {
    "kalman": "#818cf8",
    "ols": "#fb923c",
    "bh": "#6b7280",
    "long": "#4ade80",
    "short": "#f87171",
    "flat": "#6b7280",
    "band": "rgba(129,140,248,0.15)",
    "spread_raw": "rgba(251,146,60,0.4)",
    "spread_filtered": "#818cf8",
}


# ──────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return parameters."""
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.markdown("---")

        st.markdown("##### 📊 Data Parameters")
        n_days = st.slider("Data length (trading days)", 500, 2000, 1260, 10)
        seed = st.number_input("Random seed", 1, 9999, 42)

        st.markdown("---")
        st.markdown("##### 🎯 Signal Thresholds")
        entry_threshold = st.slider("Z-score entry threshold", 0.5, 3.0, 1.5, 0.1)
        exit_threshold = st.slider("Z-score exit threshold", 0.0, 1.5, 0.5, 0.1)

        st.markdown("---")
        st.markdown("##### 💰 Cost Model")
        txn_cost = st.slider("Transaction cost (bps)", 0, 50, 5, 1)
        slippage = st.slider("Slippage (bps)", 0, 20, 2, 1)

        st.markdown("---")
        st.markdown("##### 🔧 Kalman Parameters")
        delta = st.select_slider(
            "Delta (adaptation speed)",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        R = st.select_slider(
            "R (observation noise)",
            options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
        )

        st.markdown("---")
        st.markdown(
            "<p style='color:#6b7280;font-size:0.75rem;text-align:center;'>"
            "Kalman Filter Forecasting System<br>Built with Streamlit + Plotly</p>",
            unsafe_allow_html=True,
        )

    return {
        "n_days": n_days,
        "seed": int(seed),
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "txn_cost_bps": float(txn_cost),
        "slippage_bps": float(slippage),
        "delta": delta,
        "R": R,
    }


# ──────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running data pipeline & cointegration tests …")
def cached_load_data(seed: int, n_days: int):
    return load_data(seed=seed, n_days=n_days)


@st.cache_data(show_spinner="Running Kalman filter & backtests …")
def cached_backtest(
    seed: int,
    n_days: int,
    delta: float,
    R: float,
    entry_threshold: float,
    exit_threshold: float,
    txn_cost_bps: float,
    slippage_bps: float,
):
    df, _ = load_data(seed=seed, n_days=n_days)
    return run_full_backtest(
        df,
        delta=delta,
        R=R,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        txn_cost_bps=txn_cost_bps,
        slippage_bps=slippage_bps,
    )


# ──────────────────────────────────────────────────────────────────
# Helper for metric cards
# ──────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, css_class: str = "metric-neutral") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────
# PAGE 1: Model State
# ──────────────────────────────────────────────────────────────────

def page_model_state(df: pd.DataFrame, results: dict):
    st.markdown('<div class="section-header">Kalman-Estimated Hedge Ratio Over Time</div>', unsafe_allow_html=True)

    kf_out = results["kalman_output"]
    dates = df["date"]

    # Hedge ratio with uncertainty bands
    beta = kf_out["beta"]
    beta_std = kf_out["beta_std"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=beta + 2 * beta_std,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=beta - 2 * beta_std,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=COLORS["band"],
        showlegend=True, name="±2σ Uncertainty Band",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=beta,
        mode="lines", line=dict(color=COLORS["kalman"], width=2),
        name="Hedge Ratio (β)",
    ))

    # Add split lines
    split = results["split"]
    train_end = df["date"].iloc[len(split.train) - 1]
    val_end = df["date"].iloc[len(split.train) + len(split.validation) - 1]
    for dt, label in [(train_end, "Train|Val"), (val_end, "Val|Test")]:
        fig.add_vline(x=dt, line=dict(color="#6b7280", dash="dash", width=1),
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        title="Dynamic Hedge Ratio (Kalman Filter)",
        yaxis_title="Hedge Ratio (β)",
        xaxis_title="Date",
        height=420,
        **CHART_TEMPLATE["layout"].to_plotly_json(),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw spread vs filtered spread
    st.markdown('<div class="section-header">Raw Spread vs Kalman-Filtered Spread</div>', unsafe_allow_html=True)

    raw_spread = df["series_y"].values - (kf_out["alpha"] + kf_out["beta"] * df["series_x"].values)
    filtered_spread = kf_out["spread"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dates, y=raw_spread,
        mode="lines", line=dict(color=COLORS["spread_raw"], width=1.5),
        name="Raw Spread (y - α̂ - β̂·x)",
    ))
    fig2.add_trace(go.Scatter(
        x=dates, y=filtered_spread,
        mode="lines", line=dict(color=COLORS["spread_filtered"], width=2),
        name="Kalman Prediction Error",
    ))

    fig2.update_layout(
        title="Spread Comparison",
        yaxis_title="Spread",
        height=380,
        **CHART_TEMPLATE["layout"].to_plotly_json(),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Intercept alpha
    col1, col2 = st.columns(2)
    with col1:
        fig3 = go.Figure()
        alpha = kf_out["alpha"]
        alpha_std = kf_out["alpha_std"]
        fig3.add_trace(go.Scatter(x=dates, y=alpha + 2 * alpha_std, mode="lines", line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=dates, y=alpha - 2 * alpha_std, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(74,222,128,0.12)", name="±2σ Band"))
        fig3.add_trace(go.Scatter(x=dates, y=alpha, mode="lines", line=dict(color="#4ade80", width=2), name="Intercept (α)"))
        fig3.update_layout(title="Kalman Intercept (α)", height=300, **CHART_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = go.Figure()
        spread_std = np.sqrt(kf_out["spread_var"])
        fig4.add_trace(go.Scatter(x=dates, y=spread_std, mode="lines", line=dict(color="#facc15", width=2), name="√S (Innovation Std)"))
        fig4.update_layout(title="Innovation Std Dev (Filter Uncertainty)", height=300, **CHART_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# PAGE 2: Performance vs Baseline
# ──────────────────────────────────────────────────────────────────

def page_performance(results: dict):
    st.markdown('<div class="section-header">Out-of-Sample Equity Curves (Test Set)</div>', unsafe_allow_html=True)

    kalman = results["kalman_test"]
    ols = results["ols_test"]
    bh = results["bh_test"]

    # Equity curves
    fig = go.Figure()
    for res, color in [(kalman, COLORS["kalman"]), (ols, COLORS["ols"]), (bh, COLORS["bh"])]:
        fig.add_trace(go.Scatter(
            x=res.signals_df["date"], y=res.equity_curve,
            mode="lines", line=dict(color=color, width=2.5),
            name=res.name,
        ))

    fig.update_layout(
        title="Equity Curves — Kalman vs OLS vs Buy & Hold (OOS)",
        yaxis_title="Equity",
        xaxis_title="Date",
        height=450,
        **CHART_TEMPLATE["layout"].to_plotly_json(),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics comparison table
    st.markdown('<div class="section-header">Performance Metrics Comparison</div>', unsafe_allow_html=True)

    metrics_data = []
    for res in [kalman, ols, bh]:
        m = res.metrics
        metrics_data.append({
            "Strategy": res.name,
            "CAGR": f"{m['CAGR']:.1%}",
            "Sharpe Ratio": f"{m['Sharpe Ratio']:.2f}",
            "Max Drawdown": f"{m['Max Drawdown']:.1%}",
            "Win Rate": f"{m['Win Rate']:.1%}",
            "Profit Factor": f"{m['Profit Factor']:.2f}" if m['Profit Factor'] < 999 else "∞",
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(
        metrics_df.style.set_properties(**{
            "background-color": "#1a1a2e",
            "color": "#e0e0ff",
            "border-color": "rgba(99,102,241,0.2)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Kalman metrics highlight
    st.markdown('<div class="section-header">Kalman Strategy — Key Metrics</div>', unsafe_allow_html=True)
    km = kalman.metrics
    cols = st.columns(5)
    items = [
        ("CAGR", f"{km['CAGR']:.1%}", "metric-good" if km['CAGR'] > 0.10 else "metric-bad"),
        ("Sharpe", f"{km['Sharpe Ratio']:.2f}", "metric-good" if km['Sharpe Ratio'] > 1.2 else "metric-bad"),
        ("Max DD", f"{km['Max Drawdown']:.1%}", "metric-good" if km['Max Drawdown'] < 0.15 else "metric-bad"),
        ("Win Rate", f"{km['Win Rate']:.1%}", "metric-good" if km['Win Rate'] > 0.60 else "metric-bad"),
        ("Profit F.", f"{km['Profit Factor']:.2f}" if km['Profit Factor'] < 999 else "∞",
         "metric-good" if km['Profit Factor'] > 1.4 else "metric-bad"),
    ]
    for col, (label, val, css) in zip(cols, items):
        with col:
            st.markdown(metric_card(label, val, css), unsafe_allow_html=True)

    # Kalman outperformance note
    kalman_sharpe = km["Sharpe Ratio"]
    ols_sharpe = ols.metrics["Sharpe Ratio"]
    if kalman_sharpe > ols_sharpe:
        delta_sharpe = kalman_sharpe - ols_sharpe
        st.success(f"✅ Kalman outperforms OLS baseline by **{delta_sharpe:.2f}** Sharpe points")
    else:
        st.warning("⚠️ OLS baseline outperforms Kalman on this configuration — consider tuning parameters")


# ──────────────────────────────────────────────────────────────────
# PAGE 3: Forecast Health Checks
# ──────────────────────────────────────────────────────────────────

def page_health_checks(
    coint_result: CointTestResult,
    results: dict,
    params: dict,
):
    # Cointegration test results
    st.markdown('<div class="section-header">Cointegration Test Results</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            metric_card(
                "Engle-Granger p-value",
                f"{coint_result.engle_granger_pvalue:.6f}",
                "metric-good" if coint_result.engle_granger_pvalue < 0.05 else "metric-bad",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            metric_card(
                "EG Test Statistic",
                f"{coint_result.engle_granger_stat:.4f}",
                "metric-neutral",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        joh_pass = coint_result.johansen_trace_stat > coint_result.johansen_critical_5pct
        st.markdown(
            metric_card(
                "Johansen Trace vs 5% Critical",
                f"{coint_result.johansen_trace_stat:.2f} vs {coint_result.johansen_critical_5pct:.2f}",
                "metric-good" if joh_pass else "metric-bad",
            ),
            unsafe_allow_html=True,
        )

    if coint_result.is_cointegrated:
        st.success("✅ Both tests confirm cointegration at the 5% significance level")
    else:
        st.error("❌ Cointegration NOT confirmed — results may be unreliable")

    # Z-score distribution
    st.markdown('<div class="section-header">Z-Score Distribution</div>', unsafe_allow_html=True)
    kalman_full = results["kalman_full"]
    zscore = kalman_full.signals_df["zscore"].values

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=zscore, nbinsx=60,
            marker_color=COLORS["kalman"],
            opacity=0.8,
            name="Z-Score",
        ))
        entry = params["entry_threshold"]
        for thresh in [-entry, entry]:
            fig.add_vline(x=thresh, line=dict(color="#f87171", dash="dash", width=2),
                          annotation_text=f"Entry ±{entry}")
        fig.add_vline(x=0, line=dict(color="#4ade80", dash="dot", width=1))
        fig.update_layout(
            title="Z-Score Distribution (Full Series)",
            xaxis_title="Z-Score",
            yaxis_title="Frequency",
            height=380,
            **CHART_TEMPLATE["layout"].to_plotly_json(),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Z-Score Statistics**")
        st.write(f"Mean: `{np.mean(zscore):.4f}`")
        st.write(f"Std: `{np.std(zscore):.4f}`")
        st.write(f"Skew: `{pd.Series(zscore).skew():.4f}`")
        st.write(f"Kurtosis: `{pd.Series(zscore).kurtosis():.4f}`")
        st.write(f"% > +{entry:.1f}: `{(zscore > entry).mean():.1%}`")
        st.write(f"% < -{entry:.1f}: `{(zscore < -entry).mean():.1%}`")

    # Signal frequency
    st.markdown('<div class="section-header">Signal Frequency</div>', unsafe_allow_html=True)
    sig = kalman_full.signals_df["signal"]
    signal_counts = sig.value_counts().sort_index()
    labels_map = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    colors_map = {-1: COLORS["short"], 0: COLORS["flat"], 1: COLORS["long"]}

    fig2 = go.Figure()
    for val in [-1, 0, 1]:
        count = signal_counts.get(val, 0)
        fig2.add_trace(go.Bar(
            x=[labels_map[val]], y=[count],
            marker_color=colors_map[val],
            name=labels_map[val],
            text=[f"{count:,}"], textposition="outside",
        ))
    fig2.update_layout(
        title="Signal Distribution (Full Series)",
        yaxis_title="Days",
        height=320,
        showlegend=False,
        **CHART_TEMPLATE["layout"].to_plotly_json(),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # In-sample vs OOS performance gap
    st.markdown('<div class="section-header">In-Sample vs Out-of-Sample Performance Gap</div>', unsafe_allow_html=True)
    train_m = results["kalman_train"].metrics
    test_m = results["kalman_test"].metrics

    gap_data = []
    for key in ["CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Profit Factor"]:
        t_val = train_m[key]
        o_val = test_m[key]
        gap = o_val - t_val
        gap_data.append({
            "Metric": key,
            "Train (In-Sample)": f"{t_val:.4f}",
            "Test (OOS)": f"{o_val:.4f}",
            "Gap": f"{gap:+.4f}",
        })

    gap_df = pd.DataFrame(gap_data)
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

    # Gap bar chart
    metrics_list = ["CAGR", "Sharpe Ratio", "Win Rate"]
    fig3 = go.Figure()
    for metric in metrics_list:
        fig3.add_trace(go.Bar(
            x=[metric], y=[train_m[metric]],
            name="Train", marker_color=COLORS["kalman"],
            text=[f"{train_m[metric]:.3f}"], textposition="outside",
        ))
        fig3.add_trace(go.Bar(
            x=[metric], y=[test_m[metric]],
            name="Test (OOS)", marker_color=COLORS["ols"],
            text=[f"{test_m[metric]:.3f}"], textposition="outside",
        ))
    fig3.update_layout(
        title="In-Sample vs OOS — Key Metrics",
        barmode="group",
        height=350,
        **CHART_TEMPLATE["layout"].to_plotly_json(),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# PAGE 4: Daily Signal
# ──────────────────────────────────────────────────────────────────

def page_daily_signal(results: dict, params: dict):
    kalman_full = results["kalman_full"]
    sdf = kalman_full.signals_df

    # Current state
    latest = sdf.iloc[-1]
    current_zscore = latest["zscore"]
    current_signal = int(latest["signal"])
    signal_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[current_signal]
    signal_css = {1: "signal-long", -1: "signal-short", 0: "signal-flat"}[current_signal]

    st.markdown('<div class="section-header">Today\'s Signal</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(
            metric_card("Current Z-Score", f"{current_zscore:.4f}",
                        "metric-good" if abs(current_zscore) < 1 else "metric-bad"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Current Signal</div>'
                    f'<div style="margin-top:0.5rem;"><span class="{signal_css}">{signal_label}</span></div></div>',
                    unsafe_allow_html=True)
    with col3:
        entry = params["entry_threshold"]
        exit_ = params["exit_threshold"]
        st.markdown(
            metric_card(
                "Entry / Exit Levels",
                f"±{entry:.1f} / ±{exit_:.1f}",
                "metric-neutral",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Z-score gauge
    st.markdown('<div class="section-header">Z-Score Gauge</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_zscore,
        title={"text": "Current Z-Score", "font": {"color": "#c4b5fd"}},
        number={"font": {"color": "#e0e0ff", "size": 36}},
        gauge={
            "axis": {"range": [-4, 4], "tickwidth": 1, "tickcolor": "#6b7280"},
            "bar": {"color": COLORS["kalman"]},
            "bgcolor": "rgba(26,26,46,0.8)",
            "borderwidth": 2,
            "bordercolor": "rgba(99,102,241,0.3)",
            "steps": [
                {"range": [-4, -entry], "color": "rgba(74,222,128,0.2)"},
                {"range": [-entry, -exit_], "color": "rgba(250,204,21,0.15)"},
                {"range": [-exit_, exit_], "color": "rgba(107,114,128,0.15)"},
                {"range": [exit_, entry], "color": "rgba(250,204,21,0.15)"},
                {"range": [entry, 4], "color": "rgba(248,113,113,0.2)"},
            ],
            "threshold": {
                "line": {"color": "#e0e0ff", "width": 3},
                "thickness": 0.8,
                "value": current_zscore,
            },
        },
    ))
    fig_gauge.update_layout(
        height=280,
        paper_bgcolor="rgba(15,12,41,0.6)",
        font=dict(color="#c4b5fd"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Recent z-score trend
    st.markdown('<div class="section-header">Recent Z-Score Trend (Last 60 Days)</div>', unsafe_allow_html=True)
    recent = sdf.tail(60)
    fig_recent = go.Figure()
    fig_recent.add_trace(go.Scatter(
        x=recent["date"], y=recent["zscore"],
        mode="lines+markers",
        line=dict(color=COLORS["kalman"], width=2),
        marker=dict(size=4),
        name="Z-Score",
    ))
    # Threshold lines
    for y_val, color, name in [
        (entry, "#f87171", f"+{entry}"),
        (-entry, "#4ade80", f"-{entry}"),
        (exit_, "#facc15", f"+{exit_}"),
        (-exit_, "#facc15", f"-{exit_}"),
    ]:
        fig_recent.add_hline(y=y_val, line=dict(color=color, dash="dash", width=1),
                             annotation_text=name, annotation_position="right")
    fig_recent.add_hline(y=0, line=dict(color="#6b7280", dash="dot", width=1))
    fig_recent.update_layout(
        title="Z-Score — Last 60 Trading Days",
        height=350,
        **CHART_TEMPLATE["layout"].to_plotly_json(),
    )
    st.plotly_chart(fig_recent, use_container_width=True)

    # Trade log
    st.markdown('<div class="section-header">Recent Trade Log (Last 20 Signals Changes)</div>', unsafe_allow_html=True)

    # Find signal changes
    sig_changes = sdf[sdf["signal"].diff().ne(0)].tail(20).copy()
    sig_changes["action"] = sig_changes["signal"].map({1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ EXIT"})
    sig_changes["z-score"] = sig_changes["zscore"].round(4)

    display_df = sig_changes[["date", "action", "z-score", "spread"]].copy()
    display_df["spread"] = display_df["spread"].round(4)
    display_df = display_df.sort_values("date", ascending=False)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────

def main():
    params = render_sidebar()

    # Header
    st.markdown(
        '<div class="main-header">'
        '<h1>📈 Kalman Filter Forecasting System</h1>'
        '<p>Dynamic hedge ratio estimation • Cointegrated pairs • Statistical arbitrage signals</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load data
    try:
        df, coint_result = cached_load_data(params["seed"], params["n_days"])
    except ValueError as e:
        st.error(f"🚨 Pipeline Error: {e}")
        st.stop()

    # Run backtest
    results = cached_backtest(
        params["seed"],
        params["n_days"],
        params["delta"],
        params["R"],
        params["entry_threshold"],
        params["exit_threshold"],
        params["txn_cost_bps"],
        params["slippage_bps"],
    )

    # Navigation
    pages = {
        "📐 Model State": "model_state",
        "📊 Performance vs Baseline": "performance",
        "🩺 Forecast Health Checks": "health_checks",
        "📡 Daily Signal": "daily_signal",
    }

    selected = st.radio(
        "Navigate",
        list(pages.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    page_key = pages[selected]
    if page_key == "model_state":
        page_model_state(df, results)
    elif page_key == "performance":
        page_performance(results)
    elif page_key == "health_checks":
        page_health_checks(coint_result, results, params)
    elif page_key == "daily_signal":
        page_daily_signal(results, params)


if __name__ == "__main__":
    main()
