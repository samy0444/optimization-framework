# app.py
import streamlit as st
import numpy as np
import time
import pandas as pd
# from simulation import simulate_all
from rl_bo_pipeline.tri_loop import TriLoop
from layers import Layer
from benchmark_rl_bo import benchmark_all

from interfaces import Interface

import altair as alt



default_layers = [
    Layer('Substrate', 1000, 0, 0, 0, 0),
    Layer('ETL',       50, 1e-12, 0.5, 1e9, 1e-5, mobility=1e-3, band_offset=0.2),
    Layer('Perov',    400, 1e-12, 0.5, 1e9, 1e-5),
    Layer('HTL',      100,1e-12, 0.5, 1e9, 1e-5, mobility=5e-4, band_offset=0.3),
    Layer('Ag',        80,   0,   0,    0,    0),
]
st.set_page_config(layout="wide")
st.title("🔆 Self-Driving Solar Lab Dashboard")

# ────────────────────────────────
# 1. User‐driven Single Experiment
# ────────────────────────────────
st.sidebar.header("1. Manual Fabrication")

cs = st.sidebar.slider("Cs fraction", 0.0, 1.0, 0.33, 0.01)
ma = st.sidebar.slider("MA fraction", 0.0, 1.0 - cs, 0.33, 0.01)
fa = 1.0 - cs - ma
st.sidebar.markdown(f"**FA fraction** = {fa:.2f}")

spin = st.sidebar.slider("Spin speed (rpm)", 1000, 6000, 3500, 100)
anneal = st.sidebar.slider("Anneal temp (°C)", 50, 200, 125, 5)
uv = st.sidebar.slider("UV filter strength", 0.0, 1.0, 0.5, 0.01)
encap = st.sidebar.checkbox("Encapsulation", value=False)

etl_if  = Interface(name="ETL/Perovskite", recomb_velocity=1e3, trap_density=1e10)
perh_if = Interface(name="Perovskite/HTL", recomb_velocity=5e2, trap_density=5e9)



if st.sidebar.button("Run Simulation"):
    with st.spinner("Simulating…"):
        params = {
            "layers": default_layers, 
            "composition": {"Cs": cs, "MA": ma, "FA": fa},
            "uv_filter": uv,
            "spin_speed": spin,
            "anneal_T": anneal,
            "substrate": "Glass",
            "encapsulation": encap,
            "interfaces": [etl_if, perh_if]
        }
        res = simulate_all(params)
    st.metric("💡 Efficiency", f"{res['efficiency']:.3f}")
    st.metric("⏳ log₁₀(Lifetime)", f"{res['log_lifetime']:.3f}")

# ────────────────────────────────────
# 2. Closed‐Loop BO + RL “Tri-Loop”
# ────────────────────────────────────
st.sidebar.header("2. Automated BO + RL Optimization")

n_cycles   = st.sidebar.number_input("BO+RL cycles", min_value=1, max_value=20, value=5)
batch_size = st.sidebar.number_input("Compositions per cycle", min_value=1, max_value=10, value=3)
rl_steps   = st.sidebar.number_input("RL timesteps", min_value=1000, max_value=100_000, value=20_000, step=1000)

if st.sidebar.button("Run Tri-Loop"):
    # placeholders for dynamic updates
    chart_pl = st.empty()
    table_pl = st.empty()
    progress  = st.progress(0)
    loop = TriLoop(data_root="cluster-perovskite-data/Data", bo_device="cpu")

    history = []
    for c in range(n_cycles):
        # run one cycle: suggest + train + evaluate
        cycle_results = loop.run_cycle(cycle=c, bo_batch=batch_size, rl_steps=rl_steps)
        history.extend(cycle_results)

        # render table of all results so far
        df = pd.DataFrame(history)
        table_pl.dataframe(df[["cycle","composition","eff","log_life"]])

        # plot live
        df_plot = (df
            .assign(eff=df["eff"], log_life=df["log_life"])
            .groupby("cycle")[["eff","log_life"]]
            .max()
            .reset_index()
            .set_index("cycle")
        )
        chart_pl.line_chart(df_plot)

        progress.progress((c+1)/n_cycles)

    st.success("✅ Optimization complete!")




#     n_repeats    = 5  # or however many you want
if st.sidebar.button("Run Enhanced Benchmarks"):
    st.info("Running robust benchmarking… this may take a minute.")
    start = time.time()

    # publication-ready settings:
    total_cycles = int(n_cycles)       # from sidebar, e.g. 10
    bo_batch     = int(batch_size)     # e.g. 5
    rl_steps     = int(rl_steps)       # e.g. 50000
    n_repeats    = 1                # do 10 independent trials
    seed         = 42                # keep it reproducible

    # run
    results, stats, df_mean = benchmark_all(
        total_cycles=total_cycles,
        bo_batch=bo_batch,
        rl_steps=rl_steps,
        n_repeats=n_repeats,
        seed=seed,
    )

    st.success(f"Done in {(time.time()-start):.1f}s")



    # Build a DataFrame for plotting mean ± std
    plot_df = []
    strategies = list(stats.keys())
    for strat in strategies:
        mean = stats[strat]["mean"]
        std  = stats[strat]["std"]
        for cycle in range(total_cycles):
            plot_df.append({
                "strategy": strat,
                "cycle":     cycle + 1,
                "mean":      mean[cycle],
                "lower":     mean[cycle] - std[cycle],
                "upper":     mean[cycle] + std[cycle],
            })
    df_plot = pd.DataFrame(plot_df)

    # Altair chart: band for ±1 std and line for mean
    base = alt.Chart(df_plot).encode(
        x=alt.X("cycle:Q", title="Cycle"),
        color=alt.Color("strategy:N", title="Method")
    )

    band = base.mark_area(opacity=0.2).encode(
        y="lower:Q",
        y2="upper:Q"
    )

    line = base.mark_line(point=True).encode(
        y=alt.Y("mean:Q", title="Efficiency × Lifetime")
    )

    chart = (band + line).properties(
        width=700,
        height=400,
        title="Benchmark Convergence (mean ± 1 std over repeats)"
    )

    st.altair_chart(chart, use_container_width=True)

    # Show raw mean table and cycles-to-threshold
    st.subheader("Mean Performance by Cycle")
    st.dataframe(df_mean)

    threshold = st.sidebar.number_input(
        "Threshold (eff × life)", 0.0, 5.0, 1.0, 0.1
    )
    hits = []
    for strat in strategies:
        times = []
        for rep in range(n_repeats):
            arr = results[strat][rep]
            hit = np.argmax(arr >= threshold) + 1 if np.any(arr >= threshold) else np.inf
            times.append(hit)
        hits.append({
            "method": strat,
            "mean_cycles_to_threshold": np.mean(times),
            "std_cycles_to_threshold":  np.std(times),
        })
    df_hits = pd.DataFrame(hits)
    st.subheader(f"Cycles to reach ≥ {threshold:.2f}")
    st.table(df_hits)


# ────────────────────────────────
# 3. Summary of Best Recipe
# ────────────────────────────────
st.header("🏆 Best Overall Recipe Found")
if 'history' in locals() and history:
    best = max(history, key=lambda x: (x["eff"], x["log_life"]))
    comp = best["composition"]
    st.markdown(f"**Composition:** Cs={comp['Cs']:.2f}, MA={comp['MA']:.2f}, FA={comp['FA']:.2f}")
    st.markdown(f"**Efficiency:** {best['eff']:.3f}")
    st.markdown(f"**log₁₀(Lifetime):** {best['log_life']:.3f}")
else:
    st.write("Run the Tri-Loop above to see results here.")
