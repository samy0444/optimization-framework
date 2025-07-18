# ui_app.py (Streamlit)
import streamlit as st
import pandas as pd
from optimization_bo import run_bo
from train_rl import train
from environment import FabricationEnv
from stable_baselines3 import PPO
from surrogate_model import SurrogateModel
from stable_baselines3 import SAC
# from stable_baselines3 import A2C
from validation.validate_agent import validate_virtual_lab
from simulation import simulate_all
from noisy_simulation.noisy_wrapper import NoisySimulator 
import streamlit as st




st.title("Self‑Driving Solar Lab Simulator")
b_iters=st.sidebar.slider("BO iterations",5,50,20)
init_size=st.sidebar.number_input("Initial DOE size",5,50,20)
rl_steps=st.sidebar.slider("RL timesteps",1000,100000,20000)
surrogate = SurrogateModel(device="cpu")
if st.sidebar.button("Run BO"):
    st.info("Running multi‑objective BO…")
    hist=run_bo(b_iters, init_size)
    df=pd.DataFrame(hist, columns=["Best Eff","Best Log‑Life"] )
    st.line_chart(df)

# usage in ui_app.py (Streamlit)

if st.sidebar.button("Run Hierarchical BO"):
    from rl_bo_pipeline.physics_informed_optimization_bo import run_bo
    st.info("Running material-level BO + device sampling…")
    comp_hist, inst_hist = run_bo(
        num_rounds=5,
        material_batch=10
    )
    st.write("Material compositions suggested each round:")
    for i, (comps, insts) in enumerate(zip(comp_hist, inst_hist)):
        st.write(f"Round {i+1}:")
        df = pd.DataFrame({
            "Cs": comps[:,0],
            "MA": comps[:,1],
            "FA": comps[:,2],
            "pred_log10_life": insts
        })
        st.dataframe(df)


# if st.sidebar.button("Run Hierarchical BO"):
#     from physics_informed_optimization_bo import run_bo
#     st.info("Running material-level BO + device sampling…")
#     comp_hist, inst_hist = run_bo(
#         num_rounds=5,
#         material_batch=10
#     )
#     st.write("Material compositions suggested each round:")
#     for i,(comps,insts) in enumerate(zip(comp_hist, inst_hist)):
#         st.write(f"Round {i+1}:")
#         st.dataframe({
#             "Cs": comps[:,0],
#             "MA": comps[:,1],
#             "FA": comps[:,2],
#             "instability": insts
#         })



def anneal_time_penalty(T):
    return 0.1 * (T / 100.0)
if st.sidebar.button("Train RL"):
    st.info("Training RL policy…")
    # train(rl_steps)
    # validate_virtual_lab()
    # 1) train and grab the agent
    agent = train(rl_steps)

    # 2) build the same noisy‐simulator you use elsewhere


    sim = NoisySimulator(simulate_all, noise_level=0.05,
                         time_penalties={"anneal_T": anneal_time_penalty})

    # 3) validate your lab loop
    score = validate_virtual_lab(agent, sim)
    st.write(f"Validation success rate: {score*100:.1f}%")
    st.success("RL done")

if st.sidebar.button("Demo RL"):
    env=FabricationEnv()
    model=SAC.load("sac_fab_cont", env=env)
    
    obs=env.reset()
    for i in range(6):
        action,_=model.predict(obs)
        obs, r, done, info=env.step(action)
        st.write(f"Stage {i}: action={action}")
    st.write("Result:", info)


# if st.sidebar.button("Run Digital-Twin"):
#     st.info("Running digital‐twin closed‐loop…")
#     # load or initialize
#     from orchestrator.closed_loop import DigitalTwinLoop
#     from noisy_simulation.noisy_wrapper import NoisySimulator
#     from surrogate_model import SurrogateModel
#     from train_rl import train  # to get agent

#     # 1) seed data_pool from some initial random experiments
#     data_pool = initial_random_data(100)

#     # 2) build sim & surrogate & agent
#     base_sim = simulate_all
#     sim = NoisySimulator(base_sim, noise_level=0.05,
#                          time_penalties={"anneal_T": anneal_time_penalty})
#     surrogate = SurrogateModel(device="cpu")
#     agent = SAC.load("sac_fab_cont", env=env)r train(rl_steps)

#     # 3) run a few closed-loop cycles
#     loop = DigitalTwinLoop(agent, sim, surrogate, batch_size=32)
#     metrics = []
#     for cycle in range(5):
#         data_pool = loop.run_cycle(data_pool)
#         metrics.append((cycle,
#                         data_pool.efficiency.max(),
#                         data_pool.log_lifetime.max()))
#     df = pd.DataFrame(metrics, columns=["cycle","best_eff","best_life"])
#     st.line_chart(df.set_index("cycle"))


if st.sidebar.button("Run Digital-Twin"):
    st.info("Running digital-twin closed-loop…")
    from orchestrator.closed_loop import DigitalTwinLoop
    from noisy_simulation.noisy_wrapper import NoisySimulator
    from surrogate_model import SurrogateModel
    from train_rl import train
    from orchestrator.utils import initial_random_data  # <-- import it
    env=FabricationEnv()
    data_pool = initial_random_data(100)

    sim = NoisySimulator(simulate_all,
                         noise_level=0.05,
                         time_penalties={"anneal_T": anneal_time_penalty})
    surrogate = SurrogateModel(device="cpu")
    agent = SAC.load("sac_fab_cont", env=env)

    # loop = DigitalTwinLoop(agent, sim, surrogate, batch_size=32)
    # new:
    loop = DigitalTwinLoop(agent, sim, batch_size=32)

    metrics = []
    for cycle in range(1):
        data_pool = loop.run_cycle(data_pool)
        metrics.append((cycle,
                        data_pool.efficiency.max(),
                        data_pool.log_lifetime.max()))
    df = pd.DataFrame(metrics, columns=["cycle", "best_eff", "best_life"])
    st.line_chart(df.set_index("cycle"))


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from environment import FabricationEnv

if st.sidebar.button("Evaluate RL"):
    env=FabricationEnv()
    st.info("Evaluating trained policy over 30 episodes…")
    # 1) Load your saved model
    model = SAC.load("sac_fab_cont", env=env)

    # 2) Create a fresh environment
    env = FabricationEnv()

    # 3) Run 30 full episodes, return mean & std of total episode reward
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=30,
        deterministic=True,       # use deterministic actions (no random sampling)
        render=False,             # set True if you want to visualize
    )

    st.success(f"Mean reward over 30 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

# app.py snippet

from rl_bo_pipeline.tri_loop import TriLoop
# loop = TriLoop()\# history = loop.run(num_cycles=3, bo_batch=5, rl_steps=10000)


if st.sidebar.button("Run Tri‐Loop"):
    st.info("Running BO + RL closed loop…")
    loop = TriLoop()
    history = loop.run(num_cycles=3, bo_batch=5, rl_steps=10000)
    # hist = tri_loop(num_cycles=3, bo_batch=3, rl_steps=10000)
    df = pd.DataFrame(history)
    st.write(df)
    st.line_chart(df.set_index("cycle")[["eff","log_life"]])


if st.sidebar.button("Run Benchmarks"):
    st.info("Running all four benchmarking strategies… this may take a while.")
    start = time.time()
    df_bench = benchmark_all(
        total_cycles=int(n_cycles),
        bo_batch=int(batch_size),
        rl_steps=int(rl_steps),
        random_seed=42
    )
    st.success(f"Done in {time.time()-start:.1f}s")

    st.subheader("Convergence Comparison")
    st.line_chart(df_bench)

    st.subheader("Raw Data")
    st.dataframe(df_bench)

    # Optionally compute how many cycles to exceed threshold
    threshold = st.sidebar.number_input("Performance threshold (eff×life)", 0.0, 5.0, 1.0, 0.1)
    secs = []
    for col in df_bench.columns:
        reached = df_bench[col].ge(threshold).idxmax() if (df_bench[col] >= threshold).any() else None
        secs.append(reached)
    df_t = pd.DataFrame({"method":df_bench.columns, "cycles_to_threshold":secs})
    st.subheader(f"Cycles to reach eff×life ≥ {threshold:.1f}")
    st.table(df_t)

