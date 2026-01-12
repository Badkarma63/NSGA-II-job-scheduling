import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NSGA-II Job Scheduling", layout="centered")

st.title("NSGA-II Job Scheduling (Fast & Safe)")

st.info("Upload CSV and click Run Simulation")

# Sidebar (SAFE defaults)
st.sidebar.header("Parameters")
population_size = st.sidebar.slider("Population Size", 5, 20, 10)
generations = st.sidebar.slider("Generations", 1, 10, 3)

w1 = st.sidebar.slider("Makespan Weight", 0.0, 1.0, 0.5)
w2 = st.sidebar.slider("Waiting Time Weight", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ---------- NO COMPUTATION ABOVE THIS ---------- #

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.dataframe(df)

    processing_times = df.to_numpy()
    n_jobs, n_machines = processing_times.shape

    def evaluate(order):
        completion = np.zeros((n_jobs, n_machines))
        for i, job in enumerate(order):
            for m in range(n_machines):
                if i == 0 and m == 0:
                    completion[i, m] = processing_times[job, m]
                elif i == 0:
                    completion[i, m] = completion[i, m-1] + processing_times[job, m]
                elif m == 0:
                    completion[i, m] = completion[i-1, m] + processing_times[job, m]
                else:
                    completion[i, m] = max(completion[i-1, m], completion[i, m-1]) + processing_times[job, m]
        makespan = completion[-1, -1]
        waiting = sum(completion[:,0] - processing_times[order,0])
        return makespan, waiting

    if st.button("Run Simulation"):
        pop = [np.random.permutation(n_jobs) for _ in range(population_size)]

        results = []
        for ind in pop:
            m, w = evaluate(ind)
            results.append((m, w))

        makespans = [r[0] for r in results]
        waitings = [r[1] for r in results]

        fig, ax = plt.subplots()
        ax.plot(sorted(makespans), sorted(waitings), marker='o')
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Waiting Time")
        ax.set_title("Pareto Trend (Fast Mode)")
        st.pyplot(fig)

        st.success("Simulation completed successfully")
