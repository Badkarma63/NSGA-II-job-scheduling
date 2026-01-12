import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="NSGA-II Job Scheduling", layout="centered")

st.title("NSGA-II Job Scheduling (Fast Mode)")
st.write("Upload job processing time CSV and run simulation.")

# ---------------- Sidebar ----------------
st.sidebar.header("Simulation Parameters")

population_size = st.sidebar.slider("Population Size", 5, 20, 10)
generations = st.sidebar.slider("Generations (Demo)", 1, 10, 3)

w1 = st.sidebar.slider("Makespan Weight", 0.0, 1.0, 0.5)
w2 = st.sidebar.slider("Waiting Time Weight", 0.0, 1.0, 0.5)

# ---------------- Upload CSV ----------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ---------------- No Heavy Code Above ----------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.subheader("Processing Time Matrix")
    st.dataframe(df)

    processing_times = df.to_numpy()
    n_jobs, n_machines = processing_times.shape

    # ---------------- Fitness Evaluation ----------------
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
        waiting_time = sum(completion[:, 0] - processing_times[order, 0])

        return makespan, waiting_time, completion

    # ---------------- Run Simulation ----------------
    if st.button("Run Simulation"):
        population = [np.random.permutation(n_jobs) for _ in range(population_size)]

        results = []
        for ind in population:
            m, w, comp = evaluate(ind)
            weighted = w1 * m + w2 * w
            results.append((ind, m, w, weighted, comp))

        # ---------------- Pareto Line Graph ----------------
        makespans = [r[1] for r in results]
        waitings = [r[2] for r in results]

        fig, ax = plt.subplots()
        ax.plot(sorted(makespans), sorted(waitings), marker='o')
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Job Waiting Time")
        ax.set_title("Pareto Trend (Fast Mode)")
        st.pyplot(fig)

        # ---------------- Best Solution ----------------
        best = min(results, key=lambda x: x[3])
        best_order, best_m, best_w, _, best_completion = best

        st.subheader("Best Job Sequence (Weighted Objective)")
        st.write([f"J{j+1}" for j in best_order])
        st.write(f"Makespan: {best_m}")
        st.write(f"Waiting Time: {best_w}")

        # ---------------- Gantt Chart ----------------
        st.subheader("Gantt Chart of Best Schedule")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10.colors

        for i, job in enumerate(best_order):
            for m in range(n_machines):
                start = best_completion[i, m] - processing_times[job, m]
                ax2.barh(
                    f"M{m+1}",
                    processing_times[job, m],
                    left=start,
                    color=colors[i % len(colors)],
                    edgecolor='black'
                )
                ax2.text(
                    start + processing_times[job, m] / 2,
                    m,
                    f"J{job+1}",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=8
                )

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Machines")
        ax2.set_title("Gantt Chart")
        st.pyplot(fig2)

        st.success("Simulation completed successfully ✅")
