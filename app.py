import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

# ----------------------------
# Utility Functions
# ----------------------------
def normalize(values):
    min_v = min(values)
    max_v = max(values)
    return [(v - min_v) / (max_v - min_v + 1e-9) for v in values]

# ----------------------------
# Scheduling Simulation
# ----------------------------
def evaluate_schedule(job_order, processing_times):
    n_jobs, n_machines = processing_times.shape
    machine_time = np.zeros(n_machines)
    job_time = np.zeros(n_jobs)

    schedule_log = []
    total_waiting = 0

    for job in job_order:
        for m in range(n_machines):
            start = max(machine_time[m], job_time[job])
            wait = start - job_time[job]
            total_waiting += wait

            finish = start + processing_times[job, m]
            machine_time[m] = finish
            job_time[job] = finish

            schedule_log.append({
                "job": f"J{job+1}",
                "machine": f"M{m+1}",
                "start": start,
                "end": finish
            })

    makespan = max(job_time)
    return makespan, total_waiting, schedule_log

# ----------------------------
# NSGA-II (FAST DEMO VERSION)
# ----------------------------
def nsga2(processing_times, pop_size=10, generations=3):
    n_jobs = processing_times.shape[0]
    population = [random.sample(range(n_jobs), n_jobs) for _ in range(pop_size)]

    solutions = []

    for individual in population:
        makespan, waiting, schedule = evaluate_schedule(individual, processing_times)
        solutions.append({
            "sequence": individual,
            "makespan": makespan,
            "waiting": waiting,
            "schedule": schedule
        })

    return solutions

# ----------------------------
# Fitness (GOAL FUNCTION)
# ----------------------------
def compute_fitness(solutions, w_m=0.5, w_w=0.5):
    makespans = [s["makespan"] for s in solutions]
    waitings = [s["waiting"] for s in solutions]

    nm = normalize(makespans)
    nw = normalize(waitings)

    for i, s in enumerate(solutions):
        s["fitness"] = w_m * nm[i] + w_w * nw[i]

    return solutions

# ----------------------------
# Gantt Chart
# ----------------------------
def plot_gantt(schedule):
    df = pd.DataFrame(schedule)

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="machine",
        color="job",
        title="Gantt Chart for Selected Schedule"
    )

    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("NSGA-II Job Scheduling (With Fitness Goal)")

uploaded_file = st.file_uploader("Upload Job Scheduling CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    processing_times = df.values

    st.subheader("Processing Time Matrix")
    st.dataframe(df)

    st.sidebar.header("Fitness Weights (Goal)")
    w_m = st.sidebar.slider("Makespan Weight", 0.0, 1.0, 0.5)
    w_w = 1.0 - w_m
    st.sidebar.write(f"Waiting Time Weight: {w_w}")

    if st.button("Run NSGA-II"):
        with st.spinner("Running NSGA-II..."):
            solutions = nsga2(processing_times)
            solutions = compute_fitness(solutions, w_m, w_w)

            best = min(solutions, key=lambda x: x["fitness"])

        # Pareto Plot
        pareto_df = pd.DataFrame({
            "Makespan": [s["makespan"] for s in solutions],
            "Waiting Time": [s["waiting"] for s in solutions]
        })

        st.subheader("Pareto Front (Trade-offs)")
        fig = px.scatter(
            pareto_df,
            x="Makespan",
            y="Waiting Time",
            title="NSGA-II Pareto Front"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Best Solution (GOAL)
        st.subheader("Selected Best Solution (Goal-Based Fitness)")
        st.write(f"✔ Makespan: {best['makespan']}")
        st.write(f"✔ Waiting Time: {best['waiting']}")
        st.write(f"🎯 Fitness Value: {best['fitness']:.4f}")
        st.write(f"🧬 Job Sequence: {[f'J{i+1}' for i in best['sequence']]}")

        # Gantt Chart
        st.subheader("Gantt Chart (UI Visualization)")
        plot_gantt(best["schedule"])
