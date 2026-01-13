import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

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
# NSGA-II (FAST DEMO)
# ----------------------------
def nsga2(processing_times, pop_size=10):
    n_jobs = processing_times.shape[0]
    population = [random.sample(range(n_jobs), n_jobs) for _ in range(pop_size)]

    solutions = []
    for individual in population:
        m, w, s = evaluate_schedule(individual, processing_times)
        solutions.append({
            "sequence": individual,
            "makespan": m,
            "waiting": w,
            "schedule": s
        })
    return solutions

# ----------------------------
# Fitness Function (GOAL)
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
# Pareto Plot
# ----------------------------
def plot_pareto(solutions):
    m = [s["makespan"] for s in solutions]
    w = [s["waiting"] for s in solutions]

    fig, ax = plt.subplots()
    ax.scatter(m, w)
    ax.set_xlabel("Makespan")
    ax.set_ylabel("Waiting Time")
    ax.set_title("Pareto Front (Trade-off)")
    st.pyplot(fig)

# ----------------------------
# Gantt Chart
# ----------------------------
def plot_gantt(schedule):
    fig, ax = plt.subplots()

    colors = {}
    y_pos = {}

    machines = sorted(set(s["machine"] for s in schedule))
    for i, m in enumerate(machines):
        y_pos[m] = i

    for s in schedule:
        if s["job"] not in colors:
            colors[s["job"]] = np.random.rand(3,)
        ax.barh(
            y_pos[s["machine"]],
            s["end"] - s["start"],
            left=s["start"],
            color=colors[s["job"]]
        )

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Gantt Chart (Schedule Visualization)")
    st.pyplot(fig)

# ----------------------------
# STREAMLIT UI
# --------------------------
