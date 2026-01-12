import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

st.title("NSGA-II Job Scheduling Visualization")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your job processing CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.write("Processing Times:")
    st.dataframe(df)

    processing_times = df.to_numpy()
    n_jobs, n_machines = processing_times.shape

    # --- NSGA-II Parameters ---
    population_size = 20
    generations = 30  # For demo, you can increase
    crossover_prob = 0.9
    mutation_prob = 0.1

    # --- Individual Representation ---
    def create_individual():
        return np.random.permutation(n_jobs)

    def initialize_population():
        return [create_individual() for _ in range(population_size)]

    # --- Fitness / Objectives ---
    def evaluate(individual):
        completion_time = np.zeros((n_jobs, n_machines))
        for i, job in enumerate(individual):
            for m in range(n_machines):
                if i == 0 and m == 0:
                    completion_time[i, m] = processing_times[job, m]
                elif i == 0:
                    completion_time[i, m] = completion_time[i, m-1] + processing_times[job, m]
                elif m == 0:
                    completion_time[i, m] = completion_time[i-1, m] + processing_times[job, m]
                else:
                    completion_time[i, m] = max(completion_time[i-1, m], completion_time[i, m-1]) + processing_times[job, m]
        makespan = completion_time[-1, -1]
        total_tardiness = sum([job * processing_times[job, -1] for job in individual])
        return makespan, total_tardiness, completion_time

    # --- NSGA-II Simple Run (No Crossover/Mutation for Demo) ---
    population = initialize_population()
    for gen in range(generations):
        # Evaluate population
        fitness = [evaluate(ind) for ind in population]
        makespans = [f[0] for f in fitness]
        tardiness = [f[1] for f in fitness]

    # --- Pareto Front Plot ---
    fig, ax = plt.subplots()
    ax.scatter(makespans, tardiness, c='blue')
    ax.set_xlabel("Makespan")
    ax.set_ylabel("Total Tardiness")
    ax.set_title("Pareto Front (NSGA-II Demo)")
    st.pyplot(fig)

    # --- Gantt Chart for Best Schedule (min Makespan) ---
    best_index = np.argmin(makespans)
    best_individual = population[best_index]
    _, _, completion_time = evaluate(best_individual)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab20.colors
    for j, job in enumerate(best_individual):
        start_times = [0] + list(completion_time[j, :-1])
        durations = completion_time[j] - start_times
        for m in range(n_machines):
            ax2.barh(f"Machine {m+1}", durations[m], left=start_times[m], color=colors[j % len(colors)], edgecolor='black')
            ax2.text(start_times[m]+durations[m]/2, m, f"J{job+1}", va='center', ha='center', color='white', fontsize=8)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Machines")
    ax2.set_title("Gantt Chart of Best Schedule (Min Makespan)")
    st.pyplot(fig2)

    st.success("NSGA-II Job Scheduling Demo Completed!")
