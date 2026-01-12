import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

st.title("Interactive NSGA-II Job Scheduling")

# --- Sidebar Parameters ---
st.sidebar.header("NSGA-II Parameters")
population_size = st.sidebar.number_input("Population Size", min_value=5, max_value=100, value=20, step=1)
generations = st.sidebar.number_input("Number of Generations", min_value=1, max_value=200, value=30, step=1)
crossover_prob = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.9, 0.01)
mutation_prob = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.1, 0.01)

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your job processing CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.subheader("Processing Times")
    st.dataframe(df)

    processing_times = df.to_numpy()
    n_jobs, n_machines = processing_times.shape

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

    # --- Run Simulation Button ---
    if st.button("Run Simulation"):
        population = initialize_population()
        for gen in range(generations):
            fitness = [evaluate(ind) for ind in population]
            makespans = [f[0] for f in fitness]
            tardiness = [f[1] for f in fitness]

        # --- Pareto Front Line Graph ---
        sorted_indices = np.argsort(makespans)
        sorted_makespan = np.array(makespans)[sorted_indices]
        sorted_tardiness = np.array(tardiness)[sorted_indices]

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(sorted_makespan, sorted_tardiness, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Total Tardiness")
        ax.set_title("Pareto Front (Line Graph)")
        st.pyplot(fig)

        # --- Gantt Chart for Best Schedule ---
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

        st.success("Simulation Completed!")
