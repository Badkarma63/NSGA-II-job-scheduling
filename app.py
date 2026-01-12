import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

st.title("Full NSGA-II Job Scheduling App with Weighted Objectives")

# --- Sidebar Parameters ---
st.sidebar.header("NSGA-II Parameters")
population_size = st.sidebar.number_input("Population Size", 5, 100, 50, 1)
generations = st.sidebar.number_input("Number of Generations", 1, 200, 50, 1)
crossover_prob = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.9, 0.01)
mutation_prob = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.1, 0.01)

st.sidebar.header("Objective Weights")
makespan_weight = st.sidebar.slider("Makespan Weight", 0.0, 1.0, 0.5, 0.01)
waiting_weight = st.sidebar.slider("Job Waiting Time Weight", 0.0, 1.0, 0.5, 0.01)

# --- CSV Upload ---
uploaded_file = st.file_uploader("Upload your Job Processing CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.subheader("Processing Times")
    st.dataframe(df)

    processing_times = df.to_numpy()
    n_jobs, n_machines = processing_times.shape

    # --- NSGA-II Functions ---
    def create_individual():
        return np.random.permutation(n_jobs)

    def initialize_population():
        return [create_individual() for _ in range(population_size)]

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
        waiting_time = sum([completion_time[j, 0] - processing_times[individual[j], 0] for j in range(n_jobs)])
        weighted_obj = makespan_weight * makespan + waiting_weight * waiting_time
        return makespan, waiting_time, weighted_obj, completion_time

    # --- Crossover (PMX for permutation) ---
    def pmx_crossover(parent1, parent2):
        size = len(parent1)
        c1, c2 = parent1.copy(), parent2.copy()
        if size < 2:
            return c1, c2
        a, b = sorted(random.sample(range(size), 2))
        mapping1 = {c2[i]: c1[i] for i in range(a, b)}
        mapping2 = {c1[i]: c2[i] for i in range(a, b)}
        for i in range(size):
            if i < a or i >= b:
                while c1[i] in mapping1:
                    c1[i] = mapping1[c1[i]]
                while c2[i] in mapping2:
                    c2[i] = mapping2[c2[i]]
        c1[a:b], c2[a:b] = c2[a:b], c1[a:b]
        return c1, c2

    # --- Mutation (swap) ---
    def swap_mutation(individual):
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
        return individual

    # --- Tournament Selection ---
    def tournament_selection(pop, fitness, k=2):
        selected = []
        for _ in range(len(pop)):
            i1, i2 = random.sample(range(len(pop)), 2)
            if fitness[i1][2] < fitness[i2][2]:  # minimize weighted objective
                selected.append(pop[i1])
            else:
                selected.append(pop[i2])
        return selected

    # --- Run Simulation Button ---
    if st.button("Run Simulation"):
        population = initialize_population()
        for gen in range(generations):
            # Evaluate population
            fitness = [evaluate(ind) for ind in population]

            # Selection
            population = tournament_selection(population, fitness)

            # Crossover
            new_pop = []
            for i in range(0, len(population), 2):
                p1, p2 = population[i], population[i+1 if i+1 < len(population) else 0]
                if random.random() < crossover_prob:
                    c1, c2 = pmx_crossover(p1, p2)
                    new_pop.extend([c1, c2])
                else:
                    new_pop.extend([p1, p2])
            population = new_pop

            # Mutation
            population = [swap_mutation(ind) if random.random() < mutation_prob else ind for ind in population]

        # Evaluate final population
        fitness = [evaluate(ind) for ind in population]
        makespans = [f[0] for f in fitness]
        waiting_times = [f[1] for f in fitness]
        weighted_objs = [f[2] for f in fitness]

        # --- Pareto Front Line Graph ---
        sorted_indices = np.argsort(makespans)
        sorted_makespan = np.array(makespans)[sorted_indices]
        sorted_waiting = np.array(waiting_times)[sorted_indices]

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(sorted_makespan, sorted_waiting, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Makespan")
        ax.set_ylabel("Job Waiting Time")
        ax.set_title("Pareto Front (Line Graph)")
        st.pyplot(fig)

        # --- Gantt Chart for Best Schedule ---
        best_index = np.argmin(weighted_objs)
        best_individual = population[best_index]
        _, _, _, completion_time = evaluate(best_individual)

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
        ax2.set_title("Gantt Chart of Best Schedule (Weighted Objective)")
        st.pyplot(fig2)

        st.success("NSGA-II Simulation Completed!")
