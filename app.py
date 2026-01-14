import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------
# Schedule Evaluation
# ----------------------------------
def evaluate_schedule(sequence, pt):
    n_jobs, n_machines = pt.shape
    machine_time = np.zeros(n_machines)
    job_time = np.zeros(n_jobs)

    waiting = 0
    gantt = []
    machine_processing = np.zeros(n_machines)

    for job in sequence:
        for m in range(n_machines):
            start = max(machine_time[m], job_time[job])

            waiting += start - job_time[job]

            finish = start + pt[job, m]
            machine_time[m] = finish
            job_time[job] = finish
            machine_processing[m] += pt[job, m]

            gantt.append((f"M{m+1}", start, finish, f"J{job+1}"))

    makespan = max(job_time)

    # total idle time
    idle_time = np.sum(makespan - machine_processing)

    # ✅ NORMALIZATION (brings to hundreds)
    idle_time = idle_time / (n_machines * n_jobs)

    return makespan, waiting, idle_time, gantt


# ----------------------------------
# NSGA-II Core Functions
# ----------------------------------
def dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def fast_nondominated_sort(pop):
    fronts = [[]]
    for p in pop:
        p["dominated"] = []
        p["dom_count"] = 0
        for q in pop:
            if dominates(p["obj"], q["obj"]):
                p["dominated"].append(q)
            elif dominates(q["obj"], p["obj"]):
                p["dom_count"] += 1
        if p["dom_count"] == 0:
            p["rank"] = 1
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p["dominated"]:
                q["dom_count"] -= 1
                if q["dom_count"] == 0:
                    q["rank"] = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(front):
    if not front:
        return
    for p in front:
        p["distance"] = 0

    for i in range(2):
        front.sort(key=lambda x: x["obj"][i])
        front[0]["distance"] = front[-1]["distance"] = float("inf")
        min_v = front[0]["obj"][i]
        max_v = front[-1]["obj"][i]

        for j in range(1, len(front) - 1):
            front[j]["distance"] += (
                front[j + 1]["obj"][i] - front[j - 1]["obj"][i]
            ) / (max_v - min_v + 1e-9)


def tournament_selection(pop):
    a, b = random.sample(pop, 2)
    if a["rank"] < b["rank"]:
        return a
    if a["rank"] == b["rank"] and a["distance"] > b["distance"]:
        return a
    return b


def crossover(p1, p2):
    cut = random.randint(1, len(p1) - 2)
    return p1[:cut] + [j for j in p2 if j not in p1[:cut]]


def mutate(seq):
    a, b = random.sample(range(len(seq)), 2)
    seq[a], seq[b] = seq[b], seq[a]


# ----------------------------------
# NSGA-II Algorithm
# ----------------------------------
def nsga2(pt, pop_size, generations, pc, pm):
    n_jobs = pt.shape[0]
    population = []

    for _ in range(pop_size):
        seq = random.sample(range(n_jobs), n_jobs)
        m, w, idle, g = evaluate_schedule(seq, pt)
        population.append({
            "seq": seq,
            "obj": (m, w),
            "idle": idle,
            "gantt": g
        })

    for _ in range(generations):
        fronts = fast_nondominated_sort(population)
        for f in fronts:
            crowding_distance(f)

        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            child_seq = p1["seq"][:]
            if random.random() < pc:
                child_seq = crossover(p1["seq"], p2["seq"])
            if random.random() < pm:
                mutate(child_seq)

            m, w, idle, g = evaluate_schedule(child_seq, pt)
            offspring.append({
                "seq": child_seq,
                "obj": (m, w),
                "idle": idle,
                "gantt": g
            })

        population = offspring

    return population


# ----------------------------------
# Fitness Function
# ----------------------------------
def compute_total_fitness(population):
    total_fitness = 0
    for ind in population:
        m, w = ind["obj"]
        idle = ind["idle"]
        fitness = 1000 / (m + w + idle + 1e-9)
        total_fitness += fitness
    return total_fitness


# ----------------------------------
# Gantt Chart
# ----------------------------------
def plot_gantt(gantt):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    jobs = sorted(list(set([g[3] for g in gantt])))
    job_colors = {job: plt.cm.tab20(i) for i, job in enumerate(jobs)}

    for machine, start, end, job in gantt:
        ax.barh(machine, end - start, left=start,
                color=job_colors[job], edgecolor="white")
        ax.text(start + (end - start) / 2, machine, job,
                ha="center", va="center", color="white", fontsize=9)

    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Machine", color="white")
    ax.set_title("Gantt Chart – NSGA-II Optimized Schedule", color="white")
    ax.tick_params(colors="white")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)


# ----------------------------------
# STREAMLIT UI
# ----------------------------------
st.title("NSGA-II Job Scheduling with Idle Time")

uploaded = st.file_uploader("Upload CSV File", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded, index_col=0)
    pt = df.values
    st.dataframe(df)

    st.sidebar.header("Parameters")
    pop_size = st.sidebar.number_input("Population Size", 50, 200, 100)
    generations = st.sidebar.number_input("Generations", 50, 300, 100)
    pc = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.9)
    pm = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

    if st.button("Run NSGA-II"):
        with st.spinner("Running NSGA-II..."):
            pop = nsga2(pt, pop_size, generations, pc, pm)

            fronts = fast_nondominated_sort(pop)
            for f in fronts:
                crowding_distance(f)

            pareto = fronts[0]

            pareto_df = pd.DataFrame(
                [(p["obj"][0], p["obj"][1]) for p in pareto],
                columns=["Makespan", "Waiting Time"]
            )
            st.subheader("Pareto Front")
            st.scatter_chart(pareto_df)

            best = min(pareto, key=lambda x: x["obj"][0])

            st.subheader("Best Schedule")
            st.write("Sequence:", [f"J{i+1}" for i in best["seq"]])
            st.write("Makespan:", best["obj"][0])
            st.write("Waiting Time:", best["obj"][1])
            st.write("Idle Time:", f"{best['idle']:.2f}")

            plot_gantt(best["gantt"])

            st.subheader("Total Population Fitness")
            st.write(f"{compute_total_fitness(pop):.2f}")
