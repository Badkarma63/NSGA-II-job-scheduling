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

    for job in sequence:
        for m in range(n_machines):
            start = max(machine_time[m], job_time[job])
            waiting += start - job_time[job]
            finish = start + pt[job, m]
            machine_time[m] = finish
            job_time[job] = finish

            gantt.append((f"M{m+1}", start, finish, f"J{job+1}"))

    return max(job_time), waiting, gantt


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
    child = p1[:cut] + [j for j in p2 if j not in p1[:cut]]
    return child


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
        m, w, g = evaluate_schedule(seq, pt)
        population.append({"seq": seq, "obj": (m, w), "gantt": g})

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

            m, w, g = evaluate_schedule(child_seq, pt)
            offspring.append({"seq": child_seq, "obj": (m, w), "gantt": g})

        population = offspring

    return population


# ----------------------------------
# Gantt Chart (Professional)
# ----------------------------------
import plotly.express as px

def plot_gantt(gantt):
    df = pd.DataFrame(gantt, columns=["Machine", "Start", "End", "Job"])

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Machine",
        color="Job",
        title="NSGA-II Optimized Job Schedule",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        height=600,                     # BIG boxes
        xaxis_title="Time",
        yaxis_title="Machine",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        legend_title_text="Jobs",
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white")),
        opacity=0.95
    )

    st.plotly_chart(fig, use_container_width=True)
# ----------------------------------
# STREAMLIT UI
# ----------------------------------
st.title("NSGA-II Job Scheduling")

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

        st.subheader("Best Schedule (Minimum Makespan)")
        st.write("Sequence:", [f"J{i+1}" for i in best["seq"]])
        st.write("Makespan:", best["obj"][0])
        st.write("Waiting Time:", best["obj"][1])

        plot_gantt(best["gantt"])
