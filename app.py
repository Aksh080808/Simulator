import streamlit as st
import pandas as pd
import numpy as np
import simpy
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from PIL import Image
import string

st.set_page_config(layout="wide")

# Helper to generate station names with a, b, c... suffixes
def generate_parallel_names(base_name, count):
    return [f"{base_name} {letter}" for letter in string.ascii_lowercase[:count]]

# Simulation function
def run_simulation(env, stations_config, results):
    queues = {name: [] for name in results['station_names']}

    def process_product(i):
        for group in stations_config:
            with group['resources'][i % len(group['resources'])].request() as req:
                arrival = env.now
                yield req
                wait = env.now - arrival
                service = group['cycle_time']
                station_name = group['names'][i % len(group['resources'])]
                queues[station_name].append((env.now, wait))
                yield env.timeout(service)
                results['gantt'].append({
                    'Task': station_name,
                    'Start': arrival,
                    'Finish': env.now,
                    'Product': i
                })

    i = 0
    while env.now < results['duration']:
        env.process(process_product(i))
        i += 1
        yield env.timeout(0)  # nonstop arrival

    results['throughput'] = i / results['duration']
    results['queues'] = queues

# Fixed Layout Diagram
def draw_layout(stations_config):
    G = nx.DiGraph()
    for layer_idx, group in enumerate(stations_config):
        current = group['names']
        for node in current:
            G.add_node(node, subset=layer_idx)
        if layer_idx > 0:
            for src in stations_config[layer_idx - 1]['names']:
                for tgt in current:
                    G.add_edge(src, tgt)

    pos = nx.multipartite_layout(G, subset_key="subset")
    fig, ax = plt.subplots(figsize=(12, 4))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightgreen", arrows=True, ax=ax)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# Main UI
def main():
    st.title("Production Line Simulation with Parallel Stations and Layout")

    if 'stations' not in st.session_state:
        st.session_state.stations = []

    st.subheader("Build Your Production Line")
    if st.button("Add New Station"):
        st.session_state.stations.append({"name": f"Station {len(st.session_state.stations) + 1}", "cycle_time": 1.0, "parallel": 1})

    for i, station in enumerate(st.session_state.stations):
        with st.expander(f"Configure {station['name']}"):
            station['name'] = st.text_input("Station Name", value=station['name'], key=f"name_{i}")
            station['cycle_time'] = st.number_input("Cycle Time (min)", min_value=0.1, value=station['cycle_time'], key=f"ct_{i}")
            station['parallel'] = st.number_input("Parallel Stations", min_value=1, value=station['parallel'], key=f"p_{i}")

    duration = st.number_input("Simulation Duration (minutes)", min_value=1, value=60)

    if st.button("Run Simulation") and st.session_state.stations:
        env = simpy.Environment()
        results = {'gantt': [], 'duration': duration}

        stations_config = []
        station_names = []
        for s in st.session_state.stations:
            names = generate_parallel_names(s['name'], s['parallel'])
            resources = [simpy.Resource(env, capacity=1) for _ in range(s['parallel'])]
            stations_config.append({
                'names': names,
                'resources': resources,
                'cycle_time': s['cycle_time']
            })
            station_names.extend(names)

        results['station_names'] = station_names

        env.process(run_simulation(env, stations_config, results))
        env.run()

        # Gantt chart
        df_gantt = pd.DataFrame(results['gantt'])
        if not df_gantt.empty:
            fig = px.timeline(df_gantt, x_start='Start', x_end='Finish', y='Task', color='Product', title="Gantt Chart")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        # Utilization
        utilization_data = []
        for group in stations_config:
            for name in group['names']:
                total_work = sum([row['Finish'] - row['Start'] for row in results['gantt'] if row['Task'] == name])
                util = (total_work / duration) * 100
                utilization_data.append({'Station': name, 'Utilization (%)': util})
        df_util = pd.DataFrame(utilization_data)
        st.subheader("Utilization per Station")
        st.bar_chart(df_util.set_index('Station'))

        # Queue Lengths and Waiting Time
        st.subheader("Average Waiting Time per Station")
        waits = []
        for name, entries in results['queues'].items():
            avg_wait = np.mean([e[1] for e in entries]) if entries else 0
            waits.append({'Station': name, 'Avg Wait Time (min)': avg_wait})
        df_wait = pd.DataFrame(waits)
        st.bar_chart(df_wait.set_index('Station'))

        st.subheader("Histogram of Queue Events")
        fig2, ax2 = plt.subplots()
        for name, entries in results['queues'].items():
            times = [e[0] for e in entries]
            ax2.hist(times, bins=30, alpha=0.5, label=name)
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Queue Events")
        ax2.legend()
        st.pyplot(fig2)

        # Throughput
        st.subheader(f"Total Line Throughput: {results['throughput']:.2f} units/min")

        # Layout
        st.subheader("Production Line Layout")
        layout_img = draw_layout(stations_config)
        st.image(Image.open(layout_img))

if __name__ == "__main__":
    main()
