import streamlit as st
import pandas as pd
import numpy as np
import simpy
import plotly.express as px
import graphviz
import string

# Set the page layout to wide
st.set_page_config(layout="wide")

# Generate names for parallel stations
def generate_parallel_names(base_name, count):
    return [f"{base_name} {letter}" for letter in string.ascii_lowercase[:count]]

# Run the simulation
def run_simulation(env, stations_config, results):
    queues = {name: [] for name in results['station_names']}

    def process_product(i):
        for group in stations_config:
            # Pick one resource (for parallel stations)
            idx = i % len(group['resources'])
            with group['resources'][idx].request() as req:
                arrival = env.now
                yield req
                wait = env.now - arrival
                service = group['cycle_time']
                station_name = group['names'][idx]
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
        yield env.timeout(0)  # Non-stop arrival

    results['throughput'] = i / results['duration']
    results['queues'] = queues

# Graphviz layout
def generate_layout_graph(stations_config):
    dot = graphviz.Digraph()
    prev_layer = []

    for group in stations_config:
        current_layer = group['names']
        for node in current_layer:
            dot.node(node)
        for prev in prev_layer:
            for curr in current_layer:
                dot.edge(prev, curr)
        prev_layer = current_layer

    return dot

# Main app
def main():
    st.title("🛠️ Production Line Simulator")

    if 'stations' not in st.session_state:
        st.session_state.stations = []

    st.subheader("📦 Add Stations Step by Step")
    if st.button("➕ Add Station"):
        st.session_state.stations.append({"name": f"Station {len(st.session_state.stations)+1}", "cycle_time": 1.0, "parallel": 1})

    for i, station in enumerate(st.session_state.stations):
        with st.expander(f"Configure {station['name']}", expanded=True):
            station['name'] = st.text_input("Station Name", station['name'], key=f"name_{i}")
            station['cycle_time'] = st.number_input("Cycle Time (min)", min_value=0.1, value=station['cycle_time'], key=f"ct_{i}")
            station['parallel'] = st.number_input("Parallel Units", min_value=1, value=station['parallel'], key=f"p_{i}")

    duration = st.number_input("Simulation Duration (minutes)", min_value=1, value=60)

    if st.button("▶️ Run Simulation") and st.session_state.stations:
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

        # Gantt Chart
        st.subheader("📊 Gantt Chart")
        df_gantt = pd.DataFrame(results['gantt'])
        if not df_gantt.empty:
            fig = px.timeline(df_gantt, x_start='Start', x_end='Finish', y='Task', color='Product')
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        # Utilization
        st.subheader("⚙️ Utilization per Station")
        utilization = []
        for group in stations_config:
            for name in group['names']:
                total_work = sum(
                    row['Finish'] - row['Start']
                    for row in results['gantt'] if row['Task'] == name
                )
                util = (total_work / duration) * 100
                utilization.append({'Station': name, 'Utilization (%)': util})
        df_util = pd.DataFrame(utilization)
        st.bar_chart(df_util.set_index('Station'))

        # Waiting Time
        st.subheader("⏱️ Average Waiting Time per Station")
        waits = []
        for name, events in results['queues'].items():
            avg_wait = np.mean([e[1] for e in events]) if events else 0
            waits.append({'Station': name, 'Avg Wait Time (min)': avg_wait})
        df_wait = pd.DataFrame(waits)
        st.bar_chart(df_wait.set_index('Station'))

        # Queue Histogram
        st.subheader("📈 Queue Events Over Time")
        fig2, ax2 = plt.subplots()
        for name, entries in results['queues'].items():
            times = [e[0] for e in entries]
            ax2.hist(times, bins=30, alpha=0.5, label=name)
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Queue Events")
        ax2.legend()
        st.pyplot(fig2)

        # Throughput
        st.subheader(f"🚀 Throughput: {results['throughput']:.2f} units/min")

        # Layout
        st.subheader("🗺️ Line Layout")
        graph = generate_layout_graph(stations_config)
        st.graphviz_chart(graph)

if __name__ == "__main__":
    main()
