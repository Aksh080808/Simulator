import streamlit as st
import pandas as pd
import plotly.express as px
import simpy
 
# Station logic
class Station:
    def __init__(self, env, name, cycle_time, parallel):
        self.env = env
        self.name = name
        self.cycle_time = cycle_time
        self.resource = simpy.Resource(env, capacity=parallel)
        self.completed = 0
        self.wait_times = []
        self.queue_lengths = []
        self.timeline = []
 
    def process(self, product_id):
        arrival = self.env.now
        with self.resource.request() as req:
            yield req
            wait = self.env.now - arrival
            self.wait_times.append(wait)
            self.queue_lengths.append((self.env.now, len(self.resource.queue)))
            self.timeline.append((product_id, self.env.now, self.env.now + self.cycle_time))
            yield self.env.timeout(self.cycle_time)
            self.completed += 1
 
# Product passing through all stations
def product(env, name, stations, final_store):
    for station in stations:
        yield env.process(station.process(name))
    final_store.put(name)
 
# Run simulation
def run_simulation(station_configs, sim_duration):
    env = simpy.Environment()
    stations = [Station(env, s["name"], s["cycle_time"], s["parallel"]) for s in station_configs]
    final_store = simpy.Store(env)
 
    def source(env):
        i = 0
        while True:
            env.process(product(env, f"P{i}", stations, final_store))
            i += 1
            yield env.timeout(0.5)  # inter-arrival time
 
    env.process(source(env))
    env.run(until=sim_duration)
 
    return {
        "stations": stations,
        "completed": len(final_store.items),
        "final_store": final_store.items
    }
 
# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.title("🔧 Production Line Simulation (with Parallel Stations & DES)")
 
    if "stations" not in st.session_state:
        st.session_state.stations = []
 
    with st.expander("🛠️ Station Configuration", expanded=True):
        for i, s in enumerate(st.session_state.stations):
            cols = st.columns([3, 2, 2, 1])
            s["name"] = cols[0].text_input("Station Name", value=s["name"], key=f"name_{i}")
            s["cycle_time"] = cols[1].number_input("Cycle Time (mins)", min_value=0.1, value=s["cycle_time"], key=f"ct_{i}")
            s["parallel"] = cols[2].number_input("Parallels", min_value=1, value=s["parallel"], key=f"p_{i}")
            if cols[3].button("❌", key=f"del_{i}"):
                st.session_state.stations.pop(i)
                st.experimental_rerun()
 
        st.markdown("### ➕ Add Station")
        new_name = st.text_input("Station Display Name", value=f"Station {len(st.session_state.stations)+1}")
        new_ct = st.number_input("Cycle Time (mins)", min_value=0.1, value=1.0, key="new_ct")
        new_p = st.number_input("Parallel Units", min_value=1, value=1, key="new_p")
        if st.button("Add Station"):
            st.session_state.stations.append({"name": new_name, "cycle_time": new_ct, "parallel": new_p})
            st.experimental_rerun()
 
    sim_duration = st.number_input("⏱️ Simulation Duration (minutes)", min_value=1, value=100)
    if st.button("🚀 Run Simulation") and st.session_state.stations:
        with st.spinner("Simulating..."):
            result = run_simulation(st.session_state.stations, sim_duration)
            stations = result["stations"]
            completed = result["completed"]
 
            if completed == 0:
                st.error("❗ No throughput data found. Try increasing the simulation duration or check your station setup.")
                return
 
            st.success(f"✅ {completed} units completed in {sim_duration} minutes")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 Total Throughput", f"{completed / sim_duration:.2f} units/min")
            with col2:
                takt_time = sim_duration / completed
                st.metric("⏱️ Takt Time", f"{takt_time:.2f} min/unit")
 
            utilizations = [min(100.0, (s.cycle_time * s.completed) / (sim_duration * s.resource.capacity) * 100) for s in stations]
            util_df = pd.DataFrame({
                "Station": [s.name for s in stations],
                "Utilization (%)": utilizations
            })
            st.subheader("📊 Utilization per Station")
            st.bar_chart(util_df.set_index("Station"))
 
            wait_df = pd.DataFrame({
                "Station": [s.name for s in stations],
                "Avg Wait Time (min)": [sum(s.wait_times)/len(s.wait_times) if s.wait_times else 0 for s in stations]
            })
            st.subheader("⏱️ Average Waiting Time per Station")
            st.bar_chart(wait_df.set_index("Station"))
 
            st.subheader("📈 Queue Length Over Time")
            queue_data = []
            for s in stations:
                for t, q in s.queue_lengths:
                    queue_data.append({"Station": s.name, "Time": t, "Queue": q})
            qdf = pd.DataFrame(queue_data)
            if not qdf.empty:
                fig = px.line(qdf, x="Time", y="Queue", color="Station")
                st.plotly_chart(fig)
 
            st.subheader("🗂️ Gantt Chart")
            gantt_data = []
            for s in stations:
                for pid, start, end in s.timeline:
                    gantt_data.append({"Task": f"{s.name}", "Start": start, "Finish": end, "Product": pid})
            gdf = pd.DataFrame(gantt_data)
            fig = px.timeline(gdf, x_start="Start", x_end="Finish", y="Task", color="Product")
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig)
 
            st.subheader("🛠️ Bottleneck Analysis")
            bottleneck_idx = utilizations.index(max(utilizations))
            bottleneck = stations[bottleneck_idx]
            st.write(f"🔻 **Bottleneck Station:** `{bottleneck.name}` with {utilizations[bottleneck_idx]:.2f}% utilization.")
            st.info(f"Consider adding 1 more parallel unit to **{bottleneck.name}** to reduce load.")
 
            st.subheader("🧭 Production Line Layout")
            layout_str = ""
            for s in stations:
                if s.resource.capacity > 1:
                    layout_str += f"[{s.name}] x{s.resource.capacity} → "
                else:
                    layout_str += f"[{s.name}] → "
            layout_str = layout_str.rstrip(" → ")
            st.markdown(f"`{layout_str}`")
 
if __name__ == "__main__":
    main()
