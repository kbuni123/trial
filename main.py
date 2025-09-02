"""
Ethiopia Energy Planning Tool
A comprehensive PyPSA-based energy system model for Ethiopia

Main application file - run with: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pypsa
import warnings
warnings.filterwarnings('ignore')

# Custom modules (would be in separate files)
from data_processor import DREDataProcessor, TransmissionProcessor, GeneratorProcessor
from pypsa_builder import EthiopiaPyPSABuilder
from scenario_engine import ScenarioEngine
from visualization import NetworkVisualizer, ResultsVisualizer

# Page configuration
st.set_page_config(
    page_title="Ethiopia Energy Planning Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
    }
</style>
""", unsafe_allow_html=True)

class EthiopiaEnergyApp:
    def __init__(self):
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'network_built' not in st.session_state:
            st.session_state.network_built = False
        if 'pypsa_network' not in st.session_state:
            st.session_state.pypsa_network = None
            
    def run(self):
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>⚡ Ethiopia Energy Planning Tool</h1>
            <p>Comprehensive power system modeling and expansion planning using PyPSA and World Bank DRE Atlas data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.get('page', 'Data Import')
        
        if page == 'Data Import':
            self.render_data_import()
        elif page == 'Network Overview':
            self.render_network_overview()
        elif page == 'Scenario Planning':
            self.render_scenario_planning()
        elif page == 'Results & Visualization':
            self.render_results()
        elif page == 'Settlement Analysis':
            self.render_settlement_analysis()
        elif page == 'Transmission Planning':
            self.render_transmission_planning()
    
    def render_sidebar(self):
        st.sidebar.title("Navigation")
        
        pages = [
            'Data Import',
            'Network Overview', 
            'Settlement Analysis',
            'Scenario Planning',
            'Transmission Planning',
            'Results & Visualization'
        ]
        
        st.session_state.page = st.sidebar.radio("Select Page", pages)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Status")
        
        # Status indicators
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data Loaded")
        else:
            st.sidebar.error("❌ Data Not Loaded")
            
        if st.session_state.network_built:
            st.sidebar.success("✅ Network Built")
        else:
            st.sidebar.warning("⚠️ Network Not Built")
    
    def render_data_import(self):
        st.header("📊 Data Import & Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("DRE Atlas Data")
            dre_file = st.file_uploader(
                "Upload DRE Atlas CSV file", 
                type=['csv'],
                help="World Bank DRE Atlas settlement data"
            )
            
            st.subheader("Transmission Network")
            transmission_file = st.file_uploader(
                "Upload transmission shapefile", 
                type=['shp', 'zip'],
                help="Transmission lines shapefile (>45kV)"
            )
        
        with col2:
            st.subheader("Generator Data")
            generator_file = st.file_uploader(
                "Upload generator data", 
                type=['csv', 'xlsx'],
                help="Power plant locations and capacities"
            )
            
            st.subheader("Additional Shapefiles")
            boundary_file = st.file_uploader(
                "Upload administrative boundaries (optional)", 
                type=['shp', 'zip'],
                help="Regional/district boundaries for visualization"
            )
        
        if st.button("Process Data", type="primary"):
            if dre_file is not None:
                with st.spinner("Processing data..."):
                    try:
                        # Process DRE data
                        processor = DREDataProcessor()
                        dre_data = processor.load_and_process(dre_file)
                        st.session_state.dre_data = dre_data
                        
                        # Process transmission data if available
                        if transmission_file:
                            trans_processor = TransmissionProcessor()
                            trans_data = trans_processor.load_shapefile(transmission_file)
                            st.session_state.transmission_data = trans_data
                        
                        # Process generator data if available
                        if generator_file:
                            gen_processor = GeneratorProcessor()
                            gen_data = gen_processor.load_generators(generator_file)
                            st.session_state.generator_data = gen_data
                        
                        st.session_state.data_loaded = True
                        st.success("Data processed successfully!")
                        
                        # Display summary statistics
                        self.display_data_summary()
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
            else:
                st.warning("Please upload at least the DRE Atlas data file.")
        
        if st.session_state.data_loaded:
            self.display_data_summary()
    
    def display_data_summary(self):
        st.subheader("📈 Data Summary")
        
        if 'dre_data' in st.session_state:
            dre_data = st.session_state.dre_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Settlements",
                    value=f"{len(dre_data):,}"
                )
            
            with col2:
                total_pop = dre_data['Population'].sum() if 'Population' in dre_data.columns else 0
                st.metric(
                    label="Total Population",
                    value=f"{total_pop:,.0f}"
                )
            
            with col3:
                total_demand = dre_data['Energy demand'].sum() if 'Energy demand' in dre_data.columns else 0
                st.metric(
                    label="Total Energy Demand",
                    value=f"{total_demand:,.0f} kWh/day"
                )
            
            with col4:
                total_buildings = dre_data['Number of buildings'].sum() if 'Number of buildings' in dre_data.columns else 0
                st.metric(
                    label="Total Buildings",
                    value=f"{total_buildings:,.0f}"
                )
            
            # Regional breakdown
            if 'Region or other (country-specific)' in dre_data.columns:
                st.subheader("Regional Distribution")
                regional_summary = dre_data.groupby('Region or other (country-specific)').agg({
                    'Population': 'sum',
                    'Energy demand': 'sum',
                    'Number of buildings': 'sum'
                }).round(0)
                
                st.dataframe(regional_summary, use_container_width=True)
    
    def render_network_overview(self):
        st.header("🔗 Network Overview")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first in the Data Import section.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Settlement Distribution")
            if 'dre_data' in st.session_state:
                self.plot_settlement_map()
        
        with col2:
            st.subheader("Network Configuration")
            
            # PyPSA network building options
            st.markdown("#### Network Parameters")
            
            voltage_levels = st.multiselect(
                "Voltage Levels to Model",
                [400, 230, 132, 66, 45],
                default=[400, 230, 132],
                help="Select transmission voltage levels"
            )
            
            clustering_method = st.selectbox(
                "Settlement Clustering Method",
                ["Administrative", "K-means", "Hierarchical", "No clustering"],
                help="Method to aggregate settlements into network nodes"
            )
            
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=10,
                max_value=200,
                value=50,
                help="Number of network nodes (if clustering enabled)"
            )
            
            if st.button("Build PyPSA Network", type="primary"):
                with st.spinner("Building network..."):
                    try:
                        builder = EthiopiaPyPSABuilder()
                        network = builder.build_network(
                            st.session_state.dre_data,
                            st.session_state.get('transmission_data'),
                            st.session_state.get('generator_data'),
                            voltage_levels=voltage_levels,
                            clustering_method=clustering_method,
                            n_clusters=n_clusters
                        )
                        
                        st.session_state.pypsa_network = network
                        st.session_state.network_built = True
                        st.success("PyPSA network built successfully!")
                        
                        # Display network statistics
                        self.display_network_stats(network)
                        
                    except Exception as e:
                        st.error(f"Error building network: {str(e)}")
    
    def plot_settlement_map(self):
        """Create an interactive map of settlements"""
        dre_data = st.session_state.dre_data
        
        # Sample data for visualization if too large
        if len(dre_data) > 5000:
            plot_data = dre_data.sample(5000)
        else:
            plot_data = dre_data
        
        fig = px.scatter_mapbox(
            plot_data,
            lat='Latitude' if 'Latitude' in plot_data.columns else 'lat',
            lon='Longitude' if 'Longitude' in plot_data.columns else 'lon',
            size='Energy demand' if 'Energy demand' in plot_data.columns else None,
            color='Region or other (country-specific)' if 'Region or other (country-specific)' in plot_data.columns else None,
            hover_name='Name' if 'Name' in plot_data.columns else None,
            hover_data=['Population', 'Energy demand'] if all(col in plot_data.columns for col in ['Population', 'Energy demand']) else None,
            mapbox_style="open-street-map",
            height=600,
            title="Settlement Distribution in Ethiopia"
        )
        
        fig.update_layout(
            mapbox=dict(
                center=dict(lat=9.1450, lon=40.4897),  # Ethiopia center
                zoom=5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_network_stats(self, network):
        """Display PyPSA network statistics"""
        st.subheader("Network Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Buses", len(network.buses))
            st.metric("Lines", len(network.lines))
        
        with col2:
            st.metric("Generators", len(network.generators))
            st.metric("Loads", len(network.loads))
        
        with col3:
            total_gen_capacity = network.generators.p_nom.sum()
            total_load = network.loads_t.p_set.sum().sum() if not network.loads_t.p_set.empty else 0
            st.metric("Generation Capacity", f"{total_gen_capacity:.0f} MW")
            st.metric("Total Load", f"{total_load:.0f} MW")
    
    def render_settlement_analysis(self):
        st.header("🏘️ Settlement Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first.")
            return
        
        dre_data = st.session_state.dre_data
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            regions = dre_data['Region or other (country-specific)'].unique() if 'Region or other (country-specific)' in dre_data.columns else []
            selected_regions = st.multiselect("Select Regions", regions, default=regions[:5] if len(regions) > 5 else regions)
        
        with col2:
            pop_range = st.slider(
                "Population Range",
                min_value=int(dre_data['Population'].min()) if 'Population' in dre_data.columns else 0,
                max_value=int(dre_data['Population'].max()) if 'Population' in dre_data.columns else 1000,
                value=(0, 10000)
            )
        
        with col3:
            grid_connected = st.selectbox(
                "Grid Connection Status",
                ["All", "Shows light at night", "No nightlight"],
                index=0
            )
        
        # Filter data
        filtered_data = dre_data.copy()
        if selected_regions and 'Region or other (country-specific)' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Region or other (country-specific)'].isin(selected_regions)]
        
        if 'Population' in filtered_data.columns:
            filtered_data = filtered_data[
                (filtered_data['Population'] >= pop_range[0]) & 
                (filtered_data['Population'] <= pop_range[1])
            ]
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Overview", "Energy Demand", "Infrastructure"])
        
        with tab1:
            self.render_settlement_overview(filtered_data)
        
        with tab2:
            self.render_energy_analysis(filtered_data)
        
        with tab3:
            self.render_infrastructure_analysis(filtered_data)
    
    def render_settlement_overview(self, data):
        """Render settlement overview analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Region or other (country-specific)' in data.columns:
                fig = px.histogram(
                    data, 
                    x='Region or other (country-specific)',
                    title="Settlements by Region"
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Population' in data.columns:
                fig = px.histogram(
                    data,
                    x='Population',
                    nbins=50,
                    title="Population Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("Settlement Summary")
        if 'Region or other (country-specific)' in data.columns:
            summary = data.groupby('Region or other (country-specific)').agg({
                'Population': ['count', 'sum', 'mean'],
                'Energy demand': ['sum', 'mean'] if 'Energy demand' in data.columns else ['count'],
                'Number of buildings': ['sum', 'mean'] if 'Number of buildings' in data.columns else ['count']
            }).round(2)
            
            st.dataframe(summary, use_container_width=True)
    
    def render_energy_analysis(self, data):
        """Render energy demand analysis"""
        if 'Energy demand' not in data.columns:
            st.warning("Energy demand data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Energy demand distribution
            fig = px.histogram(
                data,
                x='Energy demand',
                nbins=50,
                title="Energy Demand Distribution (kWh/day)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Energy demand vs population
            if 'Population' in data.columns:
                fig = px.scatter(
                    data.sample(min(1000, len(data))),  # Sample for performance
                    x='Population',
                    y='Energy demand',
                    title="Energy Demand vs Population",
                    color='Region or other (country-specific)' if 'Region or other (country-specific)' in data.columns else None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Energy access analysis
        if 'Shows light at night?' in data.columns:
            st.subheader("Energy Access Analysis")
            
            access_summary = data.groupby('Shows light at night?').agg({
                'Population': 'sum',
                'Energy demand': 'sum'
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=access_summary['Population'],
                    names=access_summary.index,
                    title="Population by Nightlight Status"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=access_summary['Energy demand'],
                    names=access_summary.index,
                    title="Energy Demand by Nightlight Status"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_infrastructure_analysis(self, data):
        """Render infrastructure analysis"""
        st.subheader("Grid Proximity Analysis")
        
        # Distance to grid analysis
        grid_cols = [col for col in data.columns if 'Distance to' in col and 'grid' in col]
        
        if grid_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_distance = st.selectbox("Select Distance Metric", grid_cols)
                
                fig = px.histogram(
                    data,
                    x=selected_distance,
                    nbins=50,
                    title=f"Distribution of {selected_distance}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Grid connection feasibility analysis
                if selected_distance in data.columns:
                    data['Grid_Feasibility'] = pd.cut(
                        data[selected_distance],
                        bins=[0, 5, 15, 50, float('inf')],
                        labels=['Very Close (<5km)', 'Close (5-15km)', 'Moderate (15-50km)', 'Far (>50km)']
                    )
                    
                    feasibility_summary = data.groupby('Grid_Feasibility').agg({
                        'Population': 'sum',
                        'Energy demand': 'sum'
                    })
                    
                    fig = px.bar(
                        feasibility_summary,
                        y='Population',
                        title="Population by Grid Proximity"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Solar potential analysis
        if 'Potential PV production' in data.columns:
            st.subheader("Solar Potential Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    data,
                    x='Potential PV production',
                    nbins=50,
                    title="Solar Potential Distribution (kWh/kWp)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Solar vs demand analysis
                if 'Energy demand' in data.columns:
                    data['Solar_Adequacy'] = data['Potential PV production'] / (data['Energy demand'] / 365 * 1000)  # Rough conversion
                    
                    fig = px.histogram(
                        data[data['Solar_Adequacy'] < 10],  # Filter extreme values
                        x='Solar_Adequacy',
                        nbins=50,
                        title="Solar Adequacy Ratio"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_scenario_planning(self):
        st.header("📊 Scenario Planning")
        
        if not st.session_state.network_built:
            st.warning("Please build the PyPSA network first.")
            return
        
        # Scenario configuration
        st.subheader("Scenario Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Demand Growth")
            demand_growth = st.slider("Annual demand growth (%)", 0.0, 10.0, 5.0, 0.1)
            
            st.markdown("#### Renewable Targets")
            renewable_target_2030 = st.slider("Renewable share by 2030 (%)", 0, 100, 50)
            renewable_target_2050 = st.slider("Renewable share by 2050 (%)", 0, 100, 80)
        
        with col2:
            st.markdown("#### Technology Costs")
            solar_cost_decline = st.slider("Solar cost decline (% per year)", 0.0, 10.0, 3.0, 0.1)
            wind_cost_decline = st.slider("Wind cost decline (% per year)", 0.0, 10.0, 2.0, 0.1)
            battery_cost_decline = st.slider("Battery cost decline (% per year)", 0.0, 15.0, 8.0, 0.1)
        
        # Scenario execution
        if st.button("Run Scenario Analysis", type="primary"):
            with st.spinner("Running scenario analysis..."):
                try:
                    scenario_engine = ScenarioEngine()
                    results = scenario_engine.run_scenarios(
                        st.session_state.pypsa_network,
                        demand_growth=demand_growth,
                        renewable_targets={2030: renewable_target_2030, 2050: renewable_target_2050},
                        cost_declines={
                            'solar': solar_cost_decline,
                            'wind': wind_cost_decline,
                            'battery': battery_cost_decline
                        }
                    )
                    
                    st.session_state.scenario_results = results
                    st.success("Scenario analysis completed!")
                    
                    # Display results preview
                    self.display_scenario_results(results)
                    
                except Exception as e:
                    st.error(f"Error running scenarios: {str(e)}")
    
    def display_scenario_results(self, results):
        """Display scenario analysis results"""
        st.subheader("Scenario Results Preview")
        
        # Create sample results visualization
        years = [2025, 2030, 2040, 2050]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generation capacity by technology
            fig = go.Figure()
            
            technologies = ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal']
            colors = ['#FDB462', '#80B1D3', '#B3DE69', '#FCCDE5', '#D9D9D9']
            
            for i, tech in enumerate(technologies):
                capacities = np.random.rand(len(years)) * 1000 + i * 200  # Sample data
                fig.add_trace(go.Scatter(
                    x=years,
                    y=capacities,
                    mode='lines+markers',
                    name=tech,
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Generation Capacity by Technology",
                xaxis_title="Year",
                yaxis_title="Capacity (MW)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Investment requirements
            investments = np.random.rand(len(years)) * 5 + 2  # Sample data
            
            fig = px.bar(
                x=years,
                y=investments,
                title="Annual Investment Requirements",
                labels={'x': 'Year', 'y': 'Investment (Billion USD)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_transmission_planning(self):
        st.header("🔌 Transmission Planning")
        
        if not st.session_state.network_built:
            st.warning("Please build the PyPSA network first.")
            return
        
        st.subheader("Transmission Expansion Analysis")
        
        # Planning parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Planning Horizon")
            planning_years = st.multiselect(
                "Select years for analysis",
                [2025, 2030, 2035, 2040, 2045, 2050],
                default=[2030, 2040, 2050]
            )
            
            max_line_capacity = st.slider("Max line capacity (MW)", 100, 2000, 500)
        
        with col2:
            st.markdown("#### Cost Parameters")
            line_cost_per_km = st.number_input("Line cost ($/km)", value=100000)
            substation_cost = st.number_input("Substation cost ($)", value=5000000)
        
        # Network analysis
        if st.button("Analyze Transmission Needs"):
            st.info("This would perform detailed transmission expansion analysis...")
            
            # Placeholder for transmission analysis results
            st.subheader("Recommended Transmission Projects")
            
            # Sample data for demonstration
            projects_data = {
                'Project': ['Line A-B', 'Line C-D', 'Substation E', 'Line F-G'],
                'Type': ['Transmission', 'Transmission', 'Substation', 'Transmission'],
                'Capacity (MW)': [400, 230, 500, 132],
                'Length (km)': [150, 89, 0, 205],
                'Cost (Million $)': [15, 9, 5, 20],
                'Priority': ['High', 'Medium', 'High', 'Low'],
                'Target Year': [2030, 2035, 2030, 2040]
            }
            
            projects_df = pd.DataFrame(projects_data)
            st.dataframe(projects_df, use_container_width=True)
    
    def render_results(self):
        st.header("📈 Results & Visualization")
        
        if not st.session_state.get('scenario_results'):
            st.warning("Please run scenario analysis first.")
            return
        
        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Generation Mix", "Investment Timeline", "Network Map", "Regional Analysis", "Cost Analysis"]
        )
        
        if viz_type == "Generation Mix":
            self.render_generation_mix_viz()
        elif viz_type == "Investment Timeline":
            self.render_investment_timeline()
        elif viz_type == "Network Map":
            self.render_network_map()
        elif viz_type == "Regional Analysis":
            self.render_regional_analysis()
        elif viz_type == "Cost Analysis":
            self.render_cost_analysis()
    
    def render_generation_mix_viz(self):
        """Render generation mix visualization"""
        st.subheader("Generation Mix Evolution")
        
        # Sample data for demonstration
        years = [2025, 2030, 2040, 2050]
        technologies = ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal']
        
        # Create stacked area chart
        fig = go.Figure()
        
        colors = ['#FDB462', '#80B1D3', '#B3DE69', '#FCCDE5', '#D9D9D9']
        
        for i, tech in enumerate(technologies):
            values = np.random.rand(len(years)) * 100
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                stackgroup='one',
                name=tech,
                fillcolor=colors[i],
                line=dict(color=colors[i])
            ))
        
        fig.update_layout(
            title="Generation Capacity Mix Over Time",
            xaxis_title="Year",
            yaxis_title="Share (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_investment_timeline(self):
        """Render investment timeline"""
        st.subheader("Investment Timeline")
        
        # Sample investment data
        years = list(range(2025, 2051))
        generation_inv = np.random.rand(len(years)) * 2 + 1
        transmission_inv = np.random.rand(len(years)) * 1 + 0.5
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years,
            y=generation_inv,
            name='Generation',
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            x=years,
            y=transmission_inv,
            name='Transmission',
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title="Annual Investment Requirements",
            xaxis_title="Year",
            yaxis_title="Investment (Billion USD)",
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_network_map(self):
        """Render network map visualization"""
        st.subheader("Network Map")
        
        if st.session_state.pypsa_network:
            network = st.session_state.pypsa_network
            
            # Create network map using buses
            if not network.buses.empty:
                fig = px.scatter_mapbox(
                    network.buses.reset_index(),
                    lat='y',
                    lon='x',
                    hover_name='Bus',
                    size_max=15,
                    zoom=5,
                    mapbox_style="open-street-map",
                    title="PyPSA Network Buses"
                )
                
                fig.update_layout(
                    mapbox=dict(
                        center=dict(lat=9.1450, lon=40.4897),  # Ethiopia center
                        zoom=5
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No network buses to display")
        else:
            st.warning("No PyPSA network available")
    
    def render_regional_analysis(self):
        """Render regional analysis"""
        st.subheader("Regional Energy Analysis")
        
        if 'dre_data' in st.session_state and 'Region or other (country-specific)' in st.session_state.dre_data.columns:
            data = st.session_state.dre_data
            
            regional_summary = data.groupby('Region or other (country-specific)').agg({
                'Population': 'sum',
                'Energy demand': 'sum',
                'Potential PV production': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    regional_summary,
                    x='Region or other (country-specific)',
                    y='Energy demand',
                    title="Energy Demand by Region"
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    regional_summary,
                    x='Region or other (country-specific)',
                    y='Potential PV production',
                    title="Solar Potential by Region"
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_cost_analysis(self):
        """Render cost analysis"""
        st.subheader("Cost Analysis")
        
        # Sample cost data
        technologies = ['Solar PV', 'Wind', 'Hydro', 'Gas', 'Transmission']
        costs_2025 = [800, 1200, 2000, 600, 1000000]  # $/kW or $/km for transmission
        costs_2050 = [400, 800, 2000, 700, 900000]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='2025',
            x=technologies,
            y=costs_2025,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='2050',
            x=technologies,
            y=costs_2050,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Technology Cost Comparison: 2025 vs 2050',
            xaxis_title='Technology',
            yaxis_title='Cost ($/kW)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main execution
if __name__ == "__main__":
    app = EthiopiaEnergyApp()
    app.run()