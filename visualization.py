"""
Visualization Module for Ethiopia Energy Planning Tool

Handles all visualization components including maps, charts, and network diagrams
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import folium
from folium import plugins
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class NetworkVisualizer:
    """Visualize PyPSA network topology and results"""
    
    def __init__(self):
        self.color_schemes = {
            'technology': {
                'Solar': '#FDB462',
                'Wind': '#80B1D3', 
                'Hydro': '#B3DE69',
                'Gas': '#FCCDE5',
                'Coal': '#D9D9D9',
                'Battery': '#BEBADA'
            },
            'voltage': {
                400: '#FF0000',
                230: '#FF8000',
                132: '#FFFF00',
                66: '#80FF00',
                45: '#00FF00'
            }
        }
    
    def create_network_map(self, network, show_generators=True, show_loads=True, show_lines=True):
        """Create interactive network map using Folium"""
        
        # Create base map centered on Ethiopia
        network_map = folium.Map(
            location=[9.1450, 40.4897],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add buses as markers
        if not network.buses.empty:
            for bus_id, bus in network.buses.iterrows():
                folium.CircleMarker(
                    location=[bus['y'], bus['x']],
                    radius=8,
                    popup=f"Bus: {bus_id}<br>Voltage: {bus.get('v_nom', 'N/A')} kV",
                    color='blue',
                    fill=True,
                    fillColor='lightblue'
                ).add_to(network_map)
        
        # Add generators
        if show_generators and not network.generators.empty:
            for gen_id, gen in network.generators.iterrows():
                bus_id = gen['bus']
                if bus_id in network.buses.index:
                    bus = network.buses.loc[bus_id]
                    carrier = gen.get('carrier', 'Other')
                    color = self.color_schemes['technology'].get(carrier, '#808080')
                    
                    folium.CircleMarker(
                        location=[bus['y'], bus['x']],
                        radius=max(5, min(15, gen.get('p_nom', 0) / 50)),
                        popup=f"Generator: {gen_id}<br>"
                              f"Technology: {carrier}<br>"
                              f"Capacity: {gen.get('p_nom', 0):.1f} MW",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(network_map)
        
        # Add transmission lines
        if show_lines and not network.lines.empty:
            for line_id, line in network.lines.iterrows():
                bus0 = network.buses.loc[line['bus0']]
                bus1 = network.buses.loc[line['bus1']]
                
                folium.PolyLine(
                    locations=[[bus0['y'], bus0['x']], [bus1['y'], bus1['x']]],
                    weight=3,
                    color='red',
                    opacity=0.7,
                    popup=f"Line: {line_id}<br>"
                          f"Capacity: {line.get('s_nom', 0):.0f} MVA<br>"
                          f"Length: {line.get('length', 0):.1f} km"
                ).add_to(network_map)
        
        # Add legend
        self._add_map_legend(network_map)
        
        return network_map
    
    def create_network_graph(self, network):
        """Create network graph using NetworkX and Plotly"""
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (buses)
        for bus_id, bus in network.buses.iterrows():
            G.add_node(bus_id, 
                      pos=(bus['x'], bus['y']),
                      voltage=bus.get('v_nom', 132))
        
        # Add edges (lines)
        for line_id, line in network.lines.iterrows():
            G.add_edge(line['bus0'], line['bus1'],
                      capacity=line.get('s_nom', 0),
                      length=line.get('length', 0))
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                              line=dict(width=2, color='#888'),
                              hoverinfo='none',
                              mode='lines')
        
        # Create node traces
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info for hover
            voltage = G.nodes[node].get('voltage', 132)
            node_info.append(f'Bus: {node}<br>Voltage: {voltage} kV')
            node_colors.append(voltage)
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                              mode='markers+text',
                              hoverinfo='text',
                              text=list(G.nodes()),
                              textposition="middle center",
                              hovertext=node_info,
                              marker=dict(
                                  size=15,
                                  color=node_colors,
                                  colorscale='Viridis',
                                  showscale=True,
                                  colorbar=dict(title="Voltage (kV)")
                              ))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Network Topology',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Network topology visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='#888', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def create_capacity_map(self, network, technology_filter=None):
        """Create map showing generation capacity by location"""
        
        if network.generators.empty:
            return None
        
        # Filter by technology if specified
        generators = network.generators.copy()
        if technology_filter:
            generators = generators[generators['carrier'] == technology_filter]
        
        # Aggregate capacity by bus
        capacity_by_bus = generators.groupby('bus')['p_nom'].sum()
        
        # Create data for plotting
        plot_data = []
        for bus_id, capacity in capacity_by_bus.items():
            if bus_id in network.buses.index:
                bus = network.buses.loc[bus_id]
                plot_data.append({
                    'bus_id': bus_id,
                    'lat': bus['y'],
                    'lon': bus['x'],
                    'capacity': capacity,
                    'technology': technology_filter or 'All'
                })
        
        if not plot_data:
            return None
        
        df = pd.DataFrame(plot_data)
        
        # Create map
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            size='capacity',
            hover_name='bus_id',
            hover_data=['capacity', 'technology'],
            size_max=30,
            zoom=6,
            center=dict(lat=9.1450, lon=40.4897),
            mapbox_style='open-street-map',
            title=f'Generation Capacity Map - {technology_filter or "All Technologies"}'
        )
        
        return fig
    
    def _add_map_legend(self, map_obj):
        """Add legend to Folium map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Network Legend</b></p>
        <p><i class="fa fa-circle" style="color:blue"></i> Buses</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Generators</p>
        <p><i class="fa fa-minus" style="color:red"></i> Transmission Lines</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

class ResultsVisualizer:
    """Visualize scenario results and analysis outputs"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
        
    def create_generation_mix_chart(self, results, scenario_name="Scenario"):
        """Create generation mix evolution chart"""
        
        if 'generation_mix' not in results:
            return None
        
        # Prepare data
        data = []
        for year, mix in results['generation_mix'].items():
            for tech, capacity in mix.items():
                data.append({
                    'Year': year,
                    'Technology': tech,
                    'Capacity_MW': capacity
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return None
        
        # Create stacked area chart
        fig = px.area(
            df,
            x='Year',
            y='Capacity_MW',
            color='Technology',
            title=f'Generation Capacity Mix Evolution - {scenario_name}',
            labels={'Capacity_MW': 'Capacity (MW)'}
        )
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Capacity (MW)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_investment_timeline(self, results, scenario_name="Scenario"):
        """Create investment timeline chart"""
        
        if 'costs' not in results:
            return None
        
        # Prepare data
        years = []
        investments = []
        
        for year, cost_data in results['costs'].items():
            if isinstance(cost_data, dict) and 'total_investment_million_usd' in cost_data:
                years.append(year)
                investments.append(cost_data['total_investment_million_usd'])
        
        if not years:
            return None
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=years, y=investments, name='Investment')
        ])
        
        fig.update_layout(
            title=f'Annual Investment Requirements - {scenario_name}',
            xaxis_title='Year',
            yaxis_title='Investment (Million USD)',
            showlegend=False
        )
        
        return fig
    
    def create_investment_by_technology(self, results, scenario_name="Scenario"):
        """Create investment breakdown by technology"""
        
        if 'costs' not in results:
            return None
        
        # Prepare data
        data = []
        for year, cost_data in results['costs'].items():
            if isinstance(cost_data, dict) and 'investment_by_technology' in cost_data:
                for tech, investment in cost_data['investment_by_technology'].items():
                    data.append({
                        'Year': year,
                        'Technology': tech,
                        'Investment_MUSD': investment
                    })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Create stacked bar chart
        fig = px.bar(
            df,
            x='Year',
            y='Investment_MUSD',
            color='Technology',
            title=f'Investment by Technology - {scenario_name}',
            labels={'Investment_MUSD': 'Investment (Million USD)'}
        )
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Investment (Million USD)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_emissions_chart(self, results, scenario_name="Scenario"):
        """Create emissions evolution chart"""
        
        if 'emissions' not in results:
            return None
        
        # Prepare data
        years = []
        emissions = []
        intensity = []
        
        for year, emission_data in results['emissions'].items():
            if isinstance(emission_data, dict):
                years.append(year)
                emissions.append(emission_data.get('annual_co2_tonnes', 0))
                intensity.append(emission_data.get('emissions_intensity_kg_per_mwh', 0))
        
        if not years:
            return None
        
        # Create dual-axis chart
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=[f'Emissions Evolution - {scenario_name}']
        )
        
        fig.add_trace(
            go.Scatter(x=years, y=emissions, name='Annual Emissions'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=years, y=intensity, name='Emission Intensity', line=dict(dash='dash')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Annual Emissions (tonnes CO2)", secondary_y=False)
        fig.update_yaxes(title_text="Intensity (kg CO2/MWh)", secondary_y=True)
        
        return fig
    
    def create_renewable_share_chart(self, results, scenario_name="Scenario"):
        """Create renewable energy share evolution chart"""
        
        if 'supply_security' not in results:
            return None
        
        # Prepare data
        years = []
        renewable_shares = []
        
        for year, security_data in results['supply_security'].items():
            if isinstance(security_data, dict):
                years.append(year)
                renewable_shares.append(security_data.get('renewable_share_percent', 0))
        
        if not years:
            return None
        
        # Create line chart
        fig = go.Figure(data=go.Scatter(
            x=years,
            y=renewable_shares,
            mode='lines+markers',
            name='Renewable Share',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'Renewable Energy Share Evolution - {scenario_name}',
            xaxis_title='Year',
            yaxis_title='Renewable Share (%)',
            yaxis=dict(range=[0, 100])
        )
        
        # Add target lines if available
        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                     annotation_text="50% Target")
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="80% Target")
        
        return fig
    
    def create_cost_comparison_chart(self, results, scenario_name="Scenario"):
        """Create LCOE evolution chart"""
        
        if 'costs' not in results:
            return None
        
        # Prepare data
        years = []
        lcoe_values = []
        
        for year, cost_data in results['costs'].items():
            if isinstance(cost_data, dict) and 'lcoe_usd_per_mwh' in cost_data:
                years.append(year)
                lcoe_values.append(cost_data['lcoe_usd_per_mwh'])
        
        if not years:
            return None
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=years, y=lcoe_values, name='LCOE', marker_color='lightblue')
        ])
        
        fig.update_layout(
            title=f'Levelized Cost of Electricity (LCOE) - {scenario_name}',
            xaxis_title='Year',
            yaxis_title='LCOE (USD/MWh)',
            showlegend=False
        )
        
        return fig
    
    def create_scenario_comparison_chart(self, scenario_results):
        """Create multi-scenario comparison chart"""
        
        metrics = ['renewable_share', 'total_investment', 'emissions']
        scenarios = list(scenario_results.keys())
        
        if len(scenarios) < 2:
            return None
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            specs=[[{"type": "bar"}] * len(metrics)]
        )
        
        for i, metric in enumerate(metrics):
            values = []
            for scenario in scenarios:
                # Extract metric value (simplified)
                if metric == 'renewable_share':
                    value = scenario_results[scenario].get('summary', {}).get('final_renewable_share', 0)
                elif metric == 'total_investment':
                    value = scenario_results[scenario].get('summary', {}).get('total_investment_billion_usd', 0)
                elif metric == 'emissions':
                    value = scenario_results[scenario].get('summary', {}).get('cumulative_emissions_million_tonnes', 0)
                else:
                    value = 0
                values.append(value)
            
            fig.add_trace(
                go.Bar(x=scenarios, y=values, name=metric),
                row=1, col=i+1
            )
        
        fig.update_layout(title='Scenario Comparison', showlegend=False)
        return fig
    
    def create_capacity_factor_heatmap(self, network):
        """Create capacity factor heatmap for renewable generators"""
        
        if network.generators.empty:
            return None
        
        # Get renewable generators
        renewable_gens = network.generators[
            network.generators.carrier.isin(['Solar', 'Wind'])
        ]
        
        if renewable_gens.empty:
            return None
        
        # Simulate capacity factor data (in real implementation, use actual data)
        np.random.seed(42)
        hours = min(168, len(network.snapshots))  # One week for visualization
        
        cf_data = []
        for gen_id, gen in renewable_gens.head(10).iterrows():  # Limit for visualization
            if gen['carrier'] == 'Solar':
                # Solar pattern
                daily_pattern = np.tile([0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0,
                                       1.0, 0.9, 0.8, 0.6, 0.3, 0.1, 0, 0, 0, 0, 0, 0], 7)
                cf_series = daily_pattern[:hours]
            else:  # Wind
                cf_series = np.random.uniform(0.1, 0.8, hours)
            
            cf_data.append(cf_series)
        
        cf_matrix = np.array(cf_data)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cf_matrix,
            x=list(range(hours)),
            y=[f"{gen_id}_{gen['carrier']}" for gen_id, gen in renewable_gens.head(10).iterrows()],
            colorscale='Viridis',
            colorbar=dict(title="Capacity Factor")
        ))
        
        fig.update_layout(
            title='Renewable Generator Capacity Factors',
            xaxis_title='Hour',
            yaxis_title='Generator'
        )
        
        return fig
    
    def create_demand_supply_chart(self, network):
        """Create demand vs supply visualization"""
        
        if network.loads_t.p_set.empty:
            return None
        
        # Get total demand and supply for first week
        hours = min(168, len(network.snapshots))
        
        total_demand = network.loads_t.p_set.sum(axis=1)[:hours]
        
        # Calculate available supply (simplified)
        available_supply = network.generators.p_nom.sum() * 0.7  # Assume 70% availability
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(hours)),
            y=total_demand.values,
            mode='lines',
            name='Demand',
            line=dict(color='red', width=2)
        ))
        
        fig.add_hline(
            y=available_supply,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Available Supply ({available_supply:.0f} MW)"
        )
        
        fig.update_layout(
            title='Demand vs Available Supply',
            xaxis_title='Hour',
            yaxis_title='Power (MW)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_regional_analysis_map(self, dre_data):
        """Create regional analysis visualization"""
        
        if 'Region or other (country-specific)' not in dre_data.columns:
            return None
        
        # Aggregate by region
        regional_summary = dre_data.groupby('Region or other (country-specific)').agg({
            'Population': 'sum',
            'Energy demand': 'sum',
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        
        fig = px.scatter_mapbox(
            regional_summary,
            lat='Latitude',
            lon='Longitude',
            size='Energy demand',
            hover_name='Region or other (country-specific)',
            hover_data=['Population', 'Energy demand'],
            size_max=50,
            zoom=6,
            center=dict(lat=9.1450, lon=40.4897),
            mapbox_style='open-street-map',
            title='Regional Energy Demand Analysis'
        )
        
        return fig
    
    def create_dashboard_summary(self, results, scenario_name="Scenario"):
        """Create dashboard summary with key metrics"""
        
        summary = results.get('summary', {})
        
        # Key metrics
        metrics = {
            'Total Investment': f"${summary.get('total_investment_billion_usd', 0):.1f}B",
            'Final Renewable Share': f"{summary.get('final_renewable_share', 0):.1f}%",
            'Cumulative Emissions': f"{summary.get('cumulative_emissions_million_tonnes', 0):.1f}M tonnes",
            'Planning Horizon': f"{min(results.get('costs', {}).keys(), default='N/A')} - {max(results.get('costs', {}).keys(), default='N/A')}"
        }
        
        return metrics