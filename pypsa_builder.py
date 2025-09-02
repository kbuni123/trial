"""
PyPSA Network Builder for Ethiopia Energy Planning Tool

Creates PyPSA network from processed DRE Atlas, transmission, and generator data
"""

import pandas as pd
import numpy as np
import pypsa
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class EthiopiaPyPSABuilder:
    """Build PyPSA network for Ethiopia energy system"""
    
    def __init__(self):
        self.network = None
        self.bus_mapping = {}
        self.technology_costs = self.get_technology_costs()
        
    def build_network(self, dre_data, transmission_data=None, generator_data=None, 
                     voltage_levels=[400, 230, 132], clustering_method="K-means", 
                     n_clusters=50, base_year=2025):
        """Build complete PyPSA network"""
        
        # Initialize PyPSA network
        self.network = pypsa.Network()
        self.network.set_snapshots(pd.date_range(f"{base_year}-01-01", periods=8760, freq="h"))
        
        # Create buses from settlements
        if clustering_method == "No clustering":
            buses_df = self.create_buses_from_settlements(dre_data, voltage_levels)
        else:
            buses_df = self.create_clustered_buses(dre_data, clustering_method, n_clusters, voltage_levels)
        
        # Add buses to network
        for _, bus in buses_df.iterrows():
            self.network.add("Bus", 
                           bus['bus_id'], 
                           x=bus['x'], 
                           y=bus['y'], 
                           v_nom=bus['v_nom'])
        
        # Create loads from demand data
        self.add_loads_to_network(dre_data, buses_df, base_year)
        
        # Add existing generators
        if generator_data is not None:
            self.add_existing_generators(generator_data, buses_df)
        
        # Add transmission lines
        if transmission_data is not None:
            self.add_transmission_lines(transmission_data, buses_df)
        else:
            # Create basic transmission network
            self.create_basic_transmission_network(buses_df)
        
        # Add potential renewable sites
        self.add_renewable_potential(dre_data, buses_df)
        
        # Add storage options
        self.add_storage_technologies(buses_df)
        
        return self.network
    
    def create_buses_from_settlements(self, dre_data, voltage_levels):
        """Create individual buses for each settlement"""
        buses = []
        
        for idx, settlement in dre_data.iterrows():
            if pd.notna(settlement['Latitude']) and pd.notna(settlement['Longitude']):
                buses.append({
                    'bus_id': f"bus_{idx}",
                    'settlement_name': settlement.get('Name', f'Settlement_{idx}'),
                    'x': settlement['Longitude'],
                    'y': settlement['Latitude'],
                    'v_nom': min(voltage_levels),  # Use lowest voltage for local connections
                    'population': settlement.get('Population', 0),
                    'demand': settlement.get('Energy demand', 0),
                    'region': settlement.get('Region or other (country-specific)', 'Unknown')
                })
        
        return pd.DataFrame(buses)
    
    def create_clustered_buses(self, dre_data, clustering_method, n_clusters, voltage_levels):
        """Create buses by clustering settlements"""
        
        # Prepare data for clustering
        valid_settlements = dre_data.dropna(subset=['Latitude', 'Longitude'])
        coordinates = valid_settlements[['Longitude', 'Latitude']].values
        
        if len(valid_settlements) < n_clusters:
            n_clusters = len(valid_settlements)
        
        # Apply clustering
        if clustering_method == "K-means":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif clustering_method == "Hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif clustering_method == "Administrative":
            return self.create_administrative_buses(valid_settlements, voltage_levels)
        
        cluster_labels = clusterer.fit_predict(coordinates)
        valid_settlements['cluster'] = cluster_labels
        
        # Create buses from clusters
        buses = []
        for cluster_id in range(n_clusters):
            cluster_settlements = valid_settlements[valid_settlements['cluster'] == cluster_id]
            
            if len(cluster_settlements) > 0:
                # Calculate cluster center (weighted by population)
                weights = cluster_settlements.get('Population', pd.Series([1]*len(cluster_settlements)))
                weights = weights.fillna(1)
                
                center_x = np.average(cluster_settlements['Longitude'], weights=weights)
                center_y = np.average(cluster_settlements['Latitude'], weights=weights)
                
                # Aggregate cluster data
                total_population = cluster_settlements['Population'].sum()
                total_demand = cluster_settlements['Energy demand'].sum()
                
                # Determine voltage level based on cluster size
                if total_population > 100000:
                    v_nom = max(voltage_levels)
                elif total_population > 20000:
                    v_nom = voltage_levels[len(voltage_levels)//2] if len(voltage_levels) > 1 else voltage_levels[0]
                else:
                    v_nom = min(voltage_levels)
                
                buses.append({
                    'bus_id': f"cluster_{cluster_id}",
                    'settlement_name': f"Cluster_{cluster_id}",
                    'x': center_x,
                    'y': center_y,
                    'v_nom': v_nom,
                    'population': total_population,
                    'demand': total_demand,
                    'n_settlements': len(cluster_settlements),
                    'region': cluster_settlements['Region or other (country-specific)'].mode().iloc[0] if 'Region or other (country-specific)' in cluster_settlements.columns else 'Mixed'
                })
        
        return pd.DataFrame(buses)
    
    def create_administrative_buses(self, settlements, voltage_levels):
        """Create buses based on administrative divisions"""
        buses = []
        
        if 'District or other (country-specific)' in settlements.columns:
            group_col = 'District or other (country-specific)'
        elif 'Region or other (country-specific)' in settlements.columns:
            group_col = 'Region or other (country-specific)'
        else:
            # Fallback to geographic clustering
            return self.create_clustered_buses(settlements, "K-means", 50, voltage_levels)
        
        for district, district_settlements in settlements.groupby(group_col):
            if len(district_settlements) > 0:
                # Calculate district center
                weights = district_settlements.get('Population', pd.Series([1]*len(district_settlements)))
                weights = weights.fillna(1)
                
                center_x = np.average(district_settlements['Longitude'], weights=weights)
                center_y = np.average(district_settlements['Latitude'], weights=weights)
                
                total_population = district_settlements['Population'].sum()
                total_demand = district_settlements['Energy demand'].sum()
                
                # Determine voltage level
                if total_population > 200000:
                    v_nom = max(voltage_levels)
                elif total_population > 50000:
                    v_nom = voltage_levels[1] if len(voltage_levels) > 1 else voltage_levels[0]
                else:
                    v_nom = min(voltage_levels)
                
                buses.append({
                    'bus_id': f"district_{district}".replace(" ", "_"),
                    'settlement_name': str(district),
                    'x': center_x,
                    'y': center_y,
                    'v_nom': v_nom,
                    'population': total_population,
                    'demand': total_demand,
                    'n_settlements': len(district_settlements),
                    'region': district
                })
        
        return pd.DataFrame(buses)
    
    def add_loads_to_network(self, dre_data, buses_df, base_year):
        """Add electrical loads to the network"""
        
        for _, bus in buses_df.iterrows():
            demand = bus.get('demand', 0)
            
            if demand > 0:
                # Convert daily demand to hourly MW
                hourly_demand_mw = demand / 24 / 1000  # kWh/day to MW
                
                # Create load profile (simplified)
                load_profile = self.create_load_profile(hourly_demand_mw)
                
                # Add load to network
                load_name = f"load_{bus['bus_id']}"
                self.network.add("Load",
                               load_name,
                               bus=bus['bus_id'],
                               p_set=load_profile)
    
    def create_load_profile(self, base_demand_mw):
        """Create hourly load profile for a year"""
        hours = len(self.network.snapshots)
        
        # Simple load profile with daily and seasonal patterns
        hourly_pattern = np.array([
            0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 1.0, 1.0, 0.9,
            0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.2, 1.4, 1.3, 1.1,
            1.0, 0.9, 0.8, 0.7
        ])
        
        # Repeat pattern for the year with seasonal variation
        days = hours // 24
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * np.arange(days) / 365)
        
        load_profile = []
        for day in range(days):
            daily_loads = hourly_pattern * base_demand_mw * seasonal_factor[day]
            load_profile.extend(daily_loads)
        
        # Handle remaining hours
        remaining_hours = hours - len(load_profile)
        if remaining_hours > 0:
            load_profile.extend([base_demand_mw] * remaining_hours)
        
        return np.array(load_profile[:hours])
    
    def add_existing_generators(self, generator_data, buses_df):
        """Add existing power plants to the network"""
        
        for _, gen in generator_data.iterrows():
            if pd.notna(gen['Latitude']) and pd.notna(gen['Longitude']):
                # Find nearest bus
                nearest_bus = self.find_nearest_bus(
                    gen['Longitude'], gen['Latitude'], buses_df
                )
                
                if nearest_bus is not None:
                    fuel_type = gen.get('Fuel_Type_Clean', 'Other')
                    capacity = gen.get('Capacity_MW', 0)
                    
                    # Get technology parameters
                    tech_params = self.get_generator_params(fuel_type)
                    
                    gen_name = f"gen_{gen.name}_{fuel_type}"
                    
                    if fuel_type in ['Solar', 'Wind']:
                        # Variable renewable generators
                        capacity_factor = self.get_renewable_profile(fuel_type)
                        
                        self.network.add("Generator",
                                       gen_name,
                                       bus=nearest_bus,
                                       p_nom=capacity,
                                       marginal_cost=tech_params['marginal_cost'],
                                       capital_cost=tech_params['capital_cost'],
                                       p_max_pu=capacity_factor,
                                       carrier=fuel_type)
                    else:
                        # Dispatchable generators
                        self.network.add("Generator",
                                       gen_name,
                                       bus=nearest_bus,
                                       p_nom=capacity,
                                       marginal_cost=tech_params['marginal_cost'],
                                       capital_cost=tech_params['capital_cost'],
                                       efficiency=tech_params.get('efficiency', 0.4),
                                       carrier=fuel_type)
    
    def find_nearest_bus(self, lon, lat, buses_df):
        """Find nearest bus to given coordinates"""
        if len(buses_df) == 0:
            return None
        
        distances = cdist([[lon, lat]], buses_df[['x', 'y']].values)[0]
        nearest_idx = np.argmin(distances)
        
        return buses_df.iloc[nearest_idx]['bus_id']
    
    def add_transmission_lines(self, transmission_data, buses_df):
        """Add existing transmission lines to network"""
        
        line_id = 0
        for _, line in transmission_data.iterrows():
            # Extract line endpoints
            line_geom = line.geometry
            
            if line_geom.geom_type == 'LineString':
                start_coords = line_geom.coords[0]
                end_coords = line_geom.coords[-1]
                
                # Find nearest buses
                start_bus = self.find_nearest_bus(start_coords[0], start_coords[1], buses_df)
                end_bus = self.find_nearest_bus(end_coords[0], end_coords[1], buses_df)
                
                if start_bus and end_bus and start_bus != end_bus:
                    voltage = line.get('Voltage_kV', 132)
                    length = line.get('Length_km', 100)
                    capacity = line.get('Estimated_Capacity_MW', 200)
                    
                    # Calculate line parameters
                    line_params = self.get_line_params(voltage, length)
                    
                    self.network.add("Line",
                                   f"line_{line_id}",
                                   bus0=start_bus,
                                   bus1=end_bus,
                                   length=length,
                                   r=line_params['r'],
                                   x=line_params['x'],
                                   c=line_params['c'],
                                   s_nom=capacity)
                    
                    line_id += 1
    
    def create_basic_transmission_network(self, buses_df):
        """Create basic transmission network connecting nearby buses"""
        
        if len(buses_df) < 2:
            return
        
        # Calculate distances between all buses
        coords = buses_df[['x', 'y']].values
        distances = cdist(coords, coords)
        
        # Connect each bus to nearest neighbors
        for i, bus in buses_df.iterrows():
            bus_distances = distances[i]
            nearest_indices = np.argsort(bus_distances)[1:4]  # 3 nearest neighbors (excluding self)
            
            for j in nearest_indices:
                if bus_distances[j] < 2.0 and i < j:  # Max 2 degrees distance, avoid duplicates
                    bus1 = buses_df.iloc[i]['bus_id']
                    bus2 = buses_df.iloc[j]['bus_id']
                    
                    length = bus_distances[j] * 111  # Convert degrees to km (approximate)
                    voltage = max(buses_df.iloc[i]['v_nom'], buses_df.iloc[j]['v_nom'])
                    
                    line_params = self.get_line_params(voltage, length)
                    capacity = self.estimate_line_capacity(voltage)
                    
                    self.network.add("Line",
                                   f"line_{i}_{j}",
                                   bus0=bus1,
                                   bus1=bus2,
                                   length=length,
                                   r=line_params['r'],
                                   x=line_params['x'],
                                   c=line_params['c'],
                                   s_nom=capacity)
    
    def add_renewable_potential(self, dre_data, buses_df):
        """Add potential renewable generation sites"""
        
        for _, bus in buses_df.iterrows():
            bus_id = bus['bus_id']
            
            # Solar potential
            if hasattr(self, '_get_bus_solar_potential'):
                solar_potential = self._get_bus_solar_potential(bus, dre_data)
            else:
                solar_potential = 1000  # Default 1000 MW potential
            
            if solar_potential > 0:
                solar_cf = self.get_renewable_profile('Solar')
                
                self.network.add("Generator",
                               f"solar_{bus_id}",
                               bus=bus_id,
                               p_nom_extendable=True,
                               p_nom_max=solar_potential,
                               marginal_cost=0,
                               capital_cost=self.technology_costs['Solar']['capital_cost'],
                               p_max_pu=solar_cf,
                               carrier='Solar')
            
            # Wind potential (simplified - would need wind resource data)
            wind_potential = 500  # Default 500 MW potential
            
            if wind_potential > 0:
                wind_cf = self.get_renewable_profile('Wind')
                
                self.network.add("Generator",
                               f"wind_{bus_id}",
                               bus=bus_id,
                               p_nom_extendable=True,
                               p_nom_max=wind_potential,
                               marginal_cost=0,
                               capital_cost=self.technology_costs['Wind']['capital_cost'],
                               p_max_pu=wind_cf,
                               carrier='Wind')
    
    def add_storage_technologies(self, buses_df):
        """Add potential storage technologies"""
        
        for _, bus in buses_df.iterrows():
            bus_id = bus['bus_id']
            
            # Battery storage
            self.network.add("StorageUnit",
                           f"battery_{bus_id}",
                           bus=bus_id,
                           p_nom_extendable=True,
                           p_nom_max=500,  # Max 500 MW
                           capital_cost=self.technology_costs['Battery']['capital_cost'],
                           marginal_cost=0.01,
                           efficiency_store=0.95,
                           efficiency_dispatch=0.95,
                           max_hours=4,  # 4-hour battery
                           carrier='Battery')
    
    def get_renewable_profile(self, technology):
        """Generate renewable energy capacity factor profiles"""
        hours = len(self.network.snapshots)
        
        if technology == 'Solar':
            # Simplified solar profile
            daily_solar = np.array([0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0,
                                  1.0, 0.9, 0.8, 0.6, 0.3, 0.1, 0, 0, 0, 0, 0, 0])
            
            # Add seasonal variation
            days = hours // 24
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(days) / 365 - np.pi/2)
            
            profile = []
            for day in range(days):
                daily_profile = daily_solar * seasonal_factor[day]
                profile.extend(daily_profile)
            
            # Handle remaining hours
            remaining_hours = hours - len(profile)
            if remaining_hours > 0:
                profile.extend([0] * remaining_hours)
            
            return np.array(profile[:hours])
        
        elif technology == 'Wind':
            # Simplified wind profile with more randomness
            np.random.seed(42)  # For reproducibility
            base_profile = np.random.uniform(0.1, 0.8, hours)
            
            # Add some daily pattern
            hourly_factor = np.tile(np.array([0.8, 0.9, 1.0, 1.1, 1.0, 0.9] * 4), hours // 24 + 1)[:hours]
            
            return np.clip(base_profile * hourly_factor, 0, 1)
        
        return np.ones(hours) * 0.3  # Default capacity factor
    
    def get_technology_costs(self):
        """Get technology cost parameters (2025 values)"""
        return {
            'Solar': {
                'capital_cost': 800,  # $/kW
                'marginal_cost': 0,
                'lifetime': 25
            },
            'Wind': {
                'capital_cost': 1200,  # $/kW
                'marginal_cost': 0,
                'lifetime': 25
            },
            'Gas': {
                'capital_cost': 600,  # $/kW
                'marginal_cost': 40,  # $/MWh
                'efficiency': 0.45,
                'lifetime': 25
            },
            'Hydro': {
                'capital_cost': 2000,  # $/kW
                'marginal_cost': 5,   # $/MWh
                'lifetime': 50
            },
            'Battery': {
                'capital_cost': 300,  # $/kWh
                'marginal_cost': 0.01,
                'lifetime': 15
            },
            'Coal': {
                'capital_cost': 1500,  # $/kW
                'marginal_cost': 30,   # $/MWh
                'efficiency': 0.35,
                'lifetime': 40
            }
        }
    
    def get_generator_params(self, fuel_type):
        """Get generator parameters by fuel type"""
        return self.technology_costs.get(fuel_type, {
            'capital_cost': 1000,
            'marginal_cost': 50,
            'efficiency': 0.35
        })
    
    def get_line_params(self, voltage_kv, length_km):
        """Calculate transmission line parameters"""
        
        # Simplified line parameters based on voltage level
        if voltage_kv >= 400:
            r_per_km = 0.03  # ohm/km
            x_per_km = 0.3   # ohm/km
            c_per_km = 0.015 # μF/km
        elif voltage_kv >= 200:
            r_per_km = 0.05
            x_per_km = 0.4
            c_per_km = 0.010
        else:
            r_per_km = 0.1
            x_per_km = 0.4
            c_per_km = 0.005
        
        return {
            'r': r_per_km * length_km,
            'x': x_per_km * length_km,
            'c': c_per_km * length_km * 1e-6  # Convert to F
        }
    
    def estimate_line_capacity(self, voltage_kv):
        """Estimate transmission line capacity based on voltage"""
        capacity_map = {
            400: 1000,  # MW
            230: 400,
            132: 150,
            66: 50,
            45: 30
        }
        
        # Find closest voltage level
        closest_voltage = min(capacity_map.keys(), key=lambda x: abs(x - voltage_kv))
        return capacity_map[closest_voltage]
    
    def optimize_network(self, network=None, solver='cbc'):
        """Run basic optimization on the network"""
        if network is None:
            network = self.network
        
        try:
            # Solve optimal power flow
            network.lopf(solver=solver, pyomo=False)
            return True
        except Exception as e:
            print(f"Optimization failed: {e}")
            return False
    
    def get_network_statistics(self):
        """Get network statistics"""
        if self.network is None:
            return {}
        
        stats = {
            'n_buses': len(self.network.buses),
            'n_lines': len(self.network.lines),
            'n_generators': len(self.network.generators),
            'n_loads': len(self.network.loads),
            'total_load': self.network.loads_t.p_set.sum().sum(),
            'total_generation_capacity': self.network.generators.p_nom.sum(),
            'renewable_capacity': self.network.generators[
                self.network.generators.carrier.isin(['Solar', 'Wind', 'Hydro'])
            ].p_nom.sum()
        }
        
        if stats['total_generation_capacity'] > 0:
            stats['renewable_share'] = (
                stats['renewable_capacity'] / stats['total_generation_capacity'] * 100
            )
        
        return stats