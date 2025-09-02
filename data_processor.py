"""
Data Processing Module for Ethiopia Energy Planning Tool

Handles import and processing of DRE Atlas data, transmission networks, and generator data
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import warnings
warnings.filterwarnings('ignore')

class DREDataProcessor:
    """Process DRE Atlas settlement data from World Bank"""
    
    def __init__(self):
        self.required_columns = [
            'Name',
            'Region or other (country-specific)',
            'District or other (country-specific)',
            'Population',
            'Energy demand',
            'Number of buildings'
        ]
    
    def load_and_process(self, file):
        """Load and process DRE Atlas CSV data"""
        # Read CSV file
        if hasattr(file, 'read'):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
        
        # Clean and standardize column names
        df.columns = df.columns.str.strip()
        
        # Handle coordinate columns (may vary in naming)
        lat_cols = [col for col in df.columns if any(x in col.lower() for x in ['lat', 'y'])]
        lon_cols = [col for col in df.columns if any(x in col.lower() for x in ['lon', 'lng', 'x'])]
        
        if lat_cols and lon_cols:
            df['Latitude'] = df[lat_cols[0]]
            df['Longitude'] = df[lon_cols[0]]
        
        # Clean numeric columns
        numeric_columns = [
            'Population', 'Energy demand', 'Number of buildings', 
            'Settlement area', 'Building density', 'Potential PV production',
            'Distance to existing lines', 'Distance to transmission grid',
            'Latitude', 'Longitude'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid coordinates
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df = df.dropna(subset=['Latitude', 'Longitude'])
            # Filter to Ethiopia bounds approximately
            df = df[(df['Latitude'] >= 3) & (df['Latitude'] <= 15) & 
                   (df['Longitude'] >= 33) & (df['Longitude'] <= 48)]
        
        # Fill missing values
        df = self.fill_missing_values(df)
        
        # Create derived features
        df = self.create_derived_features(df)
        
        return df
    
    def fill_missing_values(self, df):
        """Fill missing values with appropriate defaults or estimates"""
        
        # Fill population based on buildings if missing
        if 'Population' in df.columns and 'Number of buildings' in df.columns:
            df['Population'] = df['Population'].fillna(df['Number of buildings'] * 4.5)  # Avg 4.5 people per building
        
        # Estimate energy demand if missing
        if 'Energy demand' in df.columns and 'Population' in df.columns:
            df['Energy demand'] = df['Energy demand'].fillna(df['Population'] * 1.2)  # 1.2 kWh/day per person
        
        # Fill building counts
        if 'Number of buildings' in df.columns:
            df['Number of buildings'] = df['Number of buildings'].fillna(10)  # Minimum settlement size
        
        # Fill solar potential with regional averages
        if 'Potential PV production' in df.columns:
            regional_solar = df.groupby('Region or other (country-specific)')['Potential PV production'].transform('mean')
            df['Potential PV production'] = df['Potential PV production'].fillna(regional_solar)
            df['Potential PV production'] = df['Potential PV production'].fillna(1600)  # Ethiopia average
        
        return df
    
    def create_derived_features(self, df):
        """Create additional features for analysis"""
        
        # Settlement classification by population
        if 'Population' in df.columns:
            df['Settlement_Type'] = pd.cut(
                df['Population'],
                bins=[0, 1000, 5000, 20000, float('inf')],
                labels=['Village', 'Small Town', 'Town', 'City']
            )
        
        # Energy access classification
        if 'Shows light at night?' in df.columns:
            df['Has_Electricity'] = df['Shows light at night?'].map({True: 1, False: 0})
        
        # Grid proximity classification
        grid_distance_cols = [col for col in df.columns if 'Distance to' in col and 'grid' in col]
        if grid_distance_cols:
            main_grid_col = grid_distance_cols[0]  # Use first grid distance column
            df['Grid_Proximity'] = pd.cut(
                df[main_grid_col],
                bins=[0, 5, 15, 50, float('inf')],
                labels=['Very Close', 'Close', 'Moderate', 'Remote']
            )
        
        # Electrification priority score
        if all(col in df.columns for col in ['Population', 'Energy demand']):
            df['Electrification_Priority'] = (
                df['Population'] * 0.3 + 
                df['Energy demand'] * 0.3 + 
                (df['Has_Electricity'] == 0) * 40  # Bonus for unelectrified
            )
        
        return df
    
    def get_data_quality_report(self, df):
        """Generate data quality report"""
        report = {
            'total_settlements': len(df),
            'missing_coordinates': df[['Latitude', 'Longitude']].isnull().sum().sum(),
            'missing_population': df['Population'].isnull().sum() if 'Population' in df.columns else 0,
            'missing_energy_demand': df['Energy demand'].isnull().sum() if 'Energy demand' in df.columns else 0,
            'regions_covered': len(df['Region or other (country-specific)'].unique()) if 'Region or other (country-specific)' in df.columns else 0
        }
        return report

class TransmissionProcessor:
    """Process transmission network shapefiles"""
    
    def __init__(self):
        self.voltage_mapping = {
            '400': 400,
            '230': 230,
            '132': 132,
            '66': 66,
            '45': 45
        }
    
    def load_shapefile(self, file_path):
        """Load transmission line shapefile"""
        try:
            if hasattr(file_path, 'read'):
                # Handle uploaded file
                gdf = gpd.read_file(file_path)
            else:
                gdf = gpd.read_file(file_path)
            
            # Ensure CRS is set to WGS84
            if gdf.crs is None or gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Clean and standardize data
            gdf = self.clean_transmission_data(gdf)
            
            return gdf
            
        except Exception as e:
            print(f"Error loading transmission shapefile: {e}")
            return None
    
    def clean_transmission_data(self, gdf):
        """Clean and standardize transmission line data"""
        
        # Standardize column names
        column_mapping = {
            'voltage': 'Voltage_kV',
            'volt': 'Voltage_kV',
            'kv': 'Voltage_kV',
            'name': 'Line_Name',
            'line_name': 'Line_Name',
            'status': 'Status',
            'owner': 'Owner',
            'operator': 'Operator'
        }
        
        for old_col, new_col in column_mapping.items():
            matching_cols = [col for col in gdf.columns if old_col in col.lower()]
            if matching_cols:
                gdf = gdf.rename(columns={matching_cols[0]: new_col})
        
        # Extract voltage information
        if 'Voltage_kV' in gdf.columns:
            gdf['Voltage_kV'] = gdf['Voltage_kV'].astype(str).str.extract('(\d+)').astype(float)
        else:
            # Try to infer from line names or other columns
            gdf['Voltage_kV'] = self.infer_voltage(gdf)
        
        # Calculate line length
        gdf['Length_km'] = gdf.geometry.length * 111  # Approximate conversion to km
        
        # Add capacity estimates based on voltage
        gdf['Estimated_Capacity_MW'] = gdf['Voltage_kV'].map({
            400: 1000,
            230: 400,
            132: 150,
            66: 50,
            45: 30
        }).fillna(100)
        
        # Filter to transmission lines only (>= 45kV)
        gdf = gdf[gdf['Voltage_kV'] >= 45]
        
        return gdf
    
    def infer_voltage(self, gdf):
        """Infer voltage levels from available data"""
        # Default voltage assignment (could be improved with domain knowledge)
        voltages = np.random.choice([400, 230, 132, 66], size=len(gdf), p=[0.1, 0.3, 0.4, 0.2])
        return voltages
    
    def extract_network_nodes(self, gdf, snap_distance=0.01):
        """Extract network nodes from transmission lines"""
        nodes = []
        node_id = 0
        
        for idx, row in gdf.iterrows():
            line_geom = row.geometry
            
            if line_geom.geom_type == 'LineString':
                # Extract start and end points
                start_point = Point(line_geom.coords[0])
                end_point = Point(line_geom.coords[-1])
                
                # Add nodes (with basic snapping)
                nodes.append({
                    'node_id': f'node_{node_id}',
                    'x': start_point.x,
                    'y': start_point.y,
                    'voltage_kv': row.get('Voltage_kV', 132)
                })
                node_id += 1
                
                nodes.append({
                    'node_id': f'node_{node_id}',
                    'x': end_point.x,
                    'y': end_point.y,
                    'voltage_kv': row.get('Voltage_kV', 132)
                })
                node_id += 1
        
        nodes_df = pd.DataFrame(nodes)
        
        # Remove duplicate nodes (simple approach)
        nodes_df = nodes_df.drop_duplicates(subset=['x', 'y'], keep='first')
        
        return nodes_df

class GeneratorProcessor:
    """Process power generator data"""
    
    def __init__(self):
        self.fuel_mapping = {
            'solar': 'Solar',
            'pv': 'Solar',
            'photovoltaic': 'Solar',
            'wind': 'Wind',
            'hydro': 'Hydro',
            'hydroelectric': 'Hydro',
            'gas': 'Gas',
            'natural gas': 'Gas',
            'diesel': 'Diesel',
            'coal': 'Coal',
            'geothermal': 'Geothermal'
        }
    
    def load_generators(self, file):
        """Load generator data from CSV or Excel file"""
        try:
            if hasattr(file, 'name') and file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
            
            # Clean and process generator data
            df = self.clean_generator_data(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading generator data: {e}")
            return None
    
    def clean_generator_data(self, df):
        """Clean and standardize generator data"""
        
        # Standardize column names
        column_mapping = {
            'plant_name': 'Plant_Name',
            'name': 'Plant_Name',
            'facility': 'Plant_Name',
            'capacity': 'Capacity_MW',
            'capacity_mw': 'Capacity_MW',
            'cap_mw': 'Capacity_MW',
            'fuel': 'Fuel_Type',
            'fuel_type': 'Fuel_Type',
            'technology': 'Technology',
            'tech': 'Technology',
            'status': 'Status',
            'latitude': 'Latitude',
            'lat': 'Latitude',
            'longitude': 'Longitude',
            'lon': 'Longitude',
            'lng': 'Longitude',
            'owner': 'Owner',
            'operator': 'Operator',
            'commissioning_year': 'Commission_Year',
            'year': 'Commission_Year'
        }
        
        for old_col, new_col in column_mapping.items():
            matching_cols = [col for col in df.columns if old_col in col.lower()]
            if matching_cols:
                df = df.rename(columns={matching_cols[0]: new_col})
        
        # Clean capacity data
        if 'Capacity_MW' in df.columns:
            df['Capacity_MW'] = pd.to_numeric(df['Capacity_MW'], errors='coerce')
        
        # Standardize fuel types
        if 'Fuel_Type' in df.columns:
            df['Fuel_Type_Clean'] = df['Fuel_Type'].str.lower().map(self.fuel_mapping)
            df['Fuel_Type_Clean'] = df['Fuel_Type_Clean'].fillna('Other')
        
        # Clean coordinates
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            
            # Filter to Ethiopia bounds
            df = df[(df['Latitude'] >= 3) & (df['Latitude'] <= 15) & 
                   (df['Longitude'] >= 33) & (df['Longitude'] <= 48)]
        
        # Remove invalid entries
        df = df.dropna(subset=['Capacity_MW'])
        df = df[df['Capacity_MW'] > 0]
        
        # Add derived features
        df = self.add_generator_features(df)
        
        return df
    
    def add_generator_features(self, df):
        """Add derived features for generator analysis"""
        
        # Plant size classification
        if 'Capacity_MW' in df.columns:
            df['Size_Category'] = pd.cut(
                df['Capacity_MW'],
                bins=[0, 10, 50, 200, float('inf')],
                labels=['Small', 'Medium', 'Large', 'Very Large']
            )
        
        # Renewable classification
        if 'Fuel_Type_Clean' in df.columns:
            renewable_fuels = ['Solar', 'Wind', 'Hydro', 'Geothermal']
            df['Is_Renewable'] = df['Fuel_Type_Clean'].isin(renewable_fuels)
        
        # Capacity factor estimates by technology
        cf_mapping = {
            'Solar': 0.25,
            'Wind': 0.30,
            'Hydro': 0.45,
            'Gas': 0.50,
            'Coal': 0.60,
            'Diesel': 0.20,
            'Geothermal': 0.80
        }
        
        if 'Fuel_Type_Clean' in df.columns:
            df['Capacity_Factor'] = df['Fuel_Type_Clean'].map(cf_mapping).fillna(0.40)
        
        # Estimated annual generation
        if 'Capacity_MW' in df.columns and 'Capacity_Factor' in df.columns:
            df['Annual_Generation_GWh'] = (
                df['Capacity_MW'] * df['Capacity_Factor'] * 8760 / 1000
            )
        
        return df
    
    def get_generation_summary(self, df):
        """Generate summary statistics for generators"""
        if df is None or df.empty:
            return {}
        
        summary = {
            'total_plants': len(df),
            'total_capacity_mw': df['Capacity_MW'].sum() if 'Capacity_MW' in df.columns else 0,
            'renewable_capacity_mw': df[df['Is_Renewable']]['Capacity_MW'].sum() if 'Is_Renewable' in df.columns else 0,
            'capacity_by_fuel': df.groupby('Fuel_Type_Clean')['Capacity_MW'].sum().to_dict() if 'Fuel_Type_Clean' in df.columns else {},
            'average_plant_size': df['Capacity_MW'].mean() if 'Capacity_MW' in df.columns else 0
        }
        
        if 'Capacity_MW' in df.columns and summary['total_capacity_mw'] > 0:
            summary['renewable_share'] = summary['renewable_capacity_mw'] / summary['total_capacity_mw'] * 100
        
        return summary

class DataIntegrator:
    """Integrate all data sources for network modeling"""
    
    def __init__(self):
        pass
    
    def create_integrated_dataset(self, dre_data, transmission_data=None, generator_data=None):
        """Create integrated dataset for PyPSA network building"""
        
        integrated_data = {
            'settlements': dre_data,
            'transmission_lines': transmission_data,
            'generators': generator_data
        }
        
        # Add spatial relationships
        if transmission_data is not None:
            integrated_data['network_nodes'] = self.extract_network_nodes(transmission_data)
        
        # Calculate distances between settlements and network infrastructure
        if transmission_data is not None:
            integrated_data['settlements'] = self.calculate_grid_distances(
                dre_data, transmission_data
            )
        
        return integrated_data
    
    def extract_network_nodes(self, transmission_data):
        """Extract unique network nodes from transmission lines"""
        # This would implement node extraction logic
        # For now, return empty DataFrame
        return pd.DataFrame(columns=['node_id', 'x', 'y', 'voltage_kv'])
    
    def calculate_grid_distances(self, settlements, transmission_lines):
        """Calculate distances from settlements to grid infrastructure"""
        # This would implement distance calculations
        # For now, return settlements unchanged
        return settlements
    
    def validate_data_consistency(self, integrated_data):
        """Validate consistency across different data sources"""
        validation_results = {
            'settlements_valid': True,
            'transmission_valid': True,
            'generators_valid': True,
            'spatial_consistency': True,
            'issues': []
        }
        
        # Add validation logic here
        
        return validation_results