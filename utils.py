"""
Utility Functions for Ethiopia Energy Planning Tool

Common helper functions used across different modules
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import os
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ConfigManager:
    """Manage configuration settings"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration if file doesn't exist"""
        return {
            'app': {'name': 'Ethiopia Energy Planning Tool', 'version': '1.0.0'},
            'geography': {
                'bounds': {'min_lat': 3.0, 'max_lat': 15.0, 'min_lon': 33.0, 'max_lon': 48.0},
                'center': {'lat': 9.1450, 'lon': 40.4897}
            },
            'planning': {
                'base_year': 2025,
                'planning_horizon': [2025, 2030, 2040, 2050],
                'default_demand_growth': 5.0
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

class DataValidator:
    """Validate input data quality and consistency"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.bounds = self.config.get('geography.bounds')
    
    def validate_dre_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate DRE Atlas data"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check required columns
        required_cols = ['Name', 'Population', 'Energy demand', 'Latitude', 'Longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            validation_results['is_valid'] = False
        
        # Check coordinate bounds
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            lat_out_of_bounds = (
                (df['Latitude'] < self.bounds['min_lat']) | 
                (df['Latitude'] > self.bounds['max_lat'])
            ).sum()
            
            lon_out_of_bounds = (
                (df['Longitude'] < self.bounds['min_lon']) | 
                (df['Longitude'] > self.bounds['max_lon'])
            ).sum()
            
            if lat_out_of_bounds > 0:
                validation_results['warnings'].append(
                    f"{lat_out_of_bounds} settlements have latitude outside Ethiopia bounds"
                )
            
            if lon_out_of_bounds > 0:
                validation_results['warnings'].append(
                    f"{lon_out_of_bounds} settlements have longitude outside Ethiopia bounds"
                )
        
        # Check data completeness
        if 'Population' in df.columns:
            pop_missing = df['Population'].isna().sum()
            pop_zero = (df['Population'] == 0).sum()
            
            validation_results['statistics']['population_missing'] = pop_missing
            validation_results['statistics']['population_zero'] = pop_zero
            
            if pop_missing / len(df) > 0.2:  # More than 20% missing
                validation_results['warnings'].append(
                    f"High percentage of missing population data: {pop_missing/len(df)*100:.1f}%"
                )
        
        # Check energy demand consistency
        if 'Energy demand' in df.columns:
            demand_missing = df['Energy demand'].isna().sum()
            demand_negative = (df['Energy demand'] < 0).sum()
            
            validation_results['statistics']['demand_missing'] = demand_missing
            validation_results['statistics']['demand_negative'] = demand_negative
            
            if demand_negative > 0:
                validation_results['errors'].append(f"{demand_negative} settlements have negative energy demand")
                validation_results['is_valid'] = False
        
        return validation_results
    
    def validate_transmission_data(self, gdf: gpd.GeoDataFrame) -> Dict[str, any]:
        """Validate transmission network shapefile"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check geometry type
        if not gdf.empty:
            geom_types = gdf.geometry.type.unique()
            if 'LineString' not in geom_types:
                validation_results['warnings'].append(
                    f"Expected LineString geometry, found: {geom_types}"
                )
        
        # Check CRS
        if gdf.crs is None:
            validation_results['warnings'].append("No CRS defined. Assuming EPSG:4326")
        elif gdf.crs != 'EPSG:4326':
            validation_results['warnings'].append(f"CRS is {gdf.crs}, will convert to EPSG:4326")
        
        # Check for required attributes
        voltage_cols = [col for col in gdf.columns if 'voltage' in col.lower() or 'kv' in col.lower()]
        if not voltage_cols:
            validation_results['warnings'].append("No voltage information found")
        
        validation_results['statistics']['total_lines'] = len(gdf)
        validation_results['statistics']['geometry_types'] = geom_types.tolist()
        
        return validation_results
    
    def validate_generator_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate generator/power plant data"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check required columns
        required_cols = ['Capacity_MW', 'Latitude', 'Longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            validation_results['is_valid'] = False
        
        # Check capacity values
        if 'Capacity_MW' in df.columns:
            negative_capacity = (df['Capacity_MW'] <= 0).sum()
            if negative_capacity > 0:
                validation_results['errors'].append(f"{negative_capacity} plants have non-positive capacity")
                validation_results['is_valid'] = False
            
            validation_results['statistics']['total_capacity_mw'] = df['Capacity_MW'].sum()
            validation_results['statistics']['average_capacity_mw'] = df['Capacity_MW'].mean()
        
        return validation_results

class GeoUtils:
    """Geographic utility functions"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2) * np.sin(dlat/2) + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * 
             np.sin(dlon/2) * np.sin(dlon/2))
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    @staticmethod
    def find_nearest_points(source_points: pd.DataFrame, target_points: pd.DataFrame,
                          source_lat: str = 'lat', source_lon: str = 'lon',
                          target_lat: str = 'lat', target_lon: str = 'lon') -> pd.DataFrame:
        """Find nearest target points for each source point"""
        
        results = []
        
        for idx, source in source_points.iterrows():
            distances = target_points.apply(
                lambda target: GeoUtils.calculate_distance(
                    source[source_lat], source[source_lon],
                    target[target_lat], target[target_lon]
                ),
                axis=1
            )
            
            nearest_idx = distances.idxmin()
            nearest_distance = distances.min()
            
            results.append({
                'source_idx': idx,
                'target_idx': nearest_idx,
                'distance_km': nearest_distance
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def create_grid_cells(bounds: Dict[str, float], cell_size_degrees: float = 0.1) -> gpd.GeoDataFrame:
        """Create a grid of cells over a geographic area"""
        from shapely.geometry import Polygon
        
        cells = []
        
        lons = np.arange(bounds['min_lon'], bounds['max_lon'], cell_size_degrees)
        lats = np.arange(bounds['min_lat'], bounds['max_lat'], cell_size_degrees)
        
        for i, lon in enumerate(lons[:-1]):
            for j, lat in enumerate(lats[:-1]):
                cell = Polygon([
                    (lon, lat),
                    (lon + cell_size_degrees, lat),
                    (lon + cell_size_degrees, lat + cell_size_degrees),
                    (lon, lat + cell_size_degrees)
                ])
                
                cells.append({
                    'grid_id': f'cell_{i}_{j}',
                    'geometry': cell
                })
        
        return gpd.GeoDataFrame(cells, crs='EPSG:4326')

class DataExporter:
    """Export data and results in various formats"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_dataframe(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> str:
        """Export DataFrame in specified format"""
        filepath = self.output_dir / f"{filename}.{format}"
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'xlsx':
            df.to_excel(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)
    
    def export_geodataframe(self, gdf: gpd.GeoDataFrame, filename: str, format: str = 'geojson') -> str:
        """Export GeoDataFrame in specified format"""
        filepath = self.output_dir / f"{filename}.{format}"
        
        if format.lower() == 'geojson':
            gdf.to_file(filepath, driver='GeoJSON')
        elif format.lower() == 'shp':
            gdf.to_file(filepath, driver='ESRI Shapefile')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)
    
    def export_pypsa_network(self, network, filename: str) -> str:
        """Export PyPSA network"""
        filepath = self.output_dir / f"{filename}.nc"
        network.export_to_netcdf(filepath)
        return str(filepath)
    
    def export_scenario_results(self, results: Dict, scenario_name: str) -> Dict[str, str]:
        """Export complete scenario results"""
        exports = {}
        
        # Export capacity evolution
        if 'generation_mix' in results:
            capacity_data = []
            for year, mix in results['generation_mix'].items():
                for tech, capacity in mix.items():
                    capacity_data.append({
                        'year': year,
                        'technology': tech,
                        'capacity_mw': capacity
                    })
            
            capacity_df = pd.DataFrame(capacity_data)
            exports['capacity_evolution'] = self.export_dataframe(
                capacity_df, f"{scenario_name}_capacity_evolution"
            )
        
        # Export investment timeline
        if 'costs' in results:
            investment_data = []
            for year, costs in results['costs'].items():
                if 'investment_by_technology' in costs:
                    for tech, investment in costs['investment_by_technology'].items():
                        investment_data.append({
                            'year': year,
                            'technology': tech,
                            'investment_million_usd': investment
                        })
            
            if investment_data:
                investment_df = pd.DataFrame(investment_data)
                exports['investment_timeline'] = self.export_dataframe(
                    investment_df, f"{scenario_name}_investment_timeline"
                )
        
        # Export summary metrics
        summary_data = []
        years = sorted(set().union(
            results.get('costs', {}).keys(),
            results.get('emissions', {}).keys(),
            results.get('supply_security', {}).keys()
        ))
        
        for year in years:
            row = {'year': year}
            
            # Add cost metrics
            if year in results.get('costs', {}):
                row.update({
                    'total_investment_musd': results['costs'][year].get('total_investment_million_usd', 0),
                    'lcoe_usd_mwh': results['costs'][year].get('lcoe_usd_per_mwh', 0)
                })
            
            # Add emission metrics
            if year in results.get('emissions', {}):
                row.update({
                    'emissions_tonnes_co2': results['emissions'][year].get('annual_co2_tonnes', 0),
                    'emission_intensity': results['emissions'][year].get('emissions_intensity_kg_per_mwh', 0)
                })
            
            # Add supply security metrics
            if year in results.get('supply_security', {}):
                row.update({
                    'renewable_share_pct': results['supply_security'][year].get('renewable_share_percent', 0),
                    'capacity_margin_pct': results['supply_security'][year].get('capacity_margin_percent', 0)
                })
            
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            exports['summary'] = self.export_dataframe(
                summary_df, f"{scenario_name}_summary"
            )
        
        return exports

class Logger:
    """Logging utility"""
    
    def __init__(self, name: str = "ethiopia_energy", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler('ethiopia_energy.log')
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

class PerformanceProfiler:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        import time
        self.timings[operation] = {'start': time.time()}
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        import time
        if operation in self.timings:
            self.timings[operation]['end'] = time.time()
            self.timings[operation]['duration'] = (
                self.timings[operation]['end'] - self.timings[operation]['start']
            )
    
    def log_memory_usage(self, operation: str):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage[operation] = memory_mb
        except ImportError:
            self.memory_usage[operation] = "psutil not available"
    
    def get_performance_report(self) -> Dict[str, any]:
        """Get performance report"""
        return {
            'timings': self.timings,
            'memory_usage': self.memory_usage
        }

def format_large_number(number: Union[int, float], precision: int = 1) -> str:
    """Format large numbers with appropriate units"""
    if number >= 1e9:
        return f"{number/1e9:.{precision}f}B"
    elif number >= 1e6:
        return f"{number/1e6:.{precision}f}M"
    elif number >= 1e3:
        return f"{number/1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    return numerator / denominator if denominator != 0 else default

def create_color_palette(n_colors: int, colormap: str = 'tab10') -> List[str]:
    """Create color palette for visualizations"""
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    cmap = cm.get_cmap(colormap)
    return [colors.to_hex(cmap(i / max(1, n_colors - 1))) for i in range(n_colors)]

def validate_file_upload(uploaded_file, expected_extensions: List[str]) -> bool:
    """Validate uploaded file extension"""
    if uploaded_file is None:
        return False
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    return file_extension in [ext.lower() for ext in expected_extensions]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize DataFrame column names"""
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(r'[^\w\s]', '', regex=True)
                  .str.replace(r'\s+', '_', regex=True)
                  .str.title())
    return df