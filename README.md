# Ethiopia Energy Planning Tool

A comprehensive PyPSA-based power system modeling and expansion planning tool for Ethiopia, utilizing World Bank DRE Atlas data and designed for energy planners and policymakers.

![Ethiopia Energy Planning Tool](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Beta-orange)

## 🌍 Overview

The Ethiopia Energy Planning Tool is designed to support electricity generation and transmission network expansion planning for Ethiopia through 2050. It integrates rich settlement data from the World Bank's Distributed Renewable Energy (DRE) Atlas with power system modeling capabilities to provide comprehensive scenario analysis.

### Key Features

- **Data Integration**: Import and process DRE Atlas data, transmission network shapefiles, and power plant databases
- **Network Modeling**: Build PyPSA networks with customizable clustering methods and voltage levels  
- **Scenario Analysis**: Multi-year expansion planning (2030, 2040, 2050) with renewable energy targets
- **Interactive Visualization**: Streamlit-based interface with maps, charts, and network diagrams
- **Flexible Planning**: Support for different demand growth, technology cost trajectories, and policy constraints

## 📊 Data Sources

### Required Data
1. **DRE Atlas Settlement Data** (CSV): 50,000+ settlements with energy demand estimates
2. **Transmission Network** (Shapefile): Existing transmission lines >45kV
3. **Generator Database** (CSV/Excel): Locations and capacities of power plants

### Settlement Characteristics (from DRE Atlas)
The tool processes over 50 settlement characteristics including:
- Population and building counts
- Energy demand estimates  
- Grid proximity measurements
- Solar potential (PV production)
- Socioeconomic indicators
- Infrastructure access metrics

See the uploaded PDF documentation for complete list of settlement characteristics.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning repository)
- 8GB+ RAM recommended for large datasets

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/ethiopia-energy-planning.git
cd ethiopia-energy-planning
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Optimization Solvers (Optional)

For better optimization performance, install additional solvers:

**CBC (Open Source - Recommended)**
```bash
conda install -c conda-forge cbc
```

**GLPK (Open Source)**
```bash
conda install -c conda-forge glpk
```

**Commercial Solvers** (require licenses):
- Gurobi: `pip install gurobipy`  
- CPLEX: `pip install cplex`

## 📖 Usage

### Running the Application

1. **Start Streamlit App**:
   ```bash
   streamlit run main.py
   ```

2. **Open Browser**: Navigate to `http://localhost:8501`

### Workflow

#### 1. Data Import
- Upload DRE Atlas CSV file
- Upload transmission network shapefile (optional)
- Upload power plant database (optional)
- Process and validate data

#### 2. Network Configuration  
- Select voltage levels to model (400kV, 230kV, 132kV, etc.)
- Choose clustering method:
  - **K-means**: Geographic clustering
  - **Hierarchical**: Agglomerative clustering  
  - **Administrative**: District/region-based
  - **No clustering**: Individual settlements
- Set number of network nodes (10-200)

#### 3. Scenario Planning
Configure planning parameters:
- **Demand Growth**: Annual percentage increase
- **Renewable Targets**: Share by 2030, 2050
- **Cost Trajectories**: Technology cost declines
- **Policy Constraints**: Emission limits, etc.

#### 4. Results Analysis
Explore results through interactive visualizations:
- Generation capacity evolution
- Investment timelines by technology
- Network topology maps  
- Regional energy analysis
- Cost and emission projections

### Example Scenario

```python
# Example scenario configuration
scenario_params = {
    'demand_growth': 5.0,  # 5% annual growth
    'renewable_targets': {2030: 50, 2050: 80},  # 50% by 2030, 80% by 2050
    'cost_declines': {
        'solar': 3.0,    # 3% annual cost decline
        'wind': 2.0,     # 2% annual cost decline  
        'battery': 8.0   # 8% annual cost decline
    }
}
```

## 🗂️ Project Structure

```
ethiopia-energy-planning/
├── main.py                 # Main Streamlit application
├── data_processor.py       # Data import and processing
├── pypsa_builder.py        # PyPSA network construction
├── scenario_engine.py      # Scenario planning and optimization
├── visualization.py        # Charts, maps, and visualizations
├── config.yaml            # Configuration parameters
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Data directory (create manually)
│   ├── dre_atlas/         # DRE Atlas CSV files
│   ├── transmission/      # Transmission shapefiles
│   └── generators/        # Power plant databases
├── outputs/               # Results and exports
└── docs/                  # Additional documentation
```

## 🔧 Configuration

Edit `config.yaml` to customize:

### Technology Parameters
```yaml
technologies:
  solar:
    capital_cost_usd_per_kw: 800
    capacity_factor: 0.25
    cost_decline_rate: 0.03
```

### Planning Settings
```yaml
planning:
  base_year: 2025
  planning_horizon: [2025, 2030, 2040, 2050]
  default_demand_growth: 5.0
```

### Solver Configuration
```yaml
solver:
  default: "cbc"
  timeout_seconds: 3600
```

## 📈 Key Outputs

### Capacity Planning Results
- **Generation Mix Evolution**: Technology shares over time
- **Investment Requirements**: Annual capital needs by technology
- **Transmission Expansion**: New line requirements
- **Cost Projections**: LCOE evolution and total system costs

### Policy Analysis
- **Renewable Energy Targets**: Achievement pathways
- **Emission Trajectories**: CO2 reduction scenarios  
- **Energy Security**: Supply adequacy and reliability metrics
- **Regional Development**: Settlement-level electrification priorities

### Visualizations
- **Interactive Maps**: Network topology, capacity distribution
- **Time Series Charts**: Demand growth, generation mix
- **Comparison Plots**: Multi-scenario analysis
- **Dashboard Summaries**: Key performance indicators

## 🛠️ Advanced Usage

### Custom Data Sources

**Adding New Settlement Characteristics**:
```python
# In data_processor.py
def add_custom_feature(self, df):
    df['custom_metric'] = df['existing_col'] * multiplier
    return df
```

**Integration with External APIs**:
```python
# Weather data integration
def fetch_weather_data(coordinates):
    # Implement API calls for solar/wind resource data
    pass
```

### Network Customization

**Custom Clustering Algorithm**:
```python
# In pypsa_builder.py  
def custom_clustering_method(self, settlements, n_clusters):
    # Implement domain-specific clustering logic
    return cluster_assignments
```

**Technology Integration**:
```python
# Add new technology types
self.technology_costs['geothermal'] = {
    'capital_cost': 3000,
    'marginal_cost': 10,
    'capacity_factor': 0.80
}
```

## 🧪 Testing

Run tests to validate functionality:
```bash
python -m pytest tests/ -v
```

## 📊 Performance Considerations

### Large Datasets
- **Settlement Limit**: Tool handles 50,000+ settlements efficiently
- **Memory Usage**: 8GB+ RAM recommended for full Ethiopia dataset  
- **Processing Time**: Network building: 2-10 minutes depending on clustering
- **Optimization**: Scenario solving: 10-60 minutes per planning year

### Optimization Tips
- Use clustering to reduce network size (50-100 nodes recommended)
- Enable parallel processing for multi-scenario analysis
- Use efficient solvers (CBC/Gurobi) for faster optimization
- Cache processed data to avoid recomputation

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
pip install -e .
pip install -r requirements-dev.txt
pre-commit install
```

### Code Style
- Black for code formatting: `black .`
- Flake8 for linting: `flake8 .`
- Type hints encouraged

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Documentation

### Data Sources
- [World Bank DRE Atlas](https://energydata.info/): Settlement-level energy access data
- [OpenStreetMap](https://www.openstreetmap.org/): Geographic and infrastructure data
- [NASA VIIRS](https://earthdata.nasa.gov/): Nighttime lights data

### Technical References
- [PyPSA Documentation](https://pypsa.readthedocs.io/): Power system modeling framework
- [Streamlit Documentation](https://docs.streamlit.io/): Web application framework
- [Plotly Documentation](https://plotly.com/python/): Interactive visualization library

### Academic References
- [Power System Planning Methods and Applications](https://link.springer.com/)
- [Renewable Energy Integration in Power Systems](https://ieeexplore.ieee.org/)
- [Energy Access and Development in Sub-Saharan Africa](https://worldbank.org/)

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Memory Issues**:
```bash
# Reduce dataset size or increase clustering
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
```

**Solver Errors**:
```bash
# Install CBC solver
conda install -c conda-forge cbc
```

**Coordinate System Issues**:
- Ensure shapefiles use EPSG:4326 (WGS84)
- Check coordinate bounds for Ethiopia (3-15°N, 33-48°E)

### Support

- **Issues**: [GitHub Issues](https://github.com/your-org/ethiopia-energy-planning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ethiopia-energy-planning/discussions)  
- **Email**: [support@ethiopia-energy-planning.org](mailto:support@ethiopia-energy-planning.org)

## 🙏 Acknowledgments

- **World Bank**: For providing comprehensive DRE Atlas data
- **PyPSA Community**: For the excellent power system modeling framework
- **Ethiopian Energy Sector**: For domain expertise and validation
- **Open Source Community**: For the foundational tools and libraries

## 🔮 Roadmap

### Version 2.0 (Planned Features)
- [ ] Real-time weather data integration
- [ ] Machine learning demand forecasting
- [ ] Economic dispatch optimization  
- [ ] Multi-objective optimization (cost, emissions, reliability)
- [ ] Uncertainty analysis and robust optimization
- [ ] Integration with GIS databases
- [ ] Mobile-responsive interface
- [ ] Multi-language support (Amharic, Oromo)

### Long-term Vision
- Regional integration (East Africa power pool)
- Real-time system monitoring integration
- Automated report generation for policymakers
- Integration with national energy databases

---

**Built with ❤️ for sustainable energy planning in Ethiopia**