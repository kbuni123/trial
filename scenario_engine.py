"""
Scenario Planning Engine for Ethiopia Energy Planning Tool

Handles multi-year expansion planning scenarios with different assumptions
"""

import pandas as pd
import numpy as np
import pypsa
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ScenarioParameters:
    """Parameters for a planning scenario"""
    name: str
    demand_growth_rate: float  # Annual % growth
    renewable_targets: Dict[int, float]  # {year: percentage}
    cost_declines: Dict[str, float]  # {technology: annual % decline}
    policy_constraints: Dict[str, any] = None
    co2_price: float = 0  # $/tonne CO2
    discount_rate: float = 0.07
    planning_horizon: List[int] = None

class ScenarioEngine:
    """Main scenario planning engine"""
    
    def __init__(self):
        self.base_network = None
        self.scenarios = {}
        self.results = {}
        self.base_year = 2025
        self.planning_years = [2025, 2030, 2040, 2050]
        
    def run_scenarios(self, base_network, **scenario_params):
        """Run scenario analysis with given parameters"""
        
        self.base_network = copy.deepcopy(base_network)
        
        # Create scenario parameters
        scenario = ScenarioParameters(
            name="Base_Scenario",
            demand_growth_rate=scenario_params.get('demand_growth', 5.0),
            renewable_targets=scenario_params.get('renewable_targets', {2030: 50, 2050: 80}),
            cost_declines=scenario_params.get('cost_declines', {
                'solar': 3.0, 'wind': 2.0, 'battery': 8.0
            }),
            planning_horizon=self.planning_years
        )
        
        # Run expansion planning
        results = self.run_expansion_planning(scenario)
        
        # Store results
        self.scenarios[scenario.name] = scenario
        self.results[scenario.name] = results
        
        return results
    
    def run_expansion_planning(self, scenario: ScenarioParameters):
        """Run multi-year expansion planning"""
        
        results = {
            'capacity_expansion': {},
            'transmission_expansion': {},
            'generation_mix': {},
            'costs': {},
            'emissions': {},
            'investment_timeline': {},
            'supply_security': {}
        }
        
        current_network = copy.deepcopy(self.base_network)
        
        # Run planning for each target year
        for year in scenario.planning_horizon:
            print(f"Planning for year {year}...")
            
            # Update demand projections
            network_year = self.project_demand(current_network, year, scenario.demand_growth_rate)
            
            # Update technology costs
            network_year = self.update_technology_costs(network_year, year, scenario.cost_declines)
            
            # Apply policy constraints
            network_year = self.apply_policy_constraints(network_year, year, scenario)
            
            # Solve expansion planning
            year_results = self.solve_expansion(network_year, year, scenario)
            
            # Store results
            for key in results.keys():
                results[key][year] = year_results.get(key, {})
            
            # Update current network with expansion results
            current_network = year_results.get('expanded_network', current_network)
        
        # Calculate summary metrics
        results['summary'] = self.calculate_summary_metrics(results, scenario)
        
        return results
    
    def project_demand(self, network, target_year, growth_rate):
        """Project demand growth to target year"""
        
        network_proj = copy.deepcopy(network)
        years_elapsed = target_year - self.base_year
        growth_factor = (1 + growth_rate / 100) ** years_elapsed
        
        # Scale all loads
        for load_name in network_proj.loads.index:
            current_profile = network_proj.loads_t.p_set[load_name]
            network_proj.loads_t.p_set[load_name] = current_profile * growth_factor
        
        return network_proj
    
    def update_technology_costs(self, network, target_year, cost_declines):
        """Update technology costs based on learning curves"""
        
        network_updated = copy.deepcopy(network)
        years_elapsed = target_year - self.base_year
        
        # Update generator costs
        for gen_name, gen in network_updated.generators.iterrows():
            carrier = gen.get('carrier', 'Other')
            decline_rate = cost_declines.get(carrier.lower(), 0)
            
            if decline_rate > 0:
                cost_factor = (1 - decline_rate / 100) ** years_elapsed
                network_updated.generators.loc[gen_name, 'capital_cost'] *= cost_factor
        
        # Update storage costs
        if hasattr(network_updated, 'storage_units') and not network_updated.storage_units.empty:
            for storage_name, storage in network_updated.storage_units.iterrows():
                carrier = storage.get('carrier', 'Battery')
                decline_rate = cost_declines.get(carrier.lower(), 0)
                
                if decline_rate > 0:
                    cost_factor = (1 - decline_rate / 100) ** years_elapsed
                    network_updated.storage_units.loc[storage_name, 'capital_cost'] *= cost_factor
        
        return network_updated
    
    def apply_policy_constraints(self, network, target_year, scenario):
        """Apply policy constraints like renewable targets"""
        
        network_constrained = copy.deepcopy(network)
        
        # Apply renewable energy targets
        if target_year in scenario.renewable_targets:
            renewable_target = scenario.renewable_targets[target_year] / 100
            
            # This would implement renewable constraints in PyPSA
            # For now, we'll adjust the renewable capacity limits
            renewable_carriers = ['Solar', 'Wind', 'Hydro']
            
            # Calculate current renewable capacity
            renewable_gens = network_constrained.generators[
                network_constrained.generators.carrier.isin(renewable_carriers)
            ]
            
            if not renewable_gens.empty:
                total_demand = network_constrained.loads_t.p_set.sum().sum()
                target_renewable_capacity = total_demand * renewable_target * 1.2  # Add buffer
                
                # Increase renewable capacity limits if needed
                for gen_name, gen in renewable_gens.iterrows():
                    if gen.get('p_nom_extendable', False):
                        current_max = gen.get('p_nom_max', 0)
                        new_max = max(current_max, target_renewable_capacity / len(renewable_gens))
                        network_constrained.generators.loc[gen_name, 'p_nom_max'] = new_max
        
        return network_constrained
    
    def solve_expansion(self, network, year, scenario):
        """Solve capacity expansion for a given year"""
        
        results = {
            'year': year,
            'expanded_network': network,
            'capacity_expansion': {},
            'transmission_expansion': {},
            'generation_mix': {},
            'costs': {},
            'emissions': {},
            'investment_timeline': {},
            'supply_security': {}
        }
        
        try:
            # Solve the expansion problem
            # Note: This is a simplified version. Real implementation would use
            # more sophisticated optimization with multiple representative periods
            
            # For demonstration, we'll simulate expansion results
            results.update(self.simulate_expansion_results(network, year, scenario))
            
        except Exception as e:
            print(f"Error solving expansion for year {year}: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def simulate_expansion_results(self, network, year, scenario):
        """Simulate expansion results (placeholder for actual optimization)"""
        
        # Calculate demand and supply balance
        total_demand = network.loads_t.p_set.sum().sum()
        existing_capacity = network.generators.p_nom.sum()
        
        # Estimate required new capacity
        years_from_base = year - self.base_year
        demand_growth = (1 + scenario.demand_growth_rate / 100) ** years_from_base
        required_capacity = total_demand * demand_growth * 1.3  # Reserve margin
        
        capacity_gap = max(0, required_capacity - existing_capacity)
        
        # Simulate capacity expansion by technology
        renewable_target = scenario.renewable_targets.get(year, 30) / 100
        
        # Allocate new capacity
        new_solar = capacity_gap * 0.4 * renewable_target / 0.6  # Assume 60% of renewables are solar/wind
        new_wind = capacity_gap * 0.2 * renewable_target / 0.6
        new_gas = capacity_gap * (1 - renewable_target)
        
        capacity_expansion = {
            'Solar': max(0, new_solar),
            'Wind': max(0, new_wind),
            'Gas': max(0, new_gas),
            'Battery': capacity_gap * 0.1  # 10% storage
        }
        
        # Calculate generation mix
        existing_by_carrier = network.generators.groupby('carrier')['p_nom'].sum()
        
        generation_mix = {}
        for carrier in ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal']:
            existing = existing_by_carrier.get(carrier, 0)
            new = capacity_expansion.get(carrier, 0)
            generation_mix[carrier] = existing + new
        
        # Calculate costs
        technology_costs = {
            'Solar': 800 * (0.97 ** years_from_base),  # Cost decline
            'Wind': 1200 * (0.98 ** years_from_base),
            'Gas': 600,
            'Battery': 300 * (0.92 ** years_from_base)
        }
        
        investment_costs = {}
        total_investment = 0
        for tech, capacity in capacity_expansion.items():
            cost_per_kw = technology_costs.get(tech, 1000)
            tech_investment = capacity * 1000 * cost_per_kw  # Convert MW to kW
            investment_costs[tech] = tech_investment
            total_investment += tech_investment
        
        # Calculate emissions (simplified)
        emission_factors = {  # kg CO2/MWh
            'Solar': 0,
            'Wind': 0,
            'Hydro': 0,
            'Gas': 350,
            'Coal': 820
        }
        
        annual_emissions = 0
        for carrier, capacity in generation_mix.items():
            capacity_factor = {'Solar': 0.25, 'Wind': 0.30, 'Hydro': 0.45, 'Gas': 0.50, 'Coal': 0.60}.get(carrier, 0.35)
            annual_generation = capacity * capacity_factor * 8760  # MWh
            annual_emissions += annual_generation * emission_factors.get(carrier, 0)
        
        # Supply security metrics
        renewable_capacity = sum(generation_mix.get(tech, 0) for tech in ['Solar', 'Wind', 'Hydro'])
        total_capacity = sum(generation_mix.values())
        renewable_share = renewable_capacity / total_capacity * 100 if total_capacity > 0 else 0
        
        return {
            'capacity_expansion': capacity_expansion,
            'generation_mix': generation_mix,
            'costs': {
                'total_investment_million_usd': total_investment / 1e6,
                'investment_by_technology': {k: v/1e6 for k, v in investment_costs.items()},
                'lcoe_usd_per_mwh': self.calculate_lcoe(generation_mix, technology_costs)
            },
            'emissions': {
                'annual_co2_tonnes': annual_emissions / 1000,  # Convert to tonnes
                'emissions_intensity_kg_per_mwh': annual_emissions / max(1, sum(
                    cap * 0.35 * 8760 for cap in generation_mix.values()
                ))
            },
            'supply_security': {
                'renewable_share_percent': renewable_share,
                'capacity_margin_percent': (total_capacity - required_capacity) / required_capacity * 100,
                'energy_independence_score': min(100, renewable_share + 20)  # Simplified score
            }
        }
    
    def calculate_lcoe(self, generation_mix, technology_costs):
        """Calculate levelized cost of electricity"""
        
        total_cost = 0
        total_generation = 0
        
        capacity_factors = {
            'Solar': 0.25,
            'Wind': 0.30,
            'Hydro': 0.45,
            'Gas': 0.50,
            'Coal': 0.60
        }
        
        for tech, capacity in generation_mix.items():
            if capacity > 0:
                cf = capacity_factors.get(tech, 0.35)
                annual_gen = capacity * cf * 8760
                
                # Simplified LCOE calculation
                capex_per_mwh = technology_costs.get(tech, 1000) * 1000 / (cf * 8760 * 20)  # 20-year lifetime
                opex_per_mwh = {'Gas': 40, 'Coal': 25}.get(tech, 5)  # $/MWh
                
                tech_cost = annual_gen * (capex_per_mwh + opex_per_mwh)
                total_cost += tech_cost
                total_generation += annual_gen
        
        return total_cost / total_generation if total_generation > 0 else 0
    
    def calculate_summary_metrics(self, results, scenario):
        """Calculate summary metrics across all years"""
        
        summary = {
            'total_investment_billion_usd': 0,
            'cumulative_emissions_million_tonnes': 0,
            'final_renewable_share': 0,
            'average_lcoe': 0,
            'capacity_by_technology': {},
            'investment_timeline': {}
        }
        
        # Sum investments across years
        for year, year_results in results['costs'].items():
            if isinstance(year_results, dict) and 'total_investment_million_usd' in year_results:
                summary['total_investment_billion_usd'] += year_results['total_investment_million_usd'] / 1000
                summary['investment_timeline'][year] = year_results['total_investment_million_usd']
        
        # Calculate cumulative emissions
        for year, year_results in results['emissions'].items():
            if isinstance(year_results, dict) and 'annual_co2_tonnes' in year_results:
                summary['cumulative_emissions_million_tonnes'] += year_results['annual_co2_tonnes'] / 1e6
        
        # Get final year metrics
        final_year = max(scenario.planning_horizon)
        if final_year in results['supply_security']:
            final_results = results['supply_security'][final_year]
            summary['final_renewable_share'] = final_results.get('renewable_share_percent', 0)
        
        # Calculate capacity by technology (final year)
        if final_year in results['generation_mix']:
            summary['capacity_by_technology'] = results['generation_mix'][final_year]
        
        return summary
    
    def compare_scenarios(self, scenario_names=None):
        """Compare multiple scenarios"""
        
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        comparison = {
            'scenarios': scenario_names,
            'metrics': {
                'total_investment': {},
                'renewable_share_2050': {},
                'emissions_2050': {},
                'lcoe_2050': {}
            }
        }
        
        for scenario_name in scenario_names:
            if scenario_name in self.results:
                results = self.results[scenario_name]
                summary = results.get('summary', {})
                
                comparison['metrics']['total_investment'][scenario_name] = \
                    summary.get('total_investment_billion_usd', 0)
                comparison['metrics']['renewable_share_2050'][scenario_name] = \
                    summary.get('final_renewable_share', 0)
        
        return comparison
    
    def create_investment_timeline(self, scenario_name):
        """Create detailed investment timeline for visualization"""
        
        if scenario_name not in self.results:
            return None
        
        results = self.results[scenario_name]
        timeline = []
        
        for year in self.planning_years:
            if year in results['costs']:
                year_costs = results['costs'][year]
                
                if 'investment_by_technology' in year_costs:
                    for tech, investment in year_costs['investment_by_technology'].items():
                        timeline.append({
                            'year': year,
                            'technology': tech,
                            'investment_million_usd': investment
                        })
        
        return pd.DataFrame(timeline)
    
    def create_capacity_evolution(self, scenario_name):
        """Create capacity evolution data for visualization"""
        
        if scenario_name not in self.results:
            return None
        
        results = self.results[scenario_name]
        evolution = []
        
        for year in self.planning_years:
            if year in results['generation_mix']:
                year_mix = results['generation_mix'][year]
                
                for tech, capacity in year_mix.items():
                    evolution.append({
                        'year': year,
                        'technology': tech,
                        'capacity_mw': capacity
                    })
        
        return pd.DataFrame(evolution)
    
    def generate_scenario_report(self, scenario_name):
        """Generate comprehensive scenario report"""
        
        if scenario_name not in self.results:
            return f"Scenario '{scenario_name}' not found"
        
        scenario = self.scenarios[scenario_name]
        results = self.results[scenario_name]
        summary = results.get('summary', {})
        
        report = f"""
        ETHIOPIA ENERGY PLANNING SCENARIO REPORT
        =======================================
        
        Scenario: {scenario_name}
        Planning Horizon: {scenario.planning_horizon}
        
        KEY ASSUMPTIONS:
        - Demand Growth Rate: {scenario.demand_growth_rate}% per year
        - Renewable Targets: {scenario.renewable_targets}
        - Technology Cost Declines: {scenario.cost_declines}
        
        KEY RESULTS:
        - Total Investment: ${summary.get('total_investment_billion_usd', 0):.1f} billion USD
        - Final Renewable Share: {summary.get('final_renewable_share', 0):.1f}%
        - Cumulative Emissions: {summary.get('cumulative_emissions_million_tonnes', 0):.1f} million tonnes CO2
        
        CAPACITY EXPANSION BY TECHNOLOGY (Final Year):
        """
        
        for tech, capacity in summary.get('capacity_by_technology', {}).items():
            report += f"- {tech}: {capacity:.0f} MW\n        "
        
        report += f"""
        
        INVESTMENT TIMELINE:
        """
        
        for year, investment in summary.get('investment_timeline', {}).items():
            report += f"- {year}: ${investment:.0f} million USD\n        "
        
        return report
    
    def export_results(self, scenario_name, format='csv'):
        """Export scenario results to files"""
        
        if scenario_name not in self.results:
            return None
        
        results = self.results[scenario_name]
        
        # Create capacity evolution DataFrame
        capacity_df = self.create_capacity_evolution(scenario_name)
        
        # Create investment timeline DataFrame  
        investment_df = self.create_investment_timeline(scenario_name)
        
        # Create summary DataFrame
        summary_data = []
        for year in self.planning_years:
            year_data = {'year': year}
            
            if year in results['costs']:
                year_data.update({
                    'total_investment_musd': results['costs'][year].get('total_investment_million_usd', 0),
                    'lcoe_usd_mwh': results['costs'][year].get('lcoe_usd_per_mwh', 0)
                })
            
            if year in results['emissions']:
                year_data.update({
                    'annual_emissions_tonnes': results['emissions'][year].get('annual_co2_tonnes', 0),
                    'emissions_intensity': results['emissions'][year].get('emissions_intensity_kg_per_mwh', 0)
                })
            
            if year in results['supply_security']:
                year_data.update({
                    'renewable_share_pct': results['supply_security'][year].get('renewable_share_percent', 0),
                    'capacity_margin_pct': results['supply_security'][year].get('capacity_margin_percent', 0)
                })
            
            summary_data.append(year_data)
        
        summary_df = pd.DataFrame(summary_data)
        
        export_data = {
            'capacity_evolution': capacity_df,
            'investment_timeline': investment_df,
            'yearly_summary': summary_df
        }
        
        return export_data