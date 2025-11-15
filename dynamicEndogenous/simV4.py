"""
SFC-ML: Stock-Flow Consistent Model for Development Policy Analysis

Open-economy SFC model implementing regime switching for comparative institutional analysis.
Calibrated to Turkish economy 2010-2020.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence
from abc import ABC, abstractmethod
import os
from pathlib import Path

# --- Constants ---
TOTAL_YEARS = 15
OUTPUT_DIR = Path("output")

def compose_schedule(*segments: Iterable[float]) -> List[float]:
    """Flattens multiple policy phase arrays into a single timeline."""
    schedule: List[float] = []
    for segment in segments:
        schedule.extend(segment)
    return schedule

def default_policy_schedule(total_years: int = TOTAL_YEARS) -> "PolicySchedule":
    """Creates the baseline 15-year policy schedule for G and Price Effects."""
    train_spending = [30.0] * 3
    housing_spending = [20.0] * 4
    baseline_years = total_years - len(train_spending) - len(housing_spending)
    if baseline_years < 0:
        raise ValueError("Total years shorter than combined project phases.")
    baseline_spending = [0.0] * baseline_years

    initial_price_effect = [0.0] * len(train_spending)
    housing_price_effect = [0.03] * len(housing_spending)
    tail_years = total_years - len(initial_price_effect) - len(housing_price_effect)
    if tail_years < 0:
        raise ValueError("Total years shorter than price reduction phases.")
    tail_price_effect = [0.01] * tail_years

    return PolicySchedule(
        g_project_train=train_spending,
        g_project_housing=housing_spending,
        g_project_baseline=baseline_spending,
        price_reduction_train=initial_price_effect,
        price_reduction_housing=housing_price_effect,
        price_reduction_tail=tail_price_effect,
    )


def boosted_transition_schedule(
    total_years: int = TOTAL_YEARS,
    train_multiplier: float = 1.5,
    housing_multiplier: float = 1.5,
    baseline_spend: float = 10.0,
) -> "PolicySchedule":
    """Transitions toward managed capitalism with stronger public demand.

    Multiplies the megaproject phases and introduces a positive baseline spend
    so that regulated markups still generate enough surplus to meet legacy debt.
    """

    baseline_policy = default_policy_schedule(total_years)

    boosted_train = [val * train_multiplier for val in baseline_policy.g_project_train]
    boosted_housing = [val * housing_multiplier for val in baseline_policy.g_project_housing]

    baseline_years = total_years - len(boosted_train) - len(boosted_housing)
    boosted_baseline = [baseline_spend] * max(baseline_years, 0)

    return PolicySchedule(
        g_project_train=boosted_train,
        g_project_housing=boosted_housing,
        g_project_baseline=boosted_baseline,
        price_reduction_train=baseline_policy.price_reduction_train,
        price_reduction_housing=baseline_policy.price_reduction_housing,
        price_reduction_tail=baseline_policy.price_reduction_tail,
    )

@dataclass(frozen=True)
class ModelParams:
    """Model parameters calibrated to Turkish economy (2010-2020).
    
    Sources: Penn World Tables 10.0, TurkStat, CBRT, World Bank, OECD.
    See turkish_calibration.py for detailed documentation.
    """
    # --- Demand Parameters (Turkish Calibration) ---
    propensity_to_consume: float = 0.85  # OECD: Turkey 2010-2020 avg 0.82-0.88
    tax_rate: float = 0.20  # IMF WEO: Turkey total tax/GDP 18-22%
    
    # --- Production Parameters (Turkish Calibration) ---
    depreciation_rate: float = 0.048  # PWT 10.0: Turkey delta 2010-2019 avg 4.8%
    labor_supply: float = 1000.0  # Normalized scale
    labor_productivity: float = 0.85  # PWT 10.0: Turkey TFP ~85% of US level
    
    # --- Financial Parameters (Turkish Calibration) ---
    nominal_interest_rate: float = 0.031  # CBRT: Real rate 2013-2019 avg 3.1%
    autonomous_investment_rate: float = 0.02  # Minimum replacement investment
    investment_sensitivity_to_pi: float = 0.25  # CBRT data: I-growth/profit correlation ~0.20-0.30
    
    # --- Distribution Parameters (Turkish Calibration) ---
    firm_markup: float = 0.28  # TurkStat I-O 2012: Manufacturing markup 0.25-0.32
    v_share_of_output: float = 0.62  # TurkStat: Wage share (compensation/GDP) 2015-2019 avg
    wage_inflation_sensitivity: float = 1.0  # Full inflation indexation (common in Turkey)
    wage_unemployment_sensitivity: float = 0.5  # Phillips curve estimates
    price_weight_energy_rent: float = 0.25  # Energy/housing weight in CPI
    gov_labor: float = 100.0  # Government employment
    
    # --- Open Economy Parameters (Turkish Calibration) ---
    propensity_to_import: float = 0.23  # World Bank: Turkey M/GDP exc. energy 23-28%
    base_exports: float = 75.0  # Scaled to match 15% exports/GDP (Turkey 14-18%)
    exchange_rate_sensitivity: float = 0.20  # Taylor structuralist models
    export_sensitivity_to_E: float = 0.50  # Gülmez & Yeldan (2014): 0.45-0.52
    import_sensitivity_to_E: float = 0.30  # Trade literature: 0.25-0.35
    
    # --- Socialist Regime Parameters ---
    planned_investment_rate: float = 0.07  # Target for steady-state socialist growth
    
    # --- Infrastructure & Productivity ---
    infrastructure_productivity_effect: float = 0.02  # Mazzucato (2013): 1-3% spillovers
    enable_productivity_feedback: bool = True  # Toggle infrastructure-led productivity feedback
    infrastructure_cost_reduction: float = 0.015  # Cost reduction for workers: transport/energy/housing (1.5% per year when active)
    
    # --- v3.2: Crisis Mechanism Parameters ---
    # Floor constraints (prevent negative/impossible values)
    min_capacity_utilization: float = 0.10       # Minimum output as fraction of capital stock
    subsistence_consumption_share: float = 0.30  # Minimum consumption as fraction of labor force
    min_replacement_investment: float = 1.0      # Investment must at least cover depreciation (multiplier)
    
    # Capital destruction (Marxian devaluation during crises)
    max_capital_destruction_rate: float = 0.15   # Max fraction of K destroyed per year in crisis
    enable_capital_destruction: bool = True      # Toggle Marxian crisis resolution mechanism
    
    # Credit rationing (Minskyan financial fragility)
    credit_crunch_threshold: float = -0.05       # Profit rate below which credit rationing begins
    severe_crisis_threshold: float = -0.10       # Profit rate triggering severe credit freeze
    enable_credit_rationing: bool = True         # Toggle Minskyan credit constraints
    target_profit_rate: float = 0.15             # Desired profit ceiling for regulatory supervisors

    # --- simV4: Capital Flight Controls ---
    global_profit_rate: float = 0.10             # Benchmark global profit rate triggering capital outflow
    capital_flight_sensitivity: float = 0.5      # Speed at which capital flees when domestic profits lag
    capital_controls_effectiveness: float = 0.0  # 0=no controls, 1=perfect controls
    enable_capital_flight: bool = True           # Toggle capital flight dynamics entirely

@dataclass(frozen=True)
class InitialStocks:
    """Dataclass for all initial stock values at t=0."""
    m_domestic: float = 600.0
    b_bond_domestic: float = 700.0
    k_firm: float = 1000.0
    l_firm_loan: float = 500.0
    g_bond_central_bank: float = 50.0
    g_bond_domestic: float = 700.0
    pi_t_minus_1: float = 0.10
    price_index: float = 1.0
    price_index_lag: float = 1.0  # Track t-1 price index for inflation calc
    exchange_rate: float = 1.0
    net_foreign_assets: float = -200.0
    # New stock needed for dynamic wage calculation
    nominal_wage_t_minus_1: float = 3.0 # Initial nominal wage per worker
    labor_productivity: float = 0.8      # Track time-varying productivity for feedback mechanisms
    cost_of_living_index: float = 1.0   # Track worker reproduction costs (1.0 = baseline)

    def as_dict(self) -> Dict[str, float]:
        """Converts the dataclass to a dictionary for the model's state."""
        return {
            "MDomestic": self.m_domestic,
            "BBondDomestic": self.b_bond_domestic,
            "KFirm": self.k_firm,
            "LFirmLoan": self.l_firm_loan, 
            "GBondCentralBank": self.g_bond_central_bank,
            "GBondDomestic": self.g_bond_domestic,
            "PiTMinus1": self.pi_t_minus_1,
            "priceIndex": self.price_index,
            "priceIndexLag": self.price_index_lag,
            "exchangeRate": self.exchange_rate,
            "netForeignAssets": self.net_foreign_assets,
            "NominalWageTMinus1": self.nominal_wage_t_minus_1, # NEW
            "LaborProductivity": self.labor_productivity,
            "CostOfLivingIndex": self.cost_of_living_index,
        }

@dataclass(frozen=True)
class PolicySchedule:
    """Policy timelines for government spending and price controls."""
    g_project_train: Sequence[float]
    g_project_housing: Sequence[float]
    g_project_baseline: Sequence[float]
    price_reduction_train: Sequence[float]
    price_reduction_housing: Sequence[float]
    price_reduction_tail: Sequence[float]
    rent_weight_in_price_index: float = 0.15
    
    _g_project: List[float] = field(init=False, repr=False)
    _price_reduction: List[float] = field(init=False, repr=False)
    train_years: int = field(init=False, repr=False)
    housing_years: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        g_project = compose_schedule(self.g_project_train, self.g_project_housing, self.g_project_baseline)
        price_reduction = compose_schedule(self.price_reduction_train, self.price_reduction_housing, self.price_reduction_tail)
        if len(g_project) != len(price_reduction):
            raise ValueError("Policy schedules must share the same length.")
        
        object.__setattr__(self, "_g_project", g_project)
        object.__setattr__(self, "_price_reduction", price_reduction)
        object.__setattr__(self, "train_years", len(self.g_project_train))
        object.__setattr__(self, "housing_years", len(self.g_project_housing))
        
    @property
    def total_years(self) -> int: return len(self._g_project)
    def spending_for_year(self, year_index: int) -> float: return self._g_project[year_index]
    def price_reduction_for_year(self, year_index: int) -> float: return self._price_reduction[year_index]


class InvestmentFinanceRegime(ABC):
    """Abstract base class for investment and finance regimes."""
    @abstractmethod
    def calculate_flows(
        self, params: ModelParams, stocks: Dict[str, float]
    ) -> (float, float, float):
        pass

class ExternalRegime(ABC):
    """Abstract base class for external sector regimes."""
    @abstractmethod
    def calculate_trade_flows(
        self, params: ModelParams, stocks: Dict[str, float]
    ) -> (float, float):
        pass
    
    @abstractmethod
    def update_exchange_rate(
        self, params: ModelParams, stocks: Dict[str, float], current_account_t: float
    ) -> float:
        pass

# --- NEW: Distribution Regime Interface ---

class DistributionRegime(ABC):
    """
    Abstract contract for all Price & Wage setting regimes.
    Replaces the old fixed v_share assumption.
    """
    @abstractmethod
    def determine_prices_and_wages(
        self,
        params: ModelParams,
        stocks: Dict[str, float],
        y_output_t: float,
        price_reduction_policy: float # Policy shield effect
    ) -> (float, float):
        """
        Calculates the new price index and the total wage bill (W).
        Returns:
            (new_price_index, nominal_wage_bill_W_t)
        """
        pass


# ---
# --- CONCRETE REGIME IMPLEMENTATIONS (THE "LEGO BRICKS") ---
# ---

class ProfitLedPrivateBanking(InvestmentFinanceRegime):
    """
    Regime: Investment is profit-led; financed by private banks.
    
    v3.2 Enhancement: Incorporates Minskyan credit rationing during financial distress.
    Banks reduce loan supply when firm profitability collapses, deepening crises
    but preventing zombie lending that would delay capital restructuring.
    """
    def calculate_flows(self, params: ModelParams, stocks: Dict[str, float]) -> (float, float, float):
        pi_t_minus_1 = stocks["PiTMinus1"]
        k_t_minus_1 = stocks["KFirm"]
        l_firm_loan_t_minus_1 = stocks["LFirmLoan"]
        
        # Profit-led investment function (Kaleckian accelerator)
        desired_investment = (
            params.autonomous_investment_rate
            + params.investment_sensitivity_to_pi * pi_t_minus_1
        ) * k_t_minus_1
        
        # v3.2: Floor investment at depreciation (capital must be replaced)
        # Economic rationale: Even in depression, worn-out machines need replacement
        depreciation_floor = params.depreciation_rate * k_t_minus_1 * params.min_replacement_investment
        desired_investment = max(desired_investment, depreciation_floor)
        
        # Interest payments on existing debt
        interest_payments_t = params.nominal_interest_rate * l_firm_loan_t_minus_1
        
        # v3.2: Minskyan credit rationing mechanism
        # Theory: Banks restrict credit when borrower solvency deteriorates
        # "Hedge → Speculative → Ponzi" transition triggers credit freeze
        if params.enable_credit_rationing:
            if pi_t_minus_1 < params.severe_crisis_threshold:
                credit_availability = 0.2
            elif pi_t_minus_1 < params.credit_crunch_threshold:
                credit_availability = 0.5
            else:
                credit_availability = 1.0
        else:
            credit_availability = 1.0

        if pi_t_minus_1 > params.target_profit_rate:
            profit_ratio = params.target_profit_rate / max(pi_t_minus_1, 1e-9)
            credit_availability = min(credit_availability, profit_ratio)

        credit_availability = np.clip(credit_availability, 0.0, 1.0)

        realized_investment = desired_investment * credit_availability
        realized_investment = max(realized_investment, depreciation_floor)
        loan_demand_t = realized_investment
        
        return (realized_investment, loan_demand_t, interest_payments_t)

class StateLedNationalizedBanking(InvestmentFinanceRegime):
    """Regime: Investment is state-planned; financed by public credit."""
    def calculate_flows(self, params: ModelParams, stocks: Dict[str, float]) -> (float, float, float):
        k_t_minus_1 = stocks["KFirm"]
        investment_t = params.planned_investment_rate * k_t_minus_1
        interest_payments_t = 0.0 # Interest is zero or recycled to state
        loan_demand_t = 0.0 # New loans handled by public finance/MB
        return (investment_t, loan_demand_t, interest_payments_t)

class FloatingExchangeRate(ExternalRegime):
    """Regime: Flexible exchange rate driven by trade balance."""
    def calculate_trade_flows(self, params: ModelParams, stocks: Dict[str, float]) -> (float, float):
        e_t_minus_1 = stocks["exchangeRate"]
        exports_t = params.base_exports * (e_t_minus_1 ** params.export_sensitivity_to_E)
        propensity_m = params.propensity_to_import * (e_t_minus_1 ** -params.import_sensitivity_to_E)
        return (exports_t, propensity_m)

    def update_exchange_rate(self, params: ModelParams, stocks: Dict[str, float], current_account_t: float) -> float:
        # Exchange rate adjustment to prevent explosive dynamics:
        # Instead of absolute delta (which can jump E from 0.5 to 20+ in one step),
        # we use a percentage-change rule based on deficit relative to GDP.
        # This keeps depreciation proportional to economic size and bounds volatility.
        e_t_minus_1 = stocks["exchangeRate"]
        deficit = -current_account_t
        
        # Get GDP from stocks (most recent output; fallback to 100 if missing)
        gdp = stocks.get("YOutput", 100.0)
        
        # Deficit as share of GDP (guards against division by zero)
        deficit_ratio = deficit / max(abs(gdp), 1e-9)
        
        # Percentage depreciation/appreciation (capped at ±20% per year)
        depreciation_rate = params.exchange_rate_sensitivity * deficit_ratio
        depreciation_rate = np.clip(depreciation_rate, -0.2, 0.2)
        
        # Apply percentage change: E_new = E_old × (1 + rate)
        new_exchange_rate = e_t_minus_1 * (1.0 + depreciation_rate)
        
        # Bound E between 0.3 and 3.0 to prevent runaway dynamics
        return max(0.3, min(new_exchange_rate, 3.0))

class ManagedExchangeRate(ExternalRegime):
    """Regime: Capital controls. Exchange rate is fixed."""
    def calculate_trade_flows(self, params: ModelParams, stocks: Dict[str, float]) -> (float, float):
        exports_t = params.base_exports
        propensity_m = params.propensity_to_import
        return (exports_t, propensity_m)

    def update_exchange_rate(self, params: ModelParams, stocks: Dict[str, float], current_account_t: float) -> float:
        return stocks["exchangeRate"] 

# --- DISTRIBUTION REGIMES (NEW) ---

class FixedShareRegime(DistributionRegime):
    """
    The 'Classical' v3.0 regime.
    - Distribution (v_share) is a fixed parameter (outcome of class truce).
    - Prices are set by applying the policy shield to last year's price.
    """
    def determine_prices_and_wages(
        self,
        params: ModelParams,
        stocks: Dict[str, float],
        y_output_t: float,
        price_reduction_policy: float
    ) -> (float, float):
        
        # 1. Calculate Wage Bill (W)
        # W is fixed share of Y
        w_t = y_output_t * params.v_share_of_output
        
        # 2. Calculate Price Index (P)
        # P is just last year's price, adjusted by our *policy* shield
        current_price_index = stocks["priceIndex"]
        new_price_index = current_price_index * (
            1.0 - price_reduction_policy * params.price_weight_energy_rent
        )
        
        return (new_price_index, w_t)

class ConflictInflationRegime(DistributionRegime):
    """
    The Post-Keynesian (Kaleckian) v3.1 regime.
    - Prices (P) are set by firm markup over unit labor costs.
    - Wages (W) are set by workers' bargaining (inflation compensation).
    """
    def determine_prices_and_wages(
        self,
        params: ModelParams,
        stocks: Dict[str, float],
        y_output_t: float,
        price_reduction_policy: float
    ) -> (float, float):
        
        # --- 1. Get Lagged Inputs ---
        last_price_index = stocks["priceIndex"]
        prev_price_index = stocks.get("priceIndexLag", last_price_index)
        last_nominal_wage_per_worker = stocks["NominalWageTMinus1"]
        labor_productivity = stocks.get("LaborProductivity", params.labor_productivity)
        
        # --- 2. Worker Bargaining (Wage Setting) ---

        # Period-to-period inflation; guard against division by zero
        inflation = (last_price_index / max(prev_price_index, 1e-9)) - 1.0
        
        # Labor demand is simplified as YOutput / Productivity
        labor_demand_L_t = y_output_t / max(labor_productivity, 1e-9)
        unemployment_rate = max(0.01, 1.0 - (labor_demand_L_t / params.labor_supply)) # Floor at 1%
        
        # The Wage Rule (Target Compensation + Phillips Curve)
        wage_growth = (
            (inflation * params.wage_inflation_sensitivity) - 
            (unemployment_rate * params.wage_unemployment_sensitivity)
        )

        wage_growth = np.clip(wage_growth, -0.5, 0.5)
        
        new_nominal_wage_per_worker = last_nominal_wage_per_worker * (1.0 + wage_growth)
        
        # Total Nominal Wage Bill (W)
        w_t = new_nominal_wage_per_worker * labor_demand_L_t
        
        # --- 3. Firm Pricing (Price Setting) ---
        denom = y_output_t if abs(y_output_t) >= 1e-9 else (1e-9 if y_output_t >= 0 else -1e-9)
        unit_labor_cost = w_t / denom
        
        # Apply socialist policy as a *reduction* on the effective markup
        effective_markup = params.firm_markup * (
            1.0 - price_reduction_policy * params.price_weight_energy_rent
        )
        
        # The Price Rule (Kaleckian Markup: P = (1 + markup) * ULC)
        new_price_index = (1.0 + effective_markup) * unit_labor_cost
        
        return (new_price_index, w_t)


# ---
# --- THE HYBRID v3.1 MODEL ENGINE (THE MANAGER) ---
# ---

class SFCModel:
    """
    Manages the economic state and runs the simulation step-by-step.
    Delegates all complex logic to the injected regime objects.
    """
    def __init__(
        self, 
        stocks: InitialStocks, 
        params: ModelParams, 
        policy: PolicySchedule,
        # We INJECT the regime objects (Dependency Injection)
        finance_regime: InvestmentFinanceRegime,
        external_regime: ExternalRegime,
        distribution_regime: DistributionRegime # <--- NEW
    ):
        self.params = params
        self.policy = policy
        self._initial_stocks = stocks.as_dict()
        self.stocks: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []
        
    # Strategy object handling investment and financing behavior
        self.finance_regime = finance_regime
    # Strategy object governing trade flows and exchange-rate adjustments
        self.external_regime = external_regime
    # Strategy object determining price and wage dynamics
        self.distribution_regime = distribution_regime # <--- NEW
        
        self.reset()

    @classmethod
    def from_defaults(cls, regime_type: str = "conflict_social_democracy") -> "SFCModel":
        """
        Factory method to create a model with default settings
        for a specific regime.
        """
        if regime_type == "fixed_social_democracy": # Original v2.4 logic
            finance_reg = ProfitLedPrivateBanking()
            external_reg = FloatingExchangeRate()
            distrib_reg = FixedShareRegime()
        elif regime_type == "conflict_social_democracy": # NEW: PK Distribution
            finance_reg = ProfitLedPrivateBanking()
            external_reg = FloatingExchangeRate()
            distrib_reg = ConflictInflationRegime() # <-- Injecting the new regime
        elif regime_type == "socialist_plan":
            finance_reg = StateLedNationalizedBanking()
            external_reg = ManagedExchangeRate()
            distrib_reg = ConflictInflationRegime()
        else:
            raise ValueError(f"Unknown regime_type: {regime_type}")
            
        return cls(
            stocks=InitialStocks(),
            params=ModelParams(),
            policy=default_policy_schedule(),
            finance_regime=finance_reg,
            external_regime=external_reg,
            distribution_regime=distrib_reg # <--- NEW
        )

    def reset(self) -> None:
        """Resets the model to its initial stocks and clears history."""
        self.stocks = self._initial_stocks.copy()
        self.history = []

    def step(self, year: int) -> Dict[str, float]:
        """
        The "manager" step function. It delegates all complex
        logic to the regime objects.
        """
        year_index = year - 1
        if year_index < 0 or year_index >= self.policy.total_years:
            raise ValueError("Year index out of range for policy schedule.")

        # --- 1. Get LAGGED Stocks & Policy Inputs ---
        g_project_t = self.policy.spending_for_year(year_index)
        price_reduction_t = self.policy.price_reduction_for_year(year_index)
        labor_productivity = self.stocks.get("LaborProductivity", self.params.labor_productivity)
        cost_of_living = self.stocks.get("CostOfLivingIndex", 1.0)

        # Infrastructure effects: productivity + cost reduction
        if self.params.enable_productivity_feedback and g_project_t > 0:
            labor_productivity *= (1.0 + self.params.infrastructure_productivity_effect)
            self.stocks["LaborProductivity"] = labor_productivity
            
            # Cost reduction: public transport, energy, housing reduce worker expenses
            cost_of_living *= (1.0 - self.params.infrastructure_cost_reduction)
            self.stocks["CostOfLivingIndex"] = cost_of_living
        
        # --- 2. Delegate Investment & Finance (Autonomous Demand) ---
        (investment_t, loan_demand_t, interest_payments_t) = \
            self.finance_regime.calculate_flows(self.params, self.stocks)
            
        # --- 3. Delegate External Sector (Autonomous Demand) ---
        (autonomous_exports_t, propensity_to_import_m) = \
            self.external_regime.calculate_trade_flows(self.params, self.stocks)
        
        # --- 4. Solve for Endogenous Output (Y) ---
        mpc = self.params.propensity_to_consume
        tax = self.params.tax_rate
        v_share = self.params.v_share_of_output
        marginal_spend_propensity = mpc * (1 - tax) * v_share
        
        multiplier = 1.0 / (1.0 - marginal_spend_propensity + propensity_to_import_m)
        
        interest_domestic_on_bonds = self.params.nominal_interest_rate * self.stocks["BBondDomestic"]
        c_autonomous = mpc * (1 - tax) * interest_domestic_on_bonds
        
        autonomous_demand = g_project_t + investment_t + c_autonomous + autonomous_exports_t

        y_output_t = autonomous_demand * multiplier
        
        # --- v3.2: Apply output floor to prevent negative production ---
        # Economic rationale: Even Great Depression saw only ~30% GDP decline, not negative output.
        # Floor set at minimum capacity utilization (10% of capital stock by default).
        k_t_minus_1 = self.stocks["KFirm"]
        min_output = self.params.min_capacity_utilization * k_t_minus_1
        y_output_t = max(y_output_t, min_output)
        
        # --- 5. Delegate Price & Wage Determination (The Conflict Engine) ---

        (new_price_index, w_t) = self.distribution_regime.determine_prices_and_wages(
            self.params, self.stocks, y_output_t, price_reduction_t
        )
        
        # --- 6. Calculate All Other Flows (now generic!) ---
        y_domestic = w_t + interest_domestic_on_bonds
        tax_t = y_domestic * self.params.tax_rate
        consumption_t = y_domestic * self.params.propensity_to_consume
        
        # --- v3.2: Apply consumption floor (subsistence constraint) ---
        # Economic rationale: Workers must consume minimum to survive/reproduce labor power.
        # Even unemployed workers consume via transfers, savings, or informal economy.
        min_consumption = self.params.subsistence_consumption_share * self.params.labor_supply
        consumption_t = max(consumption_t, min_consumption)
        
        # Marxian Dynamics are now generic.
        surplus_value_t = y_output_t - w_t - interest_payments_t
        
        k_t_minus_1 = self.stocks["KFirm"]
        pi_t = surplus_value_t / (k_t_minus_1 + w_t)
        vcc_t = k_t_minus_1 / w_t if w_t else 0.0
        depreciation_t = k_t_minus_1 * self.params.depreciation_rate

        # --- 7. Update Stocks ---
        self.stocks["GBondCentralBank"] += g_project_t
        savings_t = y_domestic - tax_t - consumption_t
        self.stocks["MDomestic"] += savings_t + loan_demand_t

        self.stocks["LFirmLoan"] += loan_demand_t
        
        # --- v3.2: Marxian capital destruction mechanism ---
        # Theory: Marx Volume III - crises resolve via devaluation of unproductive capital.
        # When surplus value is negative, capital is destroyed through:
        # (1) Bankruptcies and liquidations
        # (2) Physical scrapping of obsolete equipment
        # (3) Write-downs of asset values
        # This purges excess capacity and restores profitability for surviving capital.
        capital_change = investment_t - depreciation_t
        
        if self.params.enable_capital_destruction and surplus_value_t < 0:
            # Destruction rate proportional to losses, capped at max_rate
            destruction_amount = min(
                self.params.max_capital_destruction_rate * k_t_minus_1,
                abs(surplus_value_t)
            )
            capital_change -= destruction_amount
            
            # Optional: Corresponding debt write-off (bankruptcy forgiveness)
            # Prevents zombie firms from accumulating unpayable debt
            debt_writeoff = 0.5 * destruction_amount
            self.stocks["LFirmLoan"] = max(0.0, self.stocks["LFirmLoan"] - debt_writeoff)
        
        self.stocks["KFirm"] += capital_change
        
        # Store the new, ENDOGENOUS price and nominal wage for next period
        self.stocks["priceIndexLag"] = self.stocks.get("priceIndex", new_price_index)
        self.stocks["priceIndex"] = new_price_index
        
        # Store current output for exchange rate calculations
        self.stocks["YOutput"] = y_output_t

        # Calculate nominal wage per worker from wage bill and labor demand
        current_productivity = self.stocks.get("LaborProductivity", self.params.labor_productivity)
        labor_demand_L_t = y_output_t / max(current_productivity, 1e-9)
        if labor_demand_L_t > 0:
            new_nominal_wage_per_worker = w_t / labor_demand_L_t
        else:
            new_nominal_wage_per_worker = self.stocks["NominalWageTMinus1"]  # Keep previous wage if no employment
        
        self.stocks["NominalWageTMinus1"] = new_nominal_wage_per_worker # NEW
        
        # --- 8. Delegate Exchange Rate Update ---
        imports_t = propensity_to_import_m * y_output_t
        trade_balance_t = autonomous_exports_t - imports_t
        current_account_t = trade_balance_t
        
        capital_flight_t = 0.0
        if self.params.enable_capital_flight:
            profit_gap = self.params.global_profit_rate - pi_t
            if profit_gap > 0:
                control_factor = 1.0 - np.clip(self.params.capital_controls_effectiveness, 0.0, 1.0)
                capital_base = max(self.stocks.get("KFirm", k_t_minus_1), 0.0)
                capital_flight_t = control_factor * self.params.capital_flight_sensitivity * profit_gap * capital_base
                max_outflow = max(self.stocks.get("MDomestic", 0.0), 0.0)
                capital_flight_t = min(capital_flight_t, max_outflow)
                self.stocks["MDomestic"] = max(0.0, self.stocks.get("MDomestic", 0.0) - capital_flight_t)

        self.stocks["netForeignAssets"] += current_account_t - capital_flight_t
        
        # The regime (Floating or Managed) decides how E moves.
        self.stocks["exchangeRate"] = \
            self.external_regime.update_exchange_rate(
                self.params, self.stocks, current_account_t
            )
            
        # Update lagged variables for the next step
        self.stocks["PiTMinus1"] = pi_t
        
        # --- 9. Log Results ---
        price_denom = new_price_index if abs(new_price_index) >= 1e-9 else (1e-9 if new_price_index >= 0 else -1e-9)

        results = {
            "year": year,
            "YOutput": y_output_t,
            "PiT": pi_t,
            "VCC": vcc_t,
            "GBondCentralBank": self.stocks["GBondCentralBank"],
            "realWageIndex": new_nominal_wage_per_worker / price_denom,
            "LFirmLoan": self.stocks["LFirmLoan"],
            "exchangeRate": self.stocks["exchangeRate"],
            "netForeignAssets": self.stocks["netForeignAssets"],
            "tradeBalance": trade_balance_t,
            "capitalFlight": capital_flight_t,
            "LaborProductivity": self.stocks.get("LaborProductivity", self.params.labor_productivity),
            "NominalWage": new_nominal_wage_per_worker,
            "CostOfLiving": self.stocks.get("CostOfLivingIndex", 1.0),
        }
        self.history.append(results)
        return results

    # --- run() and summary() methods (Unchanged) ---
    
    def run(self, years: Optional[int] = None) -> pd.DataFrame:
        target_years = years or self.policy.total_years
        if target_years > self.policy.total_years:
            raise ValueError("Requested years exceed provided policy schedule.")
        self.reset()
        for year in range(1, target_years + 1):
            self.step(year)
        df = pd.DataFrame(self.history)
        
        # Attach parameter metadata to the dataframe for traceability
        df.attrs['params'] = {
            'propensity_to_consume': self.params.propensity_to_consume,
            'tax_rate': self.params.tax_rate,
            'firm_markup': self.params.firm_markup,
            'wage_inflation_sensitivity': self.params.wage_inflation_sensitivity,
            'wage_unemployment_sensitivity': self.params.wage_unemployment_sensitivity,
            'investment_sensitivity_to_pi': self.params.investment_sensitivity_to_pi,
            'nominal_interest_rate': self.params.nominal_interest_rate,
            'target_profit_rate': self.params.target_profit_rate,
            'global_profit_rate': self.params.global_profit_rate,
            'capital_flight_sensitivity': self.params.capital_flight_sensitivity,
            'capital_controls_effectiveness': self.params.capital_controls_effectiveness,
            'infrastructure_productivity_effect': self.params.infrastructure_productivity_effect,
            'enable_productivity_feedback': self.params.enable_productivity_feedback,
            'enable_capital_flight': self.params.enable_capital_flight,
            'enable_credit_rationing': self.params.enable_credit_rationing,
            'enable_capital_destruction': self.params.enable_capital_destruction,
        }
        df.attrs['regimes'] = {
            'finance': self.finance_regime.__class__.__name__,
            'external': self.external_regime.__class__.__name__,
            'distribution': self.distribution_regime.__class__.__name__,
        }
        df.attrs['initial_stocks'] = self._initial_stocks.copy()
        
        return df

    def summary(self, preview: Optional[int] = None) -> None:
        if not self.history:
            print("Run the simulation before requesting a summary.")
            return
        results_df = pd.DataFrame(self.history)
        
        # Print parameter configuration
        print("\n=== PARAMETER CONFIGURATION ===")
        print(f"Finance Regime: {self.finance_regime.__class__.__name__}")
        print(f"External Regime: {self.external_regime.__class__.__name__}")
        print(f"Distribution Regime: {self.distribution_regime.__class__.__name__}")
        print(f"\nKey Parameters:")
        print(f"  firm_markup = {self.params.firm_markup}")
        print(f"  wage_inflation_sensitivity = {self.params.wage_inflation_sensitivity}")
        print(f"  wage_unemployment_sensitivity = {self.params.wage_unemployment_sensitivity}")
        print(f"  target_profit_rate = {self.params.target_profit_rate}")
        print(f"  nominal_interest_rate = {self.params.nominal_interest_rate}")
        print(f"  capital_controls_effectiveness = {self.params.capital_controls_effectiveness}")
        print(f"  enable_productivity_feedback = {self.params.enable_productivity_feedback}")
        print(f"  infrastructure_cost_reduction = {self.params.infrastructure_cost_reduction}")
        print(f"  enable_capital_flight = {self.params.enable_capital_flight}")
        print(f"  enable_credit_rationing = {self.params.enable_credit_rationing}")
        
        print("\n=== SIMULATION RESULTS ===")
        columns = [
            "year", "YOutput", "PiT", "realWageIndex", 
            "LFirmLoan", "exchangeRate", "netForeignAssets", "capitalFlight", "LaborProductivity", "NominalWage"
        ]
        if 'CostOfLiving' in results_df.columns:
            columns.append("CostOfLiving")
        table = results_df[columns]
        if preview is not None:
            table = table.head(preview)
        print(table.to_string(index=False))
        
        # Cost reduction summary
        if 'CostOfLiving' in results_df.columns:
            initial_col = results_df['CostOfLiving'].iloc[0]
            final_col = results_df['CostOfLiving'].iloc[-1]
            col_reduction = (1 - final_col / initial_col) * 100
            print(f"\n=== COST REDUCTION SUMMARY ===")
            print(f"Initial Cost of Living Index: {initial_col:.4f}")
            print(f"Final Cost of Living Index: {final_col:.4f}")
            print(f"Total Cost Reduction: {col_reduction:.1f}%")


# --- Plotting Function (No change needed) ---
def plot_simulation_results(
    results_df: pd.DataFrame,
    model: "SFCModel",
    scenario_name: Optional[str] = None,
) -> None:
    """Generates a dashboard of key time-series metrics for the simulation."""

    if results_df.empty:
        print("No results to plot. Run the simulation first.")
        return

    # Define which variables to visualise and their subplot titles
    series_to_plot = [
        ("YOutput", "Output (Y)"),
        ("PiT", "Rate of Profit (Π)"),
        ("realWageIndex", "Real Wage Index"),
        ("NominalWage", "Nominal Wage"),
        ("LFirmLoan", "Firm Loans"),
        ("exchangeRate", "Exchange Rate"),
        ("capitalFlight", "Capital Flight"),
        ("LaborProductivity", "Labor Productivity"),
    ]

    rows = 4
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    years = results_df["year"] if "year" in results_df else range(1, len(results_df) + 1)

    for ax, (column, title) in zip(axes, series_to_plot):
        if column not in results_df:
            ax.set_visible(False)
            continue
        ax.plot(years, results_df[column], marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.grid(True, linewidth=0.4, alpha=0.6)

    title = scenario_name or "Simulation Dashboard"
    fig.suptitle(title, fontsize=16)

    finance_name = model.finance_regime.__class__.__name__
    external_name = model.external_regime.__class__.__name__
    distribution_name = model.distribution_regime.__class__.__name__

    params = model.params
    param_summary = (
        f"c={params.propensity_to_consume:.2f}, tau={params.tax_rate:.2f}, "
        f"markup={params.firm_markup:.2f}, wage_sens={params.wage_inflation_sensitivity:.2f}"
    )

    train_spend = list(model.policy.g_project_train)
    housing_spend = list(model.policy.g_project_housing)
    train_summary = (
        f"train {model.policy.train_years}y@{np.mean(train_spend):.0f}"
        if train_spend
        else "train 0y"
    )
    housing_summary = (
        f"housing {model.policy.housing_years}y@{np.mean(housing_spend):.0f}"
        if housing_spend
        else "housing 0y"
    )
    policy_summary = f"G policy: {train_summary}, {housing_summary}"

    regime_summary = (
        f"Finance: {finance_name} | External: {external_name} | Distribution: {distribution_name}"
    )

    fig.text(0.5, 0.08, regime_summary, ha="center", fontsize=12)
    fig.text(0.5, 0.05, param_summary, ha="center", fontsize=12)
    fig.text(0.5, 0.02, policy_summary, ha="center", fontsize=12)

    plt.tight_layout(rect=(0.0, 0.18, 1.0, 0.94))
    plt.show()


# --- Main execution ---
def main() -> None:
    """
    Entry point for the simulation.
    This is now the "control room" where you define the economy.
    """
    
    # ========================================================================
    # CENTRALIZED PARAMETER CONFIGURATIONS
    # ========================================================================
    
    # --- Configuration 1: Baseline Parameters ---
    PARAMS_BASELINE = ModelParams(
        propensity_to_consume=0.85,
        tax_rate=0.20,
        depreciation_rate=0.05,
        nominal_interest_rate=0.03,
        autonomous_investment_rate=0.02,
        investment_sensitivity_to_pi=0.5,
        v_share_of_output=0.65,
        firm_markup=0.25,
        wage_inflation_sensitivity=1.0,
        wage_unemployment_sensitivity=0.5,
        labor_productivity=0.8,
        propensity_to_import=0.15,
        base_exports=75.0,
        target_profit_rate=0.15,
        global_profit_rate=0.10,
        capital_flight_sensitivity=0.5,
        capital_controls_effectiveness=0.0,
        enable_productivity_feedback=True,
        enable_capital_flight=True,
        enable_credit_rationing=True,
        enable_capital_destruction=True,
    )
    
    # --- Configuration 2: Managed Conflict (Low Markup) ---
    PARAMS_MANAGED = replace(
        PARAMS_BASELINE,
        firm_markup=0.10,
        wage_inflation_sensitivity=0.7,
        wage_unemployment_sensitivity=0.1,
    )
    
    # --- Configuration 3: Stabilized Transition (Public Backstop) ---
    PARAMS_TRANSITION = replace(
        PARAMS_MANAGED,
        firm_markup=0.26,
        nominal_interest_rate=0.006,
        autonomous_investment_rate=0.05,
        wage_inflation_sensitivity=0.2,
        wage_unemployment_sensitivity=0.3,
        labor_productivity=1.15,
    )
    
    # --- Configuration 4: Coordinated Socialist Planning ---
    PARAMS_PLANNED = replace(
        PARAMS_TRANSITION,
        planned_investment_rate=0.09,
        enable_credit_rationing=False,
        enable_capital_flight=False,
        capital_controls_effectiveness=1.0,
        enable_productivity_feedback=True,
        infrastructure_productivity_effect=0.03,
        infrastructure_cost_reduction=0.02,  # 2% annual cost reduction when infrastructure active
        nominal_interest_rate=0.0,
    )
    
    # --- Configuration 5: Partial Planning (Failed) ---
    PARAMS_PARTIAL = replace(
        PARAMS_TRANSITION,
        enable_capital_flight=True,
        capital_controls_effectiveness=0.0,
        enable_productivity_feedback=False,
        infrastructure_productivity_effect=0.0,
    )
    
    # ========================================================================
    # POLICY SCHEDULES
    # ========================================================================
    
    POLICY_DEFAULT = default_policy_schedule()
    POLICY_BOOSTED = boosted_transition_schedule(
        train_multiplier=3.5,
        housing_multiplier=2.5,
        baseline_spend=140.0,
    )
    
    # ========================================================================
    # INITIAL STOCKS CONFIGURATIONS
    # ========================================================================
    
    STOCKS_DEFAULT = InitialStocks()
    STOCKS_NO_DEBT = InitialStocks(l_firm_loan=0.0)
    
    # ========================================================================
    # SCENARIO DEFINITIONS
    # ========================================================================
    
    scenarios = {
        'scenario_1': {
            'name': 'Social Democracy (Fixed Share/No Conflict)',
            'description': 'Profit-led growth with a fixed wage share; prices only move via policy shield.',
            'params': PARAMS_BASELINE,
            'stocks': STOCKS_DEFAULT,
            'policy': POLICY_DEFAULT,
            'finance_regime': ProfitLedPrivateBanking(),
            'external_regime': FloatingExchangeRate(),
            'distribution_regime': FixedShareRegime(),
        },
        'scenario_2': {
            'name': 'PK Conflict (Default Barbarian Parameters)',
            'description': 'Profit-led investment; conflict inflation with HIGH sensitivity.',
            'params': PARAMS_BASELINE,
            'stocks': STOCKS_DEFAULT,
            'policy': POLICY_DEFAULT,
            'finance_regime': ProfitLedPrivateBanking(),
            'external_regime': FloatingExchangeRate(),
            'distribution_regime': ConflictInflationRegime(),
        },
        'scenario_2b': {
            'name': 'Managed Conflict (Institutional Parameters)',
            'description': 'Same model as Scen 2, but with tamed parameters; watch for chronic stagnation from low markups.',
            'params': PARAMS_MANAGED,
            'stocks': STOCKS_DEFAULT,
            'policy': POLICY_DEFAULT,
            'finance_regime': ProfitLedPrivateBanking(),
            'external_regime': FloatingExchangeRate(),
            'distribution_regime': ConflictInflationRegime(),
            'expected_outcome': 'Π remains negative until year 14; output shrinks toward 95 as capital is destroyed.',
        },
        'scenario_2c': {
            'name': 'Managed Conflict with Public Backstop',
            'description': 'Tamed conflict parameters plus boosted public investment to support the transition.',
            'params': PARAMS_TRANSITION,
            'stocks': STOCKS_NO_DEBT,
            'policy': POLICY_BOOSTED,
            'finance_regime': ProfitLedPrivateBanking(),
            'external_regime': FloatingExchangeRate(),
            'distribution_regime': ConflictInflationRegime(),
            'expected_outcome': 'Public demand makes Π turn positive by Year 5 and drives a high-growth transition path.',
        },
        'scenario_3': {
            'name': 'Coordinated Socialist Transition (Successful)',
            'description': 'State-led credit with capital controls and productivity-led planning.',
            'params': PARAMS_PLANNED,
            'stocks': STOCKS_NO_DEBT,
            'policy': POLICY_BOOSTED,
            'finance_regime': StateLedNationalizedBanking(),
            'external_regime': ManagedExchangeRate(),
            'distribution_regime': ConflictInflationRegime(),
            'expected_outcome': 'Coordinated planning maintains high growth, low leverage, and stable external balance.',
        },
        'scenario_3b': {
            'name': 'Partial Planning (Failed Transition)',
            'description': 'State credit without capital controls or productivity program triggers instability.',
            'params': PARAMS_PARTIAL,
            'stocks': STOCKS_NO_DEBT,
            'policy': POLICY_BOOSTED,
            'finance_regime': StateLedNationalizedBanking(),
            'external_regime': FloatingExchangeRate(),
            'distribution_regime': ConflictInflationRegime(),
            'expected_outcome': 'Absent controls, capital flight and falling productivity recreate instability.',
        },
    }
    
    # ========================================================================
    # RUN ALL SCENARIOS
    # ========================================================================
    
    results_collection = {}
    
    for scenario_id, config in scenarios.items():
        print(f"\n{'='*80}")
        print(f"Running {scenario_id.upper()}: {config['name']}")
        print(f"{'='*80}")
        print(f"   {config['description']}")
        
        model = SFCModel(
            stocks=config['stocks'],
            params=config['params'],
            policy=config['policy'],
            finance_regime=config['finance_regime'],
            external_regime=config['external_regime'],
            distribution_regime=config['distribution_regime'],
        )
        
        results = model.run()
        results_collection[scenario_id] = results
        
        model.summary()
        
        if 'expected_outcome' in config:
            print(f"\n   Expected Outcome: {config['expected_outcome']}")
        
        plot_simulation_results(
            results,
            model,
            scenario_name=f"Simulation Dashboard: {config['name']}",
        )
    
    # ========================================================================
    # EXPORT RESULTS FOR FURTHER ANALYSIS
    # ========================================================================
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print(f"EXPORTING RESULTS TO {OUTPUT_DIR}/")
    print("="*80)
    
    for scenario_id, results in results_collection.items():
        # Export CSV
        csv_path = OUTPUT_DIR / f"{scenario_id}_results.csv"
        results.to_csv(csv_path, index=False)
        print(f"Exported {scenario_id} to {csv_path}")
        
        # Export metadata
        metadata_path = OUTPUT_DIR / f"{scenario_id}_metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Scenario: {scenarios[scenario_id]['name']}\n")
            f.write(f"Description: {scenarios[scenario_id]['description']}\n\n")
            f.write("Parameters:\n")
            for key, value in results.attrs['params'].items():
                f.write(f"  {key} = {value}\n")
            f.write("\nRegimes:\n")
            for key, value in results.attrs['regimes'].items():
                f.write(f"  {key} = {value}\n")
        print(f"Exported {scenario_id} metadata to {metadata_path}")


if __name__ == "__main__":
    main()