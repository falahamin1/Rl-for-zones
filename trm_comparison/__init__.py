"""TRM Frozen Lake comparison: zone-based vs region-based Q-learning."""
from .trm import TRMInstance, TRMState, INSTANCES, U_TERMINAL
from .grid import GridEnv, ACTIONS
from .zone_agent import TRMZoneAgent
from .region_agent import TRMRegionAgent

__all__ = [
    "TRMInstance", "TRMState", "INSTANCES", "U_TERMINAL",
    "GridEnv", "ACTIONS",
    "TRMZoneAgent", "TRMRegionAgent",
]
