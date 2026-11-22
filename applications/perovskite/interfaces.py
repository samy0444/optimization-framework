# interfaces.py
from dataclasses import dataclass

@dataclass
class Interface:
    """Holds parameters for one device interface."""
    name: str
    recomb_velocity: float  # in cm/s (higher → more recombination)
    trap_density: float     # in cm⁻² (higher → more traps)

