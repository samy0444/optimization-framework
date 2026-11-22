
# layers.py
from dataclasses import dataclass
from typing import Optional
@dataclass
class Layer:
    name: str
    L_nm: float           # thickness in nm
    D0: float             # diffusion constant (for lifetime model)
    Ea: float             # activation energy for diffusion or other process
    E: float              # (existing code used this for mechanical modulus, repurpose or keep)
    CTE: float            # coefficient of thermal expansion
    # New properties for device physics:
    doping: float = None          # doping concentration (cm^-3) if applicable
    mobility: float = None        # carrier mobility (cm^2/Vs)
    band_offset: float = 0.0      # band offset at interface (eV) if this layer is ETL/HTL


def default_stack():
    """Default 5-layer device used by the app, the RL environment and the benchmark.

    Layer order and values match the synthetic lifetime data the surrogate was
    trained on (synthetic_datagen/lifetime_data_gen.py): each value is the
    midpoint of the range that generator samples, with D0 at the geometric
    midpoint of 1e-16..1e-6. Units: thickness in nm, D0 in cm^2/s, Ea in eV,
    E (Young's modulus) in GPa, CTE in 1/K. Keeping inputs inside the training
    range stops the lifetime surrogate from extrapolating.
    """
    return [
        Layer('FTO',   600, 1e-11, 0.45, 70.0,  9e-6),
        Layer('HTL',   150, 1e-11, 0.50, 3.0,   60e-6, mobility=5e-4, band_offset=0.2),
        Layer('Perov', 550, 1e-11, 0.40, 15.0,  35e-6),
        Layer('ETL',    80, 1e-11, 0.60, 115.0, 10e-6, mobility=1e-3, band_offset=0.3),
        Layer('Ag',    100, 1e-11, 0.70, 82.5,  19e-6),
    ]


