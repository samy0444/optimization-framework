
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


