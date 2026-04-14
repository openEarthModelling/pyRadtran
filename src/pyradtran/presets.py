"""Site altitudes, standard atmosphere profiles, and common configurations."""

from __future__ import annotations

from typing import Union

SITES: dict[str, dict[str, float]] = {
    "LSST": {"altitude": 2.663, "pressure": 731.50433},
    "CTIO": {"altitude": 2.207, "pressure": 776.0},
    "OHP": {"altitude": 0.650, "pressure": 935.0},
    "PDM": {"altitude": 2.890, "pressure": 716.0},
    "OMK": {"altitude": 4.200, "pressure": 601.0},
    "OSL": {"altitude": 0.0, "pressure": 1013.25},
}

PROFILES: dict[str, str] = {
    "us": "US-standard",
    "ms": "midlatitude_summer",
    "mw": "midlatitude_winter",
    "tp": "tropics",
    "ss": "subarctic_summer",
    "sw": "subarctic_winter",
}


def resolve_altitude(altitude: Union[float, str]) -> float:
    """Resolve altitude -- accepts km float or site preset name."""
    if isinstance(altitude, (int, float)):
        return float(altitude)
    if isinstance(altitude, str):
        if altitude in SITES:
            return SITES[altitude]["altitude"]
        raise ValueError(
            f"Unknown site '{altitude}'. Available: {sorted(SITES.keys())}"
        )
    raise TypeError(f"altitude must be float or str, got {type(altitude)}")
