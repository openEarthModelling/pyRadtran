"""Scene builder with immutable chain API.

Scene composes all Pydantic models into a complete uvspec configuration.
Each set_*() method returns a NEW Scene via copy.deepcopy() to avoid
mutability traps.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pyradtran.core.input_builder import build_input_text
from pyradtran.models.aerosol import AerosolConfig
from pyradtran.models.atmosphere import AtmosphereConfig
from pyradtran.models.cloud import CloudConfig
from pyradtran.models.mc import McConfig
from pyradtran.models.advanced import AdvancedConfig
from pyradtran.models.output import OutputConfig
from pyradtran.models.solver import SolverConfig
from pyradtran.models.source import SourceConfig
from pyradtran.models.surface import SurfaceConfig
from pyradtran.models.wavelength import WavelengthConfig


class Scene:
    """Immutable scene builder composing all uvspec configuration models.

    Each set_*() method returns a new Scene via deepcopy.
    Use .clone() for an explicit copy.

    Usage::

        scene = (
            Scene()
            .set_atmosphere(profile="us", altitude=2.663)
            .set_source_solar(sza=30.0)
            .set_wavelength(250.0, 1200.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"])
        )
    """

    def __init__(
        self,
        atmosphere: AtmosphereConfig | None = None,
        source: SourceConfig | None = None,
        wavelength: WavelengthConfig | None = None,
        solver: SolverConfig | None = None,
        output: OutputConfig | None = None,
        surface: SurfaceConfig | None = None,
        aerosol: AerosolConfig | None = None,
        cloud: CloudConfig | None = None,
        mc: McConfig | None = None,
        advanced: AdvancedConfig | None = None,
        raw_keywords: list[tuple[str, str]] | None = None,
    ):
        self.atmosphere = atmosphere
        self.source = source
        self.wavelength = wavelength
        self.solver = solver
        self.output = output
        self.surface = surface
        self.aerosol = aerosol
        self.cloud = cloud
        self.mc = mc
        self.advanced = advanced
        self.raw_keywords = raw_keywords or []

    def clone(self) -> Scene:
        """Create a deep copy of this Scene."""
        return copy.deepcopy(self)

    # --- Atmosphere ---

    def set_atmosphere(self, **kwargs) -> Scene:
        new = self.clone()
        if new.atmosphere is not None:
            new.atmosphere = new.atmosphere.model_copy(update=kwargs)
        else:
            new.atmosphere = AtmosphereConfig(**kwargs)
        return new

    def set_mol_modify(self, species: str, value: float, unit: str) -> Scene:
        new = self.clone()
        if new.atmosphere is None:
            new.atmosphere = AtmosphereConfig(profile="us")
        mol = list(new.atmosphere.mol_modify) + [(species, value, unit)]
        new.atmosphere = new.atmosphere.model_copy(update={"mol_modify": mol})
        return new

    # --- Source ---

    def set_source_solar(self, sza: float, **kwargs) -> Scene:
        new = self.clone()
        new.source = SourceConfig(source="solar", sza=sza, **kwargs)
        return new

    def set_source_thermal(self, **kwargs) -> Scene:
        new = self.clone()
        new.source = SourceConfig(source="thermal", **kwargs)
        return new

    # --- Wavelength ---

    def set_wavelength(self, wl_min: float, wl_max: float, **kwargs) -> Scene:
        new = self.clone()
        new.wavelength = WavelengthConfig(wavelength_min=wl_min, wavelength_max=wl_max, **kwargs)
        return new

    # --- Solver ---

    def set_solver(self, method: str = "disort", streams: int = 16, **kwargs) -> Scene:
        new = self.clone()
        new.solver = SolverConfig(method=method, streams=streams, **kwargs)
        return new

    # --- Output ---

    def set_output(self, **kwargs) -> Scene:
        new = self.clone()
        if new.output is not None:
            new.output = new.output.model_copy(update=kwargs)
        else:
            new.output = OutputConfig(**kwargs)
        return new

    # --- Surface ---

    def set_surface(self, **kwargs) -> Scene:
        new = self.clone()
        new.surface = SurfaceConfig(**kwargs)
        return new

    # --- Aerosol ---

    def set_aerosol(self, **kwargs) -> Scene:
        new = self.clone()
        if new.aerosol is not None:
            new.aerosol = new.aerosol.model_copy(update=kwargs)
        else:
            new.aerosol = AerosolConfig(**kwargs)
        return new

    # --- Cloud (Phase 2) ---

    def set_cloud(self, **kwargs) -> Scene:
        new = self.clone()
        if new.cloud is not None:
            new.cloud = new.cloud.model_copy(update=kwargs)
        else:
            new.cloud = CloudConfig(**kwargs)
        return new

    # --- Raw keywords (escape hatch) ---

    def add_raw_keyword(self, key: str, value: str = "") -> Scene:
        new = self.clone()
        new.raw_keywords = list(new.raw_keywords) + [(key, value)]
        return new

    # --- Build ---

    def build_input(self, data_files_path: str | None = None) -> str:
        """Build complete uvspec input text from this Scene.

        Raises:
            ValueError: If required components are missing.
        """
        if self.atmosphere is None:
            raise ValueError("Scene is missing atmosphere configuration. Call .set_atmosphere() first.")
        if self.source is None:
            raise ValueError("Scene is missing source configuration. Call .set_source_solar() or .set_source_thermal() first.")
        if self.wavelength is None:
            raise ValueError("Scene is missing wavelength configuration. Call .set_wavelength() first.")
        if self.solver is None:
            raise ValueError("Scene is missing solver configuration. Call .set_solver() first.")
        if self.output is None:
            output = OutputConfig(quiet=True)
        else:
            output = self.output

        return build_input_text(
            atmosphere=self.atmosphere,
            source=self.source,
            wavelength=self.wavelength,
            solver=self.solver,
            output=output,
            surface=self.surface,
            aerosol=self.aerosol,
            cloud=self.cloud,
            mc=self.mc,
            advanced=self.advanced,
            raw_keywords=self.raw_keywords or None,
            data_files_path=data_files_path,
        )

    def __repr__(self) -> str:
        components = []
        if self.atmosphere: components.append("atmosphere")
        if self.source: components.append("source")
        if self.wavelength: components.append("wavelength")
        if self.solver: components.append("solver")
        if self.output: components.append("output")
        if self.surface: components.append("surface")
        if self.aerosol: components.append("aerosol")
        if self.cloud: components.append("cloud")
        n_raw = len(self.raw_keywords) if self.raw_keywords else 0
        if n_raw: components.append(f"{n_raw} raw keywords")
        return f"Scene({', '.join(components)})"
