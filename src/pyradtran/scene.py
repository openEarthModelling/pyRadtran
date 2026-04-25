"""Scene builder with immutable chain API.

Scene composes all Pydantic models into a complete uvspec configuration.
Each set_*() method returns a NEW Scene via copy.deepcopy() to avoid
mutability traps.
"""

from __future__ import annotations

import copy

from pyradtran.core.input_builder import build_input_text
from pyradtran.models.advanced import AdvancedConfig
from pyradtran.models.aerosol import AerosolModel, OpacPreset, OpacPresetName
from pyradtran.models.atmosphere import AtmosphereConfig
from pyradtran.models.cloud import CloudConfig
from pyradtran.models.mc import McConfig
from pyradtran.models.output import OutputConfig
from pyradtran.models.solver import SolverConfig
from pyradtran.models.source import SourceConfig
from pyradtran.models.special import SpecialConfig
from pyradtran.models.sslidar import SslidarConfig
from pyradtran.models.surface import SurfaceConfig
from pyradtran.models.three_d import ThreeDConfig
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
        aerosol: AerosolModel | None = None,
        cloud: CloudConfig | None = None,
        mc: McConfig | None = None,
        sslidar: SslidarConfig | None = None,
        advanced: AdvancedConfig | None = None,
        three_d: ThreeDConfig | None = None,
        special: SpecialConfig | None = None,
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
        self.sslidar = sslidar
        self.advanced = advanced
        self.three_d = three_d
        self.special = special
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

    def set_wavelength(self, wl_min: float, wl_max: float | None = None, **kwargs) -> Scene:
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
        if new.surface is not None:
            new.surface = new.surface.model_copy(update=kwargs)
        else:
            new.surface = SurfaceConfig(**kwargs)
        return new

    # --- Aerosol ---

    def set_aerosol(self, aerosol: AerosolModel) -> Scene:
        """Set aerosol configuration from an AerosolModel instance."""
        new = self.clone()
        new.aerosol = aerosol
        return new

    def set_aerosol_modify(
        self, variable: str, action: str, value: float
    ) -> Scene:
        """Add an aerosol modification directive.

        Args:
            variable: Property to modify (gg, ssa, tau, tau550).
            action: Modification type (scale or set).
            value: Numeric value.
        """
        from pyradtran.models.aerosol import AerosolModifyEntry, OpacPreset, OpacPresetName

        new = self.clone()
        if new.aerosol is None:
            new.aerosol = OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE)
        entry = AerosolModifyEntry(variable=variable, action=action, value=value)
        modify_list = list(new.aerosol.modify) + [entry]
        new.aerosol = new.aerosol.model_copy(update={"modify": modify_list})
        return new

    # --- Cloud (Phase 2) ---

    def set_cloud(self, **kwargs) -> Scene:
        new = self.clone()
        if new.cloud is not None:
            new.cloud = new.cloud.model_copy(update=kwargs)
        else:
            new.cloud = CloudConfig(**kwargs)
        return new

    # --- Monte Carlo (Phase 3) ---

    def set_mc(self, **kwargs) -> Scene:
        new = self.clone()
        if new.mc is not None:
            new.mc = new.mc.model_copy(update=kwargs)
        else:
            new.mc = McConfig(**kwargs)
        return new

    def set_sslidar(self, **kwargs) -> Scene:
        new = self.clone()
        if new.sslidar is not None:
            new.sslidar = new.sslidar.model_copy(update=kwargs)
        else:
            new.sslidar = SslidarConfig(**kwargs)
        return new

    def set_advanced(self, **kwargs) -> Scene:
        new = self.clone()
        if new.advanced is not None:
            new.advanced = new.advanced.model_copy(update=kwargs)
        else:
            new.advanced = AdvancedConfig(**kwargs)
        return new

    # --- 3D (Phase 4) ---

    def set_three_d(self, **kwargs) -> Scene:
        new = self.clone()
        if new.three_d is not None:
            new.three_d = new.three_d.model_copy(update=kwargs)
        else:
            new.three_d = ThreeDConfig(**kwargs)
        return new

    def set_special(self, **kwargs) -> Scene:
        new = self.clone()
        if new.special is not None:
            new.special = new.special.model_copy(update=kwargs)
        else:
            new.special = SpecialConfig(**kwargs)
        return new

    def set_satellite(self, geometry: str | None = None,
                      pixel: tuple[int, int] | None = None,
                      **source_kwargs) -> Scene:
        new = self.clone()
        sat_updates = {}
        if geometry is not None:
            sat_updates["satellite_geometry"] = geometry
        if pixel is not None:
            sat_updates["satellite_pixel"] = pixel
        if new.source is not None:
            new.source = new.source.model_copy(update={**sat_updates, **source_kwargs})
        else:
            new.source = SourceConfig(source="solar", sza=0.0, **sat_updates, **source_kwargs)
        return new

    def set_dynamic(self, method: str = "dynamic_tenstream",
                    iterations: int | None = None, **kwargs) -> Scene:
        new = self.clone()
        solver_kwargs = {"method": method, **kwargs}
        if iterations is not None:
            solver_kwargs["dynamic_iterations"] = iterations
        new.solver = SolverConfig(**solver_kwargs)
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
            raise ValueError(
                "Scene is missing atmosphere. Call .set_atmosphere() first."
            )
        if self.source is None:
            raise ValueError(
                "Scene is missing source. Call .set_source_solar() or "
                ".set_source_thermal() first."
            )
        if self.wavelength is None:
            raise ValueError(
                "Scene is missing wavelength. Call .set_wavelength() first."
            )
        if self.solver is None:
            raise ValueError(
                "Scene is missing solver. Call .set_solver() first."
            )
        output = self.output if self.output is not None else OutputConfig(quiet=True)

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
            sslidar=self.sslidar,
            advanced=self.advanced,
            three_d=self.three_d,
            special=self.special,
            raw_keywords=self.raw_keywords or None,
            data_files_path=data_files_path,
        )

    def __repr__(self) -> str:
        components = []
        if self.atmosphere:
            components.append("atmosphere")
        if self.source:
            components.append("source")
        if self.wavelength:
            components.append("wavelength")
        if self.solver:
            components.append("solver")
        if self.output:
            components.append("output")
        if self.surface:
            components.append("surface")
        if self.aerosol:
            components.append("aerosol")
        if self.cloud:
            components.append("cloud")
        if self.mc:
            components.append("mc")
        if self.sslidar:
            components.append("sslidar")
        if self.advanced:
            components.append("advanced")
        if self.three_d:
            components.append("three_d")
        if self.special:
            components.append("special")
        n_raw = len(self.raw_keywords) if self.raw_keywords else 0
        if n_raw:
            components.append(f"{n_raw} raw keywords")
        return f"Scene({', '.join(components)})"
