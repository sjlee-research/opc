"""Configuration dataclasses, loadable from YAML.

All lengths are in nm. Every field has a default, so a YAML file only needs to
list the values that differ from the defaults.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OpticsConfig:
    model: str = "hopkins"            # "intensity" (PSF intensity superposition) | "hopkins" (TCC -> SOCS, interference)
                                      # | "gaussian" (single coherent Gaussian kernel)
    wavelength_nm: float = 193.0
    NA: float = 1.35
    immersion_index: float = 1.44     # refractive index on the wafer side (1.0 = dry)
    illumination: str = "annular"     # conventional | annular | dipole_x | dipole_y | quasar
    sigma: float = 0.7                # conventional: partial coherence factor
    sigma_in: float = 0.6             # annular / dipole / quasar
    sigma_out: float = 0.9
    pole_angle_deg: float = 30.0      # opening angle of each dipole / quasar pole
    source_grid: int = 41             # source sampling points across sigma in [-1, 1]
    num_kernels: int = 24             # SOCS truncation order
    gaussian_sigma_nm: float | None = None   # Gaussian model; None -> 0.2 * wavelength / NA
    pixel_compensation: bool = True   # undo the 1-pixel box blur introduced by area-coverage rasterization
    # intensity model: E(x) = sum over mask points of transmission * PSF(x - x')
    psf: str = "gaussian"             # "gaussian" | "array"
    psf_sigma_nm: float | None = None   # None -> 0.21 * wavelength / NA (Gaussian fit of the Airy intensity PSF)
    psf_radius_sigma: float = 4.0     # PSF truncated at radius psf_radius_sigma * sigma, then renormalized
    psf_array: list[list[float]] | None = None   # "array": relative weights per cell (odd square, center = peak)
    psf_array_cell_nm: float = 3.5    # physical size of one cell of psf_array


@dataclass
class ResistConfig:
    threshold: float | None = None    # None -> derived by threshold_rule
    threshold_rule: str = "anchor"    # "anchor": dose-to-size on the anchor L/S | "edge": intensity at the edge of
                                      # a large isolated feature (that edge then prints at its drawn position)
    diffusion_nm: float = 0.0         # Gaussian blur sigma of the latent image (acid diffusion)
    anchor_cd_nm: float = 60.0        # anchor: infinite line/space array printed exactly on target
    anchor_pitch_nm: float = 180.0    # 1:2 line/space


@dataclass
class GridConfig:
    pixel_nm: float = 2.0
    margin_nm: float = 600.0          # optical ambit kept around each simulated clip
    max_pixels: int = 4096            # per side; larger clips must be split


@dataclass
class ProcessConfig:
    """Process-window conditions: PV bands use the full doses x defocus grid."""
    doses: list[float] = field(default_factory=lambda: [0.97, 1.0, 1.03])
    defocus_nm: list[float] = field(default_factory=lambda: [-40.0, 0.0, 40.0])


@dataclass
class FragmentConfig:
    corner_len_nm: float = 20.0       # segment length next to every corner
    segment_len_nm: float = 50.0      # target length of interior segments
    min_segment_nm: float = 10.0
    line_end_max_nm: float = 100.0    # edge between two convex corners shorter than this = line end
    corner_site: float = 1.0          # control site on corner segments (0 = at the vertex, 1 = corner_len away):
                                      # 1.0 leaves a corner-rounding allowance of corner_len around every vertex
    max_range_nm: float = 1000.0      # width/space ray-cast range ("isolated" beyond this)


@dataclass
class MRCConfig:
    """Wafer-scale mask rules (x4 on the reticle): used both as OPC move limits and for the final check."""
    min_width_nm: float = 20.0
    min_space_nm: float = 20.0


@dataclass
class RuleConfig:
    """Rule-based OPC deck. Bias table and line-end extension are derived from 1D/2D simulations."""
    bias_spaces_nm: list[float] = field(
        default_factory=lambda: [60, 70, 80, 90, 105, 120, 140, 160, 180, 210, 240, 300, 400, 600, 1000])
    serif_nm: float = 10.0            # outward move of corner segments at convex corners
    inner_serif_nm: float = 6.0       # inward move of corner segments at concave corners
    hammer_nm: float = 6.0            # extra outward move of the side segments at a line end
    line_end_ext_nm: float | None = None   # None -> calibrated so an isolated line end prints on target
    max_bias_nm: float = 25.0


@dataclass
class MBOPCConfig:
    iterations: int = 30
    damping: float = 0.5              # constant feedback gain: d <- d - damping * EPE
    adaptive: bool = False            # contribution A: per-segment adaptive damping (sign-history)
    adapt_up: float = 1.25
    adapt_down: float = 0.5
    damping_min: float = 0.1
    damping_max: float = 1.2
    init: str = "none"                # "none" | "rule"   (contribution B: rule-based initial mask)
    pw_aware: bool = False            # contribution C: process-window-aware EPE
    pw_nominal_weight: float = 0.5    # EPE_eff = w * EPE_nominal + (1 - w) * mean(EPE at PW corners)
    pw_dose: float = 0.03             # PW corners used inside the loop: (1 +- dose, 0) and (1, +-defocus)
    pw_defocus_nm: float = 40.0
    max_out_nm: float = 40.0
    max_in_nm: float = 25.0
    mrc_guard: bool = True            # revert segment moves that would create a width/space/notch violation
    tol_rms_nm: float = 0.5           # "converged" = rms EPE < tol_rms and max |EPE| < tol_max
    tol_max_nm: float = 2.0


@dataclass
class ILTConfig:
    iterations: int = 150
    lr: float = 0.08
    mask_steepness: float = 4.0       # m = sigmoid(beta * theta)
    resist_steepness: float = 60.0    # Z = sigmoid(alpha * (I - threshold))
    binarization_weight: float = 0.02
    tv_weight: float = 0.002
    corner_weight: float = 0.1        # loss weight inside the corner-rounding allowance (corner_len around vertices)


@dataclass
class PatternConfig:
    cd_nm: float = 60.0
    line_length_nm: float = 1000.0
    dense_pitches_nm: list[float] = field(default_factory=lambda: [120.0, 180.0, 240.0])
    num_lines: int = 7
    line_end_gap_nm: float = 80.0
    line_end_pitch_nm: float = 180.0
    arm_length_nm: float = 500.0
    contact_nm: float = 90.0
    contact_pitch_nm: float = 180.0
    through_pitch_nm: list[float] = field(
        default_factory=lambda: [120, 130, 140, 150, 165, 180, 200, 220, 240, 270, 300, 350, 400, 500, 600, 800])
    iso_pitch_nm: float = 2400.0      # pitch treated as "isolated" in the periodic through-pitch sweep


@dataclass
class Config:
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    resist: ResistConfig = field(default_factory=ResistConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    patterns: PatternConfig = field(default_factory=PatternConfig)
    fragment: FragmentConfig = field(default_factory=FragmentConfig)
    mrc: MRCConfig = field(default_factory=MRCConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    mbopc: MBOPCConfig = field(default_factory=MBOPCConfig)
    ilt: ILTConfig = field(default_factory=ILTConfig)
    device: str = "cpu"
    dtype: str = "float32"            # float32 | float64

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path is not None:
            with open(path) as f:
                _update(cfg, yaml.safe_load(f) or {})
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _update(obj: Any, values: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, val in values.items():
        if key not in names:
            raise KeyError(f"unknown config key '{key}' in {type(obj).__name__}")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            _update(cur, val or {})
        else:
            setattr(obj, key, val)
