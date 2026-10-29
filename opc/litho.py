"""Lithography model: optics, torch simulator, resist and measurements.

Sections
  1. Optics          source, pupil, Hopkins TCC -> SOCS kernels, Gaussian kernel, Abbe reference
  2. Simulator       LithoSimulator: mask -> aerial image -> resist latent image -> printed contour
  3. Measurements    EPE, CD gauges, dose-to-size threshold calibration, through-pitch CD
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage
from scipy.fft import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from .config import Config, OpticsConfig
from .layout import Gauge, Grid, Polygon, ensure_ccw, extract_contours, rasterize, rect


# ==================================================================================================
# 1. Optics
# ==================================================================================================
#
# Partially coherent imaging kernels (scalar, thin-mask / Kirchhoff).
#
# Hopkins model
# -------------
# With source intensity J(s) and pupil P(f), the aerial image of a mask with
# spectrum M(f) is
#
#     I(x) = sum_{f1, f2} TCC(f1, f2) M(f1) M*(f2) exp(i 2 pi (f1 - f2) x),
#     TCC(f1, f2) = sum_s J(s) P(f1 + s) P*(f2 + s).
#
# Sampling the source at points s_n with weights w_n and stacking
# A[n, f] = sqrt(w_n) P(f + s_n) gives TCC = A^H A. Its eigen-decomposition
# (computed through the small Gram matrix A A^H) yields the SOCS form
#
#     I(x) = sum_k lambda_k |(phi_k * m)(x)|^2,    phi_k = k-th eigenvector of TCC,
#
# which is truncated to the K strongest kernels.
#
# Kernels are stored in the frequency domain on the box |kx| <= rx, |ky| <= ry of
# the simulation grid's DFT lattice (f = k / (n * pixel)), where rx, ry cover the
# TCC support |f| <= NA (1 + sigma_max) / lambda.


@dataclass
class FreqKernels:
    phi: np.ndarray        # (K, 2ry+1, 2rx+1) complex, frequency-domain kernels
    weights: np.ndarray    # (K,) SOCS eigenvalues lambda_k
    ry: int
    rx: int
    eigenvalues: np.ndarray  # full TCC spectrum (diagnostics)

    @property
    def captured_energy(self) -> float:
        return float(self.weights.sum() / self.eigenvalues.sum())


# ------------------------------------------------------------------------ source
def source_points(o: OpticsConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretized illumination: (sx, sy, weights) in sigma units (|s| = 1 is the pupil edge)."""
    s = np.linspace(-1.0, 1.0, o.source_grid)
    sx, sy = np.meshgrid(s, s)
    r, th = np.hypot(sx, sy), np.arctan2(sy, sx)
    tol = 1e-9                                       # linspace round-off must not break the source symmetry
    half = np.deg2rad(o.pole_angle_deg) / 2 + tol
    ring = (r >= o.sigma_in - tol) & (r <= o.sigma_out + tol)

    def near(angles):
        return np.any([np.abs(np.angle(np.exp(1j * (th - a)))) <= half for a in angles], axis=0)

    kind = o.illumination
    if kind == "conventional":
        sel = r <= o.sigma + tol
    elif kind == "annular":
        sel = ring
    elif kind == "dipole_x":
        sel = ring & near([0, np.pi])
    elif kind == "dipole_y":
        sel = ring & near([np.pi / 2, -np.pi / 2])
    elif kind == "quasar":
        sel = ring & near([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4])
    else:
        raise ValueError(f"unknown illumination '{kind}'")
    if not sel.any():
        raise ValueError("illumination selects no source points; increase source_grid")
    w = np.full(sel.sum(), 1.0 / sel.sum())
    return sx[sel], sy[sel], w


def source_bitmap(o: OpticsConfig) -> np.ndarray:
    s = np.linspace(-1.0, 1.0, o.source_grid)
    sx, sy, _ = source_points(o)
    img = np.zeros((o.source_grid, o.source_grid))
    img[np.searchsorted(s, sy - 1e-9), np.searchsorted(s, sx - 1e-9)] = 1.0
    return img


# ------------------------------------------------------------------------- pupil
def pupil(fx: np.ndarray, fy: np.ndarray, o: OpticsConfig, defocus_nm: float = 0.0) -> np.ndarray:
    """Scalar pupil: circular aperture NA / lambda with an exact (non-paraxial) defocus phase."""
    lam, n = o.wavelength_nm, o.immersion_index
    if o.NA >= n:
        raise ValueError(f"NA={o.NA} must be smaller than the immersion index {n}")
    rho2 = fx**2 + fy**2
    P = (rho2 <= (o.NA / lam) ** 2 * (1 + 1e-9)).astype(complex)
    if defocus_nm:
        kz = np.sqrt(np.maximum((n / lam) ** 2 - rho2, 0.0))
        P *= np.exp(2j * np.pi * defocus_nm * (kz - n / lam))
    return P


def _box_freqs(shape: tuple[int, int], pixel: float, ry: int, rx: int):
    ny, nx = shape
    fx = np.arange(-rx, rx + 1) / (nx * pixel)
    fy = np.arange(-ry, ry + 1) / (ny * pixel)
    return np.meshgrid(fx, fy)


def _pixel_compensation(FX: np.ndarray, FY: np.ndarray, pixel: float) -> np.ndarray:
    # Area-coverage rasterization = true mask convolved with a 1-pixel box, i.e. spectrum * sinc.
    return 1.0 / (np.sinc(FX * pixel) * np.sinc(FY * pixel))


def _box_radius(f_max: float, n: int, pixel: float) -> int:
    r = int(np.floor(f_max * n * pixel))
    if 4 * r + 1 > n:
        raise ValueError(f"grid too coarse: pixel={pixel} nm cannot hold the image bandwidth "
                         f"(need pixel < {1 / (4 * f_max):.1f} nm)")
    return r


# ------------------------------------------------------------------------ models
def hopkins_kernels(o: OpticsConfig, shape: tuple[int, int], pixel: float,
                    defocus_nm: float = 0.0, num_kernels: int | None = None) -> FreqKernels:
    """SOCS decomposition of the Hopkins TCC on the DFT lattice of a (ny, nx) grid."""
    sx, sy, w = source_points(o)
    na_f = o.NA / o.wavelength_nm
    f_max = na_f * (1.0 + np.hypot(sx, sy).max())
    ry, rx = _box_radius(f_max, shape[0], pixel), _box_radius(f_max, shape[1], pixel)
    FX, FY = _box_freqs(shape, pixel, ry, rx)
    support = FX**2 + FY**2 <= f_max**2 * (1 + 1e-9)

    # A[n, f] = sqrt(w_n) P(f + s_n) restricted to the TCC support -> TCC = A^H A
    A = np.sqrt(w)[:, None] * pupil(FX[support][None, :] + sx[:, None] * na_f,
                                    FY[support][None, :] + sy[:, None] * na_f, o, defocus_nm)
    gram = A @ A.conj().T                                   # (S, S), same nonzero spectrum as TCC
    evals, U = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals, U = np.clip(evals[order], 0.0, None), U[:, order]

    K = num_kernels or o.num_kernels
    K = int(min(K, np.count_nonzero(evals > 1e-10 * evals[0])))
    V = (U[:, :K].conj().T @ A) / np.sqrt(evals[:K])[:, None]   # (K, F) orthonormal TCC eigenvectors

    phi = np.zeros((K, 2 * ry + 1, 2 * rx + 1), dtype=complex)
    phi[:, support] = V
    if o.pixel_compensation:
        phi *= _pixel_compensation(FX, FY, pixel)[None]
    return FreqKernels(phi=phi, weights=evals[:K].copy(), ry=ry, rx=rx, eigenvalues=evals)


def gaussian_sigma(o: OpticsConfig) -> float:
    return o.gaussian_sigma_nm if o.gaussian_sigma_nm else 0.2 * o.wavelength_nm / o.NA


def gaussian_kernels(o: OpticsConfig, shape: tuple[int, int], pixel: float) -> FreqKernels:
    """Single coherent Gaussian kernel: I = |m * g|^2, g ~ exp(-r^2 / 2 sigma^2), unit DC gain."""
    sigma = gaussian_sigma(o)
    f_max = np.sqrt(np.log(1e6) / (2 * np.pi**2)) / sigma      # spectrum below 1e-6 outside
    ry = min(int(np.floor(f_max * shape[0] * pixel)), (shape[0] - 1) // 4)
    rx = min(int(np.floor(f_max * shape[1] * pixel)), (shape[1] - 1) // 4)
    FX, FY = _box_freqs(shape, pixel, ry, rx)
    phi = np.exp(-2 * np.pi**2 * sigma**2 * (FX**2 + FY**2)).astype(complex)
    if o.pixel_compensation:
        phi *= _pixel_compensation(FX, FY, pixel)
    return FreqKernels(phi=phi[None], weights=np.ones(1), ry=ry, rx=rx, eigenvalues=np.ones(1))


def abbe_aerial(mask: np.ndarray, o: OpticsConfig, pixel: float, defocus_nm: float = 0.0) -> np.ndarray:
    """Reference aerial image by brute-force Abbe summation over every source point (validation only)."""
    ny, nx = mask.shape
    FX, FY = np.meshgrid(np.fft.fftfreq(nx, pixel), np.fft.fftfreq(ny, pixel))
    M = np.fft.fft2(mask, norm="forward")
    if o.pixel_compensation:
        M = M * _pixel_compensation(FX, FY, pixel)
    sx, sy, w = source_points(o)
    na_f = o.NA / o.wavelength_nm
    img = np.zeros(mask.shape)
    for a, b, wt in zip(sx, sy, w):
        E = np.fft.ifft2(M * pupil(FX + a * na_f, FY + b * na_f, o, defocus_nm), norm="forward")
        img += wt * np.abs(E) ** 2
    return img


def spatial_kernels(k: FreqKernels, shape: tuple[int, int], pixel: float, size_nm: float = 600.0) -> np.ndarray:
    """Spatial-domain kernels phi_k(x), cropped to a size_nm window (for figures)."""
    ny, nx = shape
    out = []
    for phi in k.phi:
        full = np.zeros(shape, dtype=complex)
        iy = np.arange(-k.ry, k.ry + 1) % ny
        ix = np.arange(-k.rx, k.rx + 1) % nx
        full[np.ix_(iy, ix)] = phi
        sp = np.fft.fftshift(np.fft.ifft2(full, norm="forward"))
        h = int(size_nm / pixel / 2)
        cy, cx = ny // 2, nx // 2
        out.append(sp[cy - h:cy + h, cx - h:cx + h])
    return np.array(out)


def psf_sigma(o: OpticsConfig) -> float:
    """Gaussian PSF width of the intensity model: 0.21 * lambda / NA (Gaussian fit of the Airy intensity pattern)."""
    return o.psf_sigma_nm if o.psf_sigma_nm else 0.21 * o.wavelength_nm / o.NA


def intensity_psf(o: OpticsConfig, pixel: float) -> np.ndarray:
    """Point-spread function of the intensity-superposition model on the simulation pixel grid.

    gaussian: exp(-r^2 / 2 sigma^2) inside r <= psf_radius_sigma * sigma (zero outside);
    array:    the user weight table, each cell spread uniformly over its pixels.
    Both are normalized to sum 1, so a fully open (clear) area receives exposure 1.
    """
    if o.psf == "gaussian":
        sig = psf_sigma(o)
        R = o.psf_radius_sigma * sig
        r = int(np.ceil(R / pixel))
        x = np.arange(-r, r + 1) * pixel
        X, Y = np.meshgrid(x, x)
        k = np.exp(-(X**2 + Y**2) / (2 * sig**2)) * (X**2 + Y**2 <= R**2)
    elif o.psf == "array":
        W = np.asarray(o.psf_array, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] % 2 == 0:
            raise ValueError("psf_array must be an odd square table")
        n = int(round(o.psf_array_cell_nm / pixel))
        k = np.kron(W, np.ones((n, n))) / n**2
        if k.shape[0] % 2 == 0:                                  # keep the peak on a pixel center
            k = np.pad(k, ((0, 1), (0, 1)))
    else:
        raise ValueError(f"unknown psf '{o.psf}'")
    return k / k.sum()


def psf_cell_table(o: OpticsConfig, cell_nm: float = 3.5, n: int = 5) -> np.ndarray:
    """PSF integrated over an n x n table of cell_nm cells around the center, scaled so the center cell is 10
    (the same presentation as a weight-table PSF)."""
    sub = 21                                                     # odd: cell centers fall on pixel centers
    k = intensity_psf(o, cell_nm / sub)
    half, r = n // 2, k.shape[0] // 2
    pad = max(0, (half + 1) * sub - r)
    k = np.pad(k, pad)
    c = k.shape[0] // 2
    t = np.array([[k[c + a * sub - sub // 2: c + a * sub + sub // 2 + 1,
                     c + b * sub - sub // 2: c + b * sub + sub // 2 + 1].sum()
                   for b in range(-half, half + 1)] for a in range(-half, half + 1)])
    return 10 * t / t[half, half]


# ==================================================================================================
# 2. Simulator
# ==================================================================================================
#
# Torch lithography simulator: mask (pixels) -> aerial image -> resist latent image -> printed contour.
#
# Aerial image evaluation (SOCS, all steps differentiable):
#   1. M = FFT(mask)                                  full grid
#   2. keep the kernel box |kx| <= rx, |ky| <= ry     only frequencies the pupil passes
#   3. E_k = IFFT(M * phi_k) on a small coarse grid   (>= 4r+1 samples, so |E_k|^2 does not alias)
#   4. I_c = sum_k lambda_k |E_k|^2
#   5. band-limited (Fourier) upsampling of I_c back to the full grid
# Steps 3-4 cost K tiny FFTs instead of K full-size ones; the result equals the
# direct full-grid SOCS sum because I is band-limited to |f| <= 2 f_max.


@dataclass
class _TorchKernels:
    phi: torch.Tensor       # (K, by, bx) complex
    weights: torch.Tensor   # (K,)
    iy: torch.Tensor        # box rows in the full spectrum
    ix: torch.Tensor
    ncy: int                # coarse grid size
    ncx: int
    icy: torch.Tensor       # box rows in the coarse spectrum
    icx: torch.Tensor
    jcy: torch.Tensor       # intensity band (|k| <= 2r) in the coarse spectrum
    jcx: torch.Tensor
    jy: torch.Tensor        # intensity band in the full spectrum
    jx: torch.Tensor
    info: FreqKernels


@dataclass
class SimResult:
    grid: Grid
    mask: np.ndarray        # rasterized transmission (coverage) image
    aerial: np.ndarray      # aerial image (clear field = 1)
    latent: np.ndarray      # resist latent image (aerial after diffusion)
    level: float            # contour level on the latent image = threshold / dose
    contours: list[Polygon] = field(default_factory=list)


class LithoSimulator:
    """Intensity-superposition, Hopkins/SOCS or coherent-Gaussian imaging + constant-threshold resist.

    Mask tone: transmission 1 inside drawn polygons, 0 outside (dark-field mask);
    resist prints where dose * latent >= threshold.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rdtype = torch.float64 if cfg.dtype == "float64" else torch.float32
        self.cdtype = torch.complex128 if cfg.dtype == "float64" else torch.complex64
        self._kernels: dict[tuple, _TorchKernels] = {}
        self._diffusion: dict[tuple, torch.Tensor] = {}
        self._psf_hat: dict[tuple, torch.Tensor] = {}
        self._threshold: float | None = cfg.resist.threshold

    # ------------------------------------------------------------------ kernels
    def kernels(self, shape: tuple[int, int], pixel: float, defocus_nm: float = 0.0) -> _TorchKernels:
        o = self.cfg.optics
        key = (shape, pixel, round(float(defocus_nm), 6))
        if key in self._kernels:
            return self._kernels[key]
        if o.model == "hopkins":
            fk = hopkins_kernels(o, shape, pixel, defocus_nm)
        elif o.model == "gaussian":
            if defocus_nm:
                raise NotImplementedError("the Gaussian model has no focus dependence; use model='hopkins'")
            fk = gaussian_kernels(o, shape, pixel)
        else:
            raise ValueError(f"unknown optics model '{o.model}'")

        ny, nx = shape
        ncy = min(next_fast_len(4 * fk.ry + 2), ny)
        ncx = min(next_fast_len(4 * fk.rx + 2), nx)

        def idx(r: int, n: int) -> torch.Tensor:
            return torch.as_tensor(np.arange(-r, r + 1) % n, device=self.device)

        tk = _TorchKernels(
            phi=torch.as_tensor(fk.phi, dtype=self.cdtype, device=self.device),
            weights=torch.as_tensor(fk.weights, dtype=self.rdtype, device=self.device),
            iy=idx(fk.ry, ny), ix=idx(fk.rx, nx), ncy=ncy, ncx=ncx,
            icy=idx(fk.ry, ncy), icx=idx(fk.rx, ncx),
            jcy=idx(2 * fk.ry, ncy), jcx=idx(2 * fk.rx, ncx),
            jy=idx(2 * fk.ry, ny), jx=idx(2 * fk.rx, nx),
            info=fk,
        )
        self._kernels[key] = tk
        return tk

    # ------------------------------------------------------------------ imaging
    def psf_hat(self, shape: tuple[int, int], pixel: float) -> torch.Tensor:
        """FFT of the intensity-model PSF placed at the origin of a (ny, nx) periodic grid (cached)."""
        key = (shape, pixel)
        if key not in self._psf_hat:
            k = intensity_psf(self.cfg.optics, pixel)
            r = k.shape[0] // 2
            full = np.zeros(shape)                                   # periodic (wrapped) placement at the origin
            iy, ix = np.meshgrid(np.arange(-r, r + 1) % shape[0], np.arange(-r, r + 1) % shape[1], indexing="ij")
            np.add.at(full, (iy, ix), k)
            self._psf_hat[key] = torch.fft.fft2(torch.as_tensor(full, dtype=self.rdtype, device=self.device))
        return self._psf_hat[key]

    def aerial(self, mask: torch.Tensor, pixel: float, defocus_nm: float = 0.0) -> torch.Tensor:
        """Aerial image of a (ny, nx) or (B, ny, nx) mask transmission tensor."""
        single = mask.dim() == 2
        m = mask[None] if single else mask
        B, ny, nx = m.shape
        if self.cfg.optics.model == "intensity":               # E = mask (*) PSF, a plain sum of light amounts
            if defocus_nm:
                raise NotImplementedError("the intensity model has no focus dependence")
            I = torch.fft.ifft2(torch.fft.fft2(m.to(self.rdtype)) * self.psf_hat((ny, nx), pixel)).real
            return I[0] if single else I
        k = self.kernels((ny, nx), pixel, defocus_nm)

        M = torch.fft.fft2(m.to(self.rdtype), norm="forward")
        Mb = M[:, k.iy[:, None], k.ix[None, :]]                                  # (B, by, bx)
        Ec = torch.zeros((B, len(k.weights), k.ncy, k.ncx), dtype=self.cdtype, device=self.device)
        Ec[:, :, k.icy[:, None], k.icx[None, :]] = Mb[:, None] * k.phi[None]
        e = torch.fft.ifft2(Ec, norm="forward")                                   # coarse coherent fields
        Ic = torch.einsum("k,bkyx->byx", k.weights, e.real**2 + e.imag**2)
        S = torch.fft.fft2(Ic, norm="forward")[:, k.jcy[:, None], k.jcx[None, :]]
        F = torch.zeros((B, ny, nx), dtype=self.cdtype, device=self.device)
        F[:, k.jy[:, None], k.jx[None, :]] = S
        I = torch.fft.ifft2(F, norm="forward").real
        return I[0] if single else I

    def latent(self, aerial: torch.Tensor, pixel: float) -> torch.Tensor:
        """Resist latent image: aerial image blurred by a Gaussian acid-diffusion kernel."""
        sig = self.cfg.resist.diffusion_nm
        if sig <= 0:
            return aerial
        shape = tuple(aerial.shape[-2:])
        key = (shape, pixel, sig)
        if key not in self._diffusion:
            fy = torch.fft.fftfreq(shape[0], pixel, device=self.device, dtype=self.rdtype)
            fx = torch.fft.fftfreq(shape[1], pixel, device=self.device, dtype=self.rdtype)
            self._diffusion[key] = torch.exp(-2 * np.pi**2 * sig**2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
        return torch.fft.ifft2(torch.fft.fft2(aerial) * self._diffusion[key]).real

    # ------------------------------------------------------------------- resist
    @property
    def threshold(self) -> float:
        if self._threshold is None:
            r = self.cfg.resist
            self._threshold = (edge_threshold(self) if r.threshold_rule == "edge"
                               else calibrate_threshold(self, r.anchor_cd_nm, r.anchor_pitch_nm))
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    # ------------------------------------------------------------- convenience
    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=self.rdtype, device=self.device)

    def simulate_mask(self, mask: np.ndarray, grid: Grid, defocus_nm: float = 0.0, dose: float = 1.0,
                      contours: bool = True) -> SimResult:
        with torch.no_grad():
            a = self.aerial(self.to_tensor(mask), grid.pixel, defocus_nm)
            lat = self.latent(a, grid.pixel)
        a, lat = a.cpu().numpy().astype(float), lat.cpu().numpy().astype(float)
        level = self.threshold / dose
        cs = extract_contours(lat, level, grid) if contours else []
        return SimResult(grid=grid, mask=mask, aerial=a, latent=lat, level=level, contours=cs)

    def simulate(self, polygons: list[Polygon], grid: Grid | None = None, defocus_nm: float = 0.0,
                 dose: float = 1.0, contours: bool = True) -> SimResult:
        if grid is None:
            g = self.cfg.grid
            grid = Grid.around(polygons, g.pixel_nm, g.margin_nm)
        if max(grid.shape) > self.cfg.grid.max_pixels:
            raise ValueError(f"clip {grid.shape} exceeds grid.max_pixels={self.cfg.grid.max_pixels}")
        return self.simulate_mask(rasterize(polygons, grid), grid, defocus_nm, dose, contours)


# ==================================================================================================
# 3. Measurements
# ==================================================================================================
#
# Measurements on simulated images: EPE, CD gauges, threshold calibration, through-pitch CD.
#
# Sign convention: an edge displacement / EPE is measured along the *outward*
# normal of the target edge, so EPE > 0 means the printed edge lies outside the
# target (over-print) and EPE < 0 means it is pulled inside (under-print).


class ImageSampler:
    """Cubic-spline interpolation of an image at arbitrary nm coordinates."""

    def __init__(self, image: np.ndarray, grid: Grid):
        self.grid = grid
        self.coef = ndimage.spline_filter(np.asarray(image, dtype=float), order=3)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        rc = self.grid.to_index(pts)
        shape = rc.shape[:-1]
        rc = rc.reshape(-1, 2).T
        return ndimage.map_coordinates(self.coef, rc, order=3, prefilter=False, mode="nearest").reshape(shape)


def edge_displacement(sampler: ImageSampler, level: float, points: np.ndarray, normals: np.ndarray,
                      search_nm: float = 80.0, step_nm: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance along ``normals`` from ``points`` to the printed edge.

    The printed edge is the threshold crossing going from printed (>= level) to
    not printed (< level) in the +normal direction, nearest to the point.
    Returns (displacement, found). Where no crossing exists inside +-search_nm,
    the displacement is clipped to -search_nm (feature missing) or +search_nm
    (merged / bridged) and found is False.
    """
    points, normals = np.atleast_2d(points), np.atleast_2d(normals)
    t = np.arange(-search_nm, search_nm + step_nm / 2, step_nm)
    ray = points[:, None, :] + t[None, :, None] * normals[:, None, :]
    v = sampler(ray) - level                                         # (N, T)
    inside = v >= 0
    cross = inside[:, :-1] & ~inside[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = t[:-1] + step_nm * v[:, :-1] / (v[:, :-1] - v[:, 1:])
    dist = np.where(cross, np.abs(tc), np.inf)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    found = np.isfinite(dist[rows, j])
    disp = np.where(found, tc[rows, j], 0.0)
    v0 = v[:, len(t) // 2]
    disp = np.where(found, disp, np.where(v0 >= 0, search_nm, -search_nm))
    return disp, found


def signed_contour_distance(points: np.ndarray, contours: list[Polygon], sampler: ImageSampler,
                            level: float, resample_nm: float = 0.25) -> np.ndarray:
    """Distance from each point to the nearest printed contour; + if the point is printed, - if not."""
    from scipy.spatial import cKDTree

    dense = []
    for c in contours:
        seg = np.vstack([c, c[:1]])
        for a, b in zip(seg[:-1], seg[1:]):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / resample_nm)))
            dense.append(a + (np.arange(n) / n)[:, None] * (b - a))
    if not dense:
        return np.full(len(points), -np.inf)
    d, _ = cKDTree(np.concatenate(dense)).query(points)
    return np.where(sampler(points) >= level, d, -d)


def measure_epe(samples: "EdgeSamples", sampler: ImageSampler, level: float, contours: list[Polygon],
                search_nm: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """EPE at each edge sample: crossing along the normal, or (where the normal ray misses the
    printed edge, e.g. beside a pulled-back line end) the signed distance to the nearest contour."""
    epe, found = edge_displacement(sampler, level, samples.points, samples.normals, search_nm)
    if not found.all():
        epe[~found] = signed_contour_distance(samples.points[~found], contours, sampler, level)
    return epe, found


# ---------------------------------------------------------------- edge samples
@dataclass
class EdgeSamples:
    points: np.ndarray      # (N, 2) nm, on the target edges
    normals: np.ndarray     # (N, 2) outward unit normals
    poly_id: np.ndarray     # (N,)
    corner_dist: np.ndarray  # (N,) distance along the edge to the nearest vertex


def sample_edges(polygons: list[Polygon], spacing_nm: float = 4.0) -> EdgeSamples:
    polygons = [ensure_ccw(p) for p in polygons]
    key = lambda a, b: (round(a[0], 3), round(a[1], 3), round(b[0], 3), round(b[1], 3))
    edges = {key(a, b) for p in polygons for a, b in zip(p, np.roll(p, -1, axis=0))}
    pts, nrm, pid, cdist = [], [], [], []
    for i, poly in enumerate(polygons):
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            d = b - a
            L = float(np.hypot(*d))
            if L < 1e-9 or key(b, a) in edges:          # skip keyhole cuts (edge paired with its reverse)
                continue
            n = max(1, int(round(L / spacing_nm)))
            s = (np.arange(n) + 0.5) / n
            pts.append(a + s[:, None] * d)
            nrm.append(np.repeat([[d[1] / L, -d[0] / L]], n, axis=0))    # CCW -> outward = rotate edge clockwise
            pid.append(np.full(n, i))
            cdist.append(np.minimum(s, 1 - s) * L)
    return EdgeSamples(np.concatenate(pts), np.concatenate(nrm), np.concatenate(pid), np.concatenate(cdist))


def epe_stats(epe: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    e = epe if mask is None else epe[mask]
    if len(e) == 0:
        return {"n": 0, "mean_abs": np.nan, "rms": np.nan, "max_abs": np.nan, "mean": np.nan}
    return {"n": int(len(e)), "mean_abs": float(np.mean(np.abs(e))), "rms": float(np.sqrt(np.mean(e**2))),
            "max_abs": float(np.max(np.abs(e))), "mean": float(np.mean(e))}


def measure_gauge(g: Gauge, sampler: ImageSampler, level: float, search_nm: float = 80.0) -> tuple[float, bool]:
    p = np.asarray(g.point, dtype=float)
    d = np.asarray(g.direction, dtype=float)
    d = d / np.linalg.norm(d)
    if g.kind == "edge":
        disp, ok = edge_displacement(sampler, level, p, d, search_nm)
        return float(disp[0]), bool(ok[0])
    if g.kind == "cd":
        h = g.target / 2
        disp, ok = edge_displacement(sampler, level, np.array([p + h * d, p - h * d]), np.array([d, -d]), search_nm)
        width = g.target + disp.sum()
        if not ok.all():
            width = 0.0 if disp.sum() < 0 else np.nan
        return float(width), bool(ok.all())
    raise ValueError(g.kind)


# ----------------------------------------------- periodic line/space utilities
def periodic_ls_mask(cd: float, pitch: float, pixel: float, min_width: float = 1000.0, ny: int = 16):
    """Infinite vertical line/space array as a periodic simulation cell.

    The grid width is an integer number of pitches, so FFT periodicity makes
    the array truly infinite. Returns (mask, grid, x_center_of_line0).
    """
    for n in range(max(1, int(np.ceil(min_width / pitch))), 400):
        nx = n * pitch / pixel
        if abs(nx - round(nx)) < 1e-6:
            break
    else:
        raise ValueError(f"no integer pixel count for pitch {pitch} at pixel {pixel}")
    nx = int(round(nx))
    grid = Grid(0.0, 0.0, pixel, nx, ny)
    H = ny * pixel
    lines = [rect(i * pitch + pitch / 2 - cd / 2, 0.0, i * pitch + pitch / 2 + cd / 2, H) for i in range(n)]
    return rasterize(lines, grid), grid, pitch / 2


def periodic_profile(sim, cd: float, pitch: float, defocus_nm: float = 0.0) -> tuple[CubicSpline, float, float]:
    """Latent-image profile across an infinite L/S array, as a periodic spline. Returns (spline, x_center, period)."""
    mask, grid, xc = periodic_ls_mask(cd, pitch, sim.cfg.grid.pixel_nm)
    with torch.no_grad():
        a = sim.aerial(sim.to_tensor(mask), grid.pixel, defocus_nm)
        lat = sim.latent(a, grid.pixel).cpu().numpy().astype(float)
    prof = lat.mean(axis=0)
    x = grid.x_centers()
    period = grid.nx * grid.pixel
    spline = CubicSpline(np.append(x, x[0] + period), np.append(prof, prof[0]), bc_type="periodic")
    return spline, xc, period


def edge_threshold(sim, size_nm: float = 400.0) -> float:
    """Intensity at the edge of a large isolated feature (a 400 nm line at 800 nm pitch): with this threshold
    that edge prints exactly at its drawn position, so remaining deviations are proximity effects."""
    spline, xc, _ = periodic_profile(sim, size_nm, 2 * size_nm)
    return float(spline(xc + size_nm / 2))


def calibrate_threshold(sim, cd: float, pitch: float) -> float:
    """Dose-to-size: the threshold at which the infinite anchor L/S prints exactly at ``cd``."""
    spline, xc, _ = periodic_profile(sim, cd, pitch)
    return float(0.5 * (spline(xc - cd / 2) + spline(xc + cd / 2)))


def periodic_cd(sim, cd: float, pitch: float, threshold: float | None = None, dose: float = 1.0,
                defocus_nm: float = 0.0) -> float:
    """Printed CD of one line in an infinite L/S array (0 if it does not print, pitch if it merges)."""
    level = (sim.threshold if threshold is None else threshold) / dose
    spline, xc, _ = periodic_profile(sim, cd, pitch, defocus_nm)
    f = lambda x: float(spline(x)) - level
    if f(xc) < 0:
        return 0.0
    if f(xc + pitch / 2) >= 0:
        return float(pitch)
    right = brentq(f, xc, xc + pitch / 2, xtol=1e-6)
    left = brentq(f, xc - pitch / 2, xc, xtol=1e-6)
    return float(right - left)
