"""Physical constants (SI) and unit helpers."""
import numpy as np

H = 6.62607015e-34        # J s
C = 2.99792458e8          # m/s
KB = 1.380649e-23         # J/K
AU = 1.495978707e11       # m
PC = 3.0856775814913673e16  # m
JY = 1e-26                # W m^-2 Hz^-1
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
HC_K = H * C / KB         # m K   (E[K] = HC_K * wavenumber[1/m])
CM1_TO_K = HC_K * 100.0   # K per cm^-1 = 1.4387769


def planck_nu(wave_um, T):
    """Planck function B_nu(T) in SI (W m^-2 Hz^-1 sr^-1) on a wavelength grid in micron."""
    lam = np.asarray(wave_um, dtype=float) * 1e-6
    x = H * C / (lam * KB * T)
    # guard overflow for very cold / short wavelengths
    x = np.minimum(x, 700.0)
    return 2.0 * H * C / lam**3 / np.expm1(x)


def omega_from_radius(R_au, d_pc):
    """Solid angle of a face-on disk of radius R [au] at distance d [pc] (sr)."""
    return np.pi * (R_au * AU) ** 2 / (d_pc * PC) ** 2


def radius_from_area(A_au2):
    return np.sqrt(np.asarray(A_au2) / np.pi)
