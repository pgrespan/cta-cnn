import numpy as np
import matplotlib.pyplot as plt
import ctaplot
from astropy.io import fits
import pandas as pd

from gammapy.irf import EnergyDispersion2D


def read_sensitivity(irf_filename):

    with fits.open(irf_filename) as irf:
        t = irf['SENSITIVITY']
        elo = t.data['ENERG_LO']
        ehi = t.data['ENERG_HI']
        sens = t.data['SENSITIVITY']

    return elo, ehi, sens


def read_angular_resolution(irf_filename):
    """

    Parameters
    ----------
    irf_filename

    Returns
    -------
    energy_low, energy_high, angular_resolution
    """

    with fits.open(irf_filename) as irf:
        psf_hdu = irf['POINT SPREAD FUNCTION']
        e_lo = psf_hdu.data['ENERG_LO']
        e_hi = psf_hdu.data['ENERG_HI']
        psf = psf_hdu.data['PSF68']
    return e_lo, e_hi, psf


def read_effective_area(irf_filename):

    with fits.open(irf_filename) as irf:
        elo = irf['SPECRESP'].data['ENERG_LO']
        ehi = irf['SPECRESP'].data['ENERG_HI']
        eff_area = irf['SPECRESP'].data['SPECRESP']
        # eff_area_no_cut = irf['SPECRESP (NO CUTS)'].data['SPECRESP (NO CUTS)']

    return elo, ehi, eff_area

def read_energy_resolution(irf_filename):
    e2d = EnergyDispersion2D.read(irf_filename, hdu='ENERGY DISPERSION')
    edisp = e2d.to_energy_dispersion('0 deg')

    energy_bin = np.logspace(-1.5, 1, 15)

    energy, energy_err = bins_limits_to_errorbars( energy_bin[:-1], energy_bin[1:], log=True)
    e_res = edisp.get_resolution(energy)
    return energy_bin[1:], energy_bin[:-1], e_res

def bins_limits_to_errorbars(x_lo, x_hi, log=False):
    """
    From bins limits, return the mean of the bin and the errorbar on each side

    Parameters
    ----------
    x_lo
    x_hi
    log: optional, if True, a logarithmic bean is computed

    Returns
    -------
    x_mid, x_err: (float, (float, float))
    """

    if not log:
        x_mid = (x_lo + x_hi)/2.
    else:
        x_mid = np.sqrt(x_lo * x_hi)

    x_err = x_mid - x_lo, x_hi - x_mid

    return x_mid, x_err



def plot_sensitivity(energy_lo, energy_hi, sensitivity, ax=None, **kwargs):
    """

    Parameters
    ----------
    energy_lo
    energy_hi
    sensitivity
    ax
    kwargs

    Returns
    -------

    """

    ax = plt.gca() if ax is None else ax

    energy, energy_err = bins_limits_to_errorbars(energy_lo, energy_hi, log=True)

    kwargs.setdefault('fmt', 'o')

    ax.errorbar(energy, sensitivity,
                xerr=energy_err,
                **kwargs
                )

    ax.legend()
    ax.grid(True, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    return ax

def plot_resolution(energy_lo, energy_hi, resolution, ax=None, **kwargs):
    """
    Plot any resolution

    Parameters
    ----------
    energy_lo
    energy_hi
    resolution
    ax
    kwargs

    Returns
    -------

    """

    ax = plt.gca() if ax is None else ax

    energy, energy_err = bins_limits_to_errorbars(energy_lo, energy_hi, log=True)

    kwargs.setdefault('fmt', 'o')

    ax.errorbar(energy, resolution, xerr=energy_err, **kwargs)

    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, which='both')
    ax.set_xlabel('Energy [TeV]')

    return ax


def plot_angular_resolution(energy_lo, energy_hi, angular_resolution, ax=None, **kwargs):
    """

    Parameters
    ----------
    energy_lo
    energy_hi
    angular_resolution
    ax
    kwargs

    Returns
    -------

    """

    ax = plot_resolution(energy_lo, energy_hi, angular_resolution,  ax=ax, **kwargs)
    ax.set_ylabel('Angular resolution [deg]')
    return ax


def plot_energy_resolution(energy_lo, energy_hi, energy_resolution, ax=None, **kwargs):
    """

    Parameters
    ----------
    energy_lo
    energy_hi
    energy_resolution
    ax
    kwargs

    Returns
    -------

    """

    ax = plot_resolution(energy_lo, energy_hi, energy_resolution,  ax=ax, **kwargs)
    ax.set_ylabel('Energy resolution [deg]')
    return ax

def plot_effective_area(energy_lo, energy_hi, effective_area, ax=None, **kwargs):
    """

    Parameters
    ----------
    energy_lo
    energy_hi
    effective_area
    ax
    kwargs

    Returns
    -------

    """

    ax = plt.gca() if ax is None else ax

    energy, energy_err = bins_limits_to_errorbars(energy_lo, energy_hi, log=True)

    if 'label' not in kwargs:
        kwargs['label'] = 'Effective area [m2]'
    else:
        user_label = kwargs['label']
        kwargs['label'] = f'{user_label}'

    kwargs.setdefault('linestyle', '')
    ax.errorbar(energy, effective_area, xerr=energy_err, **kwargs)

    # kwargs['label'] = f"{kwargs['label']} (no cuts)"
    # kwargs['linestyle'] = '--'
    # ax.loglog(energy, eff_area_no_cut, **kwargs)

    ax.legend()
    ax.grid(True, which='both')
    ax.set_xlabel('Energy [TeV]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    return ax


def plot_sensitivity_from_file(irf_filename, ax=None, **kwargs):
    """
    Plot the sensitivity

    Parameters
    ----------
    irf_filename: path
    ax:
    kwargs:

    Returns
    -------
    ax
    """
    ax = plt.gca() if ax is None else ax
    e_lo, e_hi, sens = read_sensitivity(irf_filename)
    return plot_sensitivity(e_lo, e_hi, sens, ax=ax, **kwargs)


def plot_angular_resolution_from_file(irf_filename, ax=None, **kwargs):
    """
    Plot angular resolution from an IRF file

    Parameters
    ----------
    irf_filename
    ax
    kwargs

    Returns
    -------

    """
    ax = plt.gca() if ax is None else ax

    e_lo, e_hi, psf = read_angular_resolution(irf_filename)

    ax = plot_angular_resolution(e_lo, e_hi, psf, ax=ax, **kwargs)
    return ax



def plot_energy_resolution_from_file(irf_filename, ax=None, **kwargs):
    """
    Plot angular resolution from an IRF file
    Parameters
    ----------
    irf_filename
    ax
    kwargs
    Returns
    -------
    """

    ax = plt.gca() if ax is None else ax

    e_lo, e_hi, energy_res = read_energy_resolution(irf_filename)
    ax = plot_energy_resolution(e_lo, e_hi, energy_res, ax=ax, **kwargs)

    return ax



def plot_effective_area_from_file(irf_filename, ax=None, **kwargs):
    """

    Parameters
    ----------
    irf_filename
    ax
    kwargs

    Returns
    -------

    """

    ax = plt.gca() if ax is None else ax

    e_lo, e_hi, effective_area = read_effective_area(irf_filename)
    ax = plot_effective_area(e_lo, e_hi, effective_area, ax=ax, **kwargs)
    return ax


def plot_magic_sensitivity(ax=None, **kwargs):
    """

    """

    ax = plt.gca() if ax is None else ax

    magic = pd.read_csv('magic_sensitivity_50h.csv',
                        names=['energy', 'sensitivity'], delimiter=';', decimal=',')

    kwargs.setdefault('label', 'MAGIC 50h')
    ax.plot(magic.energy, magic.sensitivity, **kwargs)

    return ax