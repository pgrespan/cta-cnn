import ctaplot
import os
import numpy as np
import matplotlib.pyplot as plt
import tables


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



def read_angular_resolution(hdf5_filename, key='No_Cut'):
    """
    """
    with tables.open_file(hdf5_filename) as f:
        e_lo = f.root[key]['E_bin'][:-1]
        e_hi = f.root[key]['E_bin'][1:]
        res = f.root[key]['angular_res'][:]
        
    return e_lo, e_hi, res
        
def read_energy_resolution(hdf5_filename, key='No_Cut'):
    """
    """
    with tables.open_file(hdf5_filename) as f:
        e_lo = f.root[key]['E_bin'][:-1]
        e_hi = f.root[key]['E_bin'][1:]
        res = f.root[key]['energy_res'][:]
        
    return e_lo, e_hi, res
    

    
def plot_angular_resolution_from_file(hdf5_filename, key='No_Cut', ax=None, **kwargs):
    
    e_lo, e_hi, res = read_angular_resolution(hdf5_filename)
    ax = plot_angular_resolution(e_lo, e_hi, np.median(res, axis=1),
                            ax=ax,
                            **kwargs
                           )
    
    mean_energy = bins_limits_to_errorbars(e_lo, e_hi, log=True)[0]
    
    if 'label' in kwargs:
        kwargs.pop('label')
    kwargs.setdefault('alpha', 0.6)
    ax.fill_between(mean_energy, 
                     np.percentile(res, 16, axis=1),
                     np.percentile(res, 84, axis=1),
                     **kwargs
                    )
    
    return ax


def plot_energy_resolution_from_file(hdf5_filename, key='No_Cut', ax=None, **kwargs):
    
    e_lo, e_hi, res = read_energy_resolution(hdf5_filename)
    ax = plot_energy_resolution(e_lo, e_hi, np.median(res, axis=1),
                            ax=ax,
                            **kwargs
                           )
    
    mean_energy = bins_limits_to_errorbars(e_lo, e_hi, log=True)[0]
    
    kwargs.pop('label')
    kwargs.setdefault('alpha', 0.6)
    ax.fill_between(mean_energy, 
                     np.percentile(res, 16, axis=1),
                     np.percentile(res, 84, axis=1),
                     **kwargs
                    )
    
    return ax


    
    