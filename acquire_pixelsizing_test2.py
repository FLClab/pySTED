"""
Il faut que je fix le code du simulateur pour tenir en compte la taille de l'image originale :)
Je vais utiliser ce scripte pour faire mes tests
Pour l'instant, mon plan est de faire une fonction get_signal qui prend un param optionnel de plus, soit la taille
d'un pixel dans l'image originale (données brutes)
Le pixel_size de l'acquisition (du microscope, distance que le laser se déplace entre ses pulses) doit être plus
grand que le pixel_size de l'image originale.
"""

import argparse

from matplotlib import cm, pyplot
import numpy
import tifffile

import time

from pysted import base, utils
from RMS_calculator import rms_calculator

parser = argparse.ArgumentParser(description="Test de script d'acquisition par BT :)")
# param obligatoire du pixel size, je pense que par défault on veut 10e-9
parser.add_argument("--pixelsize", type=float, default=10e-9, help="pixel size (in m) of the acquired image")
# params avec valeurs par défaut
parser.add_argument("--pdt", type=float, default=10e-6, help="pixel dwell time (in s)")
parser.add_argument("--exc", type=float, default=1e-6,  help="excitation power (in W)")
parser.add_argument("--sted", type=float, default=30e-3, help="STED power (in W)")
parser.add_argument("--dpxsz", type=float, default=10e-9, help="Pixel size of raw data")
args = parser.parse_args()
data_pixelsize = args.dpxsz

print("Running pixel sizing test, second implementation...")

datamap = tifffile.imread("examples/data/fibres.tif")
# normalize the datamap, 3 molecules/pixel at most
datamap = datamap / numpy.amax(datamap) * 3

# paramètres du microscope j'imagine ?
egfp = {"lambda_": 535e-9,
        "qy": 0.6,
        "sigma_abs": {488: 1.15e-20,
                      575: 6e-21},
        "sigma_ste": {560: 1.2e-20,
                      575: 6.0e-21,
                      580: 5.0e-21},
        "sigma_tri": 1e-21,
        "tau": 3e-09,
        "tau_vib": 1.0e-12,
        "tau_tri": 5e-6,
        "phy_react": {488: 1e-4,
                      575: 1e-8},
        "k_isc": 0.26e6}

laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, distortion=0.04)
detector = base.Detector(noise=True)   # background=10e6
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)

numpy.random.seed(1)
signal_most_basic = microscope.get_signal(datamap, args.pixelsize, args.pdt, args.exc, args.sted)

microscope.clear_cache()

numpy.random.seed(1)
signal_base = microscope.get_signal_pxsize_test(datamap, args.pixelsize, args.pdt, args.exc, args.sted,
                                                data_pixelsize=data_pixelsize)

if signal_most_basic.shape == signal_base.shape:
    rms = rms_calculator(signal_most_basic, signal_base)
    print(f"rms = {rms}")


def pix2meters_acq(x):
    return x * args.pixelsize


def pix2meters_raw(x):
    return x * data_pixelsize


def meters2pix_acq(x):
    return x / args.pixelsize


def meters2pix_raw(x):
    return x / data_pixelsize


fig, axes = pyplot.subplots(1, 3)

basedata_imshow = axes[0].imshow(datamap, interpolation="nearest",
                                 extent=[0, data_pixelsize * datamap.shape[0], data_pixelsize * datamap.shape[1], 0])
axes[0].set_title(f"Base data")
axes[0].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
axes[0].set_xlabel("position [m]")
axes[0].set_ylabel("position [m]")
secxax_rawdata = axes[0].secondary_xaxis('top', functions=(meters2pix_raw, pix2meters_raw))
secyax_rawdata = axes[0].secondary_yaxis('right', functions=(meters2pix_raw, pix2meters_raw))
secxax_rawdata.set_xlabel("position [pixel]")
secyax_rawdata.set_ylabel("position [pixel]")
fig.colorbar(basedata_imshow, ax=axes[0], fraction=0.04, pad=0.2)

def_acquisition_imshow = axes[1].imshow(signal_most_basic, interpolation="nearest",
                                        extent=[0, args.pixelsize * signal_most_basic.shape[0],
                                                  args.pixelsize * signal_most_basic.shape[1], 0])
# les axes de l'image default acquisition ne devraient pas faire de sens si j'essaie de convertir en m
axes[1].set_title(f"Default acquisition")
axes[1].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
axes[1].set_xlabel("position [m] (wrong value)")
axes[1].set_ylabel("position [m] (wrong value)")
secxax_baseacq = axes[1].secondary_xaxis('top', functions=(meters2pix_acq, pix2meters_acq))
secyax_baseacq = axes[1].secondary_yaxis('right', functions=(meters2pix_acq, pix2meters_acq))
secxax_baseacq.set_xlabel("position [pixel]")
secyax_baseacq.set_ylabel("position [pixel]")
fig.colorbar(def_acquisition_imshow, ax=axes[1], fraction=0.04, pad=0.2)

mod_acquisition_imshow = axes[2].imshow(signal_base, interpolation="nearest",
                                        extent=[0, args.pixelsize * signal_base.shape[0],
                                                args.pixelsize * signal_base.shape[1], 0])
axes[2].set_title(f"Modified acquisition")
axes[2].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
axes[2].set_xlabel("position [m]")
axes[2].set_ylabel("position [m]")
secxax_modacq = axes[2].secondary_xaxis('top', functions=(meters2pix_acq, pix2meters_acq))
secyax_modacq = axes[2].secondary_yaxis('right', functions=(meters2pix_acq, pix2meters_acq))
secxax_modacq.set_xlabel("position [pixel]")
secyax_modacq.set_ylabel("position [pixel]")
fig.colorbar(mod_acquisition_imshow, ax=axes[2], fraction=0.04, pad=0.2)   # pad=0.04

# pyplot.tight_layout()
pyplot.show()
