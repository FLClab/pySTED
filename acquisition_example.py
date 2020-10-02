"""
Faire un scripte d'acquisition le plus simple possible avec des explications claires de qui fait quoi
"""

# Import packages
import argparse

from matplotlib import pyplot, image
import numpy
import tifffile
import os, datetime

from pysted import base, utils
from hidden_vars import *


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# This allows for a simple way to change parametres when calling the script from the command line
# To modify the value of the pdt parametre for example, one would type
# python acquisition_example.py --pdt 20e-6
parser = argparse.ArgumentParser(description="Exemple de scripte d'acquisition")
parser.add_argument("--pdt", type=float, default=10e-6, help="pixel dwell time (in s)")
parser.add_argument("--exc", type=float, default=1e-6,  help="excitation power (in W)")
parser.add_argument("--sted", type=float, default=30e-3, help="STED power (in W)")
parser.add_argument("--zero_residual", type=float, default=0, help="Fraction of the doughnut beam that bleeds into"
                                                                   "the centre (between 0 and 1)")
parser.add_argument("--background", type=int, default=0, help="Background photons")
parser.add_argument("--pixelsize", type=float, default=10e-9, help="Size of a pixel (m)")
parser.add_argument("--bleach", type=str2bool, default=False, help="Determines wether bleaching is active or not.")
parser.add_argument("--seed", type=str2bool, default=False, help="Used to seed the acquisitions if wanted")
args = parser.parse_args()

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=args.zero_residual)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
detector = base.Detector(noise=True, background=args.background)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)

# this loads the datamap from the file
datamap = tifffile.imread("examples/data/fibres.tif")
# normalize the datamap, 3 molecules/pixel at most
datamap = datamap / numpy.amax(datamap) * 3
# datamap = utils.datamap_generator(100, 100, 3)

# This function allows you to view the excitation beam, STED beam and detection PSF profiles
i_ex, i_sted, psf_det = microscope.cache(args.pixelsize, args.pixelsize)

# This is usefull if you wish to control the randomness in between acquisitions (present in photon emisson and
# bleaching) in order to compare.
if args.seed is True:
    numpy.random.seed(1)

# expliquer ça là :)
signal_confocal, bleached_datamap_confocal = microscope.get_signal_and_bleach(datamap, args.pixelsize, args.pixelsize,
                                                                              args.pdt, args.exc, 0, pixel_list=None,
                                                                              bleach=args.bleach)

signal_sted, bleached_datamap_sted = microscope.get_signal_and_bleach(datamap, args.pixelsize, args.pixelsize, args.pdt,
                                                                      args.exc, args.sted, pixel_list=None,
                                                                      bleach=args.bleach)

# save stuff as tiff files
if not os.path.exists("acquisitions"):
    os.mkdir("acquisitions")
new_acq_dir = "acquisitions/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.mkdir(new_acq_dir)
image.imsave(new_acq_dir + "/datamap.tiff", datamap)
image.imsave(new_acq_dir + "/signal_confocal.tiff", signal_confocal)
image.imsave(new_acq_dir + "/signal_sted.tiff", signal_sted)
if args.bleach:
    image.imsave(new_acq_dir + "/bleached_datamap_confocal.tiff", bleached_datamap_confocal)
    image.imsave(new_acq_dir + "/bleached_datamap_sted.tiff", bleached_datamap_sted)

# Plot the original datamap, acquired signal and bleached datamap

fig, axes = pyplot.subplots(1, 3)

datamap_imshow = axes[0].imshow(datamap)
axes[0].set_title(f"Datamap, shape = {datamap.shape}")
fig.colorbar(datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

confocal_imshow = axes[1].imshow(signal_confocal)
axes[1].set_title(f"Confocal signal, shape = {signal_confocal.shape}")
fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

sted_imshow = axes[2].imshow(signal_sted)
axes[2].set_title(f"STED signal, shape = {signal_sted.shape}")
fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

pyplot.show()

if args.bleach:
    fig, axes = pyplot.subplots(1, 3)

    datamap_imshow = axes[0].imshow(datamap)
    axes[0].set_title(f"Datamap, shape = {datamap.shape}")
    fig.colorbar(datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

    confocal_bleach_imshow = axes[1].imshow(bleached_datamap_confocal)
    axes[1].set_title(f"Bleached datamap after confocal acquisition")
    fig.colorbar(confocal_bleach_imshow, ax=axes[1], fraction=0.04, pad=0.05)

    sted_bleach_imshow = axes[2].imshow(bleached_datamap_sted)
    axes[2].set_title(f"Bleached datamap after STED acquisition")
    fig.colorbar(sted_bleach_imshow, ax=axes[2], fraction=0.04, pad=0.05)

pyplot.show()
