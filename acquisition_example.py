"""
Faire un scripte d'acquisition le plus simple possible avec des explications claires de qui fait quoi
"""

# Import packages
import argparse

from matplotlib import pyplot
import numpy
import tifffile

from pysted import base, utils


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
parser.add_argument("--seed", type=str2bool, default=False, help="Used to seed the acquisitions if wanted")
args = parser.parse_args()

# fluorescence parameters
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

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=0.04)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)

# this loads the datamap from the file
datamap = tifffile.imread("examples/data/fibres.tif")
# normalize the datamap, 3 molecules/pixel at most
datamap = datamap / numpy.amax(datamap) * 3

# pixelsize represents the distance the laser moves between each application
pixelsize = 10e-9
# datamap_pixelsize represents the size of a pixel in the datamap
datamap_pixelsize = 10e-9
# For most applications, you will want pixelsize and datamap_pixelsize to be the same value. If you wish otherwise,
# pixelsize will need to be a multiple of datamap_pixelsize.

# This function allows you to view the excitation beam, STED beam and detection PSF profiles
i_ex, i_sted, psf_det = microscope.cache(pixelsize, datamap_pixelsize)

# This is usefull if you wish to control the randomness in between acquisitions (present in photon emisson and
# bleaching) in order to compare.
if args.seed is True:
    numpy.random.seed(1)

# This line acquires signal from the datamap
# the function call looks like
# signal = microscope.get_signal(datamap, pixelsize, pdt, p_ex, p_sted, datamap_pixelsize)
# datamap is the molecule disposition which we want to image
# pixelsize is the distance the laser moves between applications
# pdt is the pixeldwelltime. It can either be a single value (constant pixeldwelltime accross acquisition) or an array
# (specific pixeldwelltimes for each pixel). If you want to make an array for the pixeldwelltime, make sure it has the
# same shape as the datamap.
# p_ex is the power of the excitation beam
# p_sted is the power of the sted beam
# datamap_pixelsize is the size of a pixel of the datamap
signal = microscope.get_signal(datamap, pixelsize, args.pdt, args.exc, args.sted, datamap_pixelsize)

# This line returns a bleached version of the datamap
bleached_datamap = microscope.bleach(datamap, pixelsize, args.pdt, args.exc, args.sted, datamap_pixelsize)

# Plot the original datamap, acquired signal and bleached datamap

fig, axes = pyplot.subplots(1, 3)

datamap_imshow = axes[0].imshow(datamap)
axes[0].set_title(f"Datamap, shape = {datamap.shape}")
fig.colorbar(datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

signal_imshow = axes[1].imshow(signal)
axes[1].set_title(f"Acquired signal, shape = {signal.shape}")
fig.colorbar(signal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

bleached_imshow = axes[2].imshow(bleached_datamap)
axes[2].set_title(f"Bleached datamap, shape = {bleached_datamap.shape}")
fig.colorbar(bleached_imshow, ax=axes[2], fraction=0.04, pad=0.05)

pyplot.show()
