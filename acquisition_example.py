"""
Code written by Benoit Turcotte, benoit.turcotte.4@ulaval.ca, October 2020
For use by FLClab (@CERVO) authorized people
"""

# Import packages
import argparse

from matplotlib import pyplot, image
import numpy
import tifffile
import os, datetime
from tkinter.filedialog import askopenfilename

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
parser.add_argument("--pixelsize", type=float, default=20e-9, help="Displacement of laser between pulses. Must be a "
                                                                   "multiple of the datamap_pixelsize, 10 nm. (m)")
parser.add_argument("--bleach", type=str2bool, default=False, help="Determines wether bleaching is active or not.")
parser.add_argument("--seed", type=int, default=None, help="Used to seed the acquisitions if wanted")
parser.add_argument("--select_datamap", type=str2bool, default=False, help="If true, will open a file explorer for you"
                                                                           "to select a .tif file as the datamap. The"
                                                                           ".tif file will then be normalized to contain"
                                                                           "between 0 and 5 molecules per pixel.")
parser.add_argument("--shape", type=int, default=100, help="An int used to generate a square datamap with axes of"
                                                           "this length.")
parser.add_argument("--sources", type=int, default=100, help="Number of fluorophore sources to be added in the datamap")
parser.add_argument("--molecules", type=int, default=5, help="Average number of fluorescent molecules per source in "
                                                             "the datamap")
args = parser.parse_args()

print("Starting acquisition...")

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
# zero_residual controls how much of the donut beam "bleeds" into the the donut hole
laser_sted = base.DonutBeam(575e-9, zero_residual=args.zero_residual)
# noise allows noise on the detector, background adds an average photon count for the empty pixels
detector = base.Detector(noise=True, background=args.background)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
datamap_pixelsize = 20e-9
utils.pxsize_comp2(args.pixelsize, datamap_pixelsize)

# this loads the datamap
if args.select_datamap:
    filename = askopenfilename(initialdir="examples/data/", title="Select a .tif file",
                               filetypes=[("tif files", "*.tif *.tiff")])
    molecules_disposition = tifffile.imread(filename)
    # normalize the datamap, 5 molecules/pixel at most
    molecules_disposition = (molecules_disposition / numpy.amax(molecules_disposition) * 5).astype(int)
else:
    molecules_disposition = utils.datamap_generator(args.shape, args.sources, args.molecules, random_state=args.seed)
molecules_disposition = molecules_disposition.astype(numpy.int32)
datamap = base.Datamap(molecules_disposition, datamap_pixelsize)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, datamap)

# This function pre-generates the excitation and STED beams, allowing you to visualize them if you wish
i_ex, i_sted, psf_det = microscope.cache(datamap_pixelsize)

# expliquer ça là :)
signal_confocal = microscope.get_signal_and_bleach(args.pixelsize, args.pdt, args.exc, 0, pixel_list=None,
                                                   bleach=args.bleach)
confocal_bleached = numpy.copy(datamap.whole_datamap)

datamap.whole_datamap = numpy.copy(molecules_disposition)
signal_sted = microscope.get_signal_and_bleach(args.pixelsize, args.pdt, args.exc, args.sted, pixel_list=None,
                                               bleach=args.bleach)
sted_bleached = numpy.copy(datamap.whole_datamap)

# save stuff as tiff files
if not os.path.exists("acquisitions"):
    os.mkdir("acquisitions")
new_acq_dir = "acquisitions/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.mkdir(new_acq_dir)
image.imsave(new_acq_dir + "/datamap.tiff", molecules_disposition)
image.imsave(new_acq_dir + "/signal_confocal.tiff", signal_confocal)
image.imsave(new_acq_dir + "/signal_sted.tiff", signal_sted)
if args.bleach:
    image.imsave(new_acq_dir + "/bleached_datamap_confocal.tiff", confocal_bleached)
    image.imsave(new_acq_dir + "/bleached_datamap_sted.tiff", sted_bleached)

# Plot the original datamap, acquired signal and bleached datamap

fig, axes = pyplot.subplots(1, 3)

datamap_imshow = axes[0].imshow(molecules_disposition)
axes[0].set_title(f"Datamap, shape = {molecules_disposition.shape}")
fig.colorbar(datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

confocal_imshow = axes[1].imshow(signal_confocal)
axes[1].set_title(f"Confocal signal, shape = {signal_confocal.shape}")
fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

sted_imshow = axes[2].imshow(signal_sted)
axes[2].set_title(f"STED signal, shape = {signal_sted.shape}")
fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

pyplot.show()

if args.bleach:
    confocal_ratio = utils.molecules_survival(molecules_disposition, confocal_bleached)
    sted_ratio = utils.molecules_survival(molecules_disposition, sted_bleached)

    fig, axes = pyplot.subplots(1, 3)

    datamap_imshow = axes[0].imshow(molecules_disposition)
    axes[0].set_title(f"Datamap, shape = {molecules_disposition.shape}")
    fig.colorbar(datamap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

    confocal_bleach_imshow = axes[1].imshow(confocal_bleached)
    axes[1].set_title(f"Bleached datamap after confocal acquisition \n"
                      f"{round(100 * confocal_ratio, 3)} % of molecules survived")
    fig.colorbar(confocal_bleach_imshow, ax=axes[1], fraction=0.04, pad=0.05)

    sted_bleach_imshow = axes[2].imshow(sted_bleached)
    axes[2].set_title(f"Bleached datamap after STED acquisition, \n"
                      f"{round(100 * sted_ratio, 3)} % of molecules survived")
    fig.colorbar(sted_bleach_imshow, ax=axes[2], fraction=0.04, pad=0.05)

    pyplot.show()
