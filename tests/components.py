
import IPython

import matplotlib.pyplot as pyplot
import numpy

from pysted import simulator, utils
from pysted import base, simulation

def plot(values):
    pyplot.figure()
    pyplot.plot(values)
#    pyplot.show()

def imshow(values):
    pyplot.figure()
    pyplot.imshow(values)
    pyplot.colorbar()
    pyplot.show()

na = 1.4
n = 1.5
f = 2e-3
obj_trans = {488e-9: 0.84,
             535e-9: 0.85,
             550e-9: 0.86,
             585e-9: 0.85,
             575e-9: 0.85}

pixelsize = 10e-9

laser_ex = simulator.LaserExcitation(488e-9)
i_ex = laser_ex.get_intensity(1e-6, f, n, na, obj_trans, pixelsize)
idx_mid = int((i_ex.shape[0]-1) / 2)
print("FWHM excitation:", utils.fwhm(i_ex[idx_mid]))
plot(i_ex[idx_mid])

laser_sted = simulator.LaserSTED(575e-9)
i_sted = laser_sted.get_intensity(30e-3, f, n, na, obj_trans, pixelsize)
idx_mid = int((i_sted.shape[0]-1) / 2)
print("FWHM STED:", utils.fwhm_donut(i_sted[idx_mid]))
plot(i_sted[idx_mid])

fluo = simulator.Fluorescence(535e-9)
psf = fluo.get_psf(na, obj_trans, pixelsize)
idx_mid = int((psf.shape[0]-1) / 2)
print("FWHM PSF:", utils.fwhm(psf[idx_mid]))
plot(psf[idx_mid])

#microscope = simulation.Microscope(pixelsize)

#i_ex2 = microscope.get_excitation_intensity(488e-9, 1e-6)
#idx_mid = int((i_ex2.shape[0]-1) / 2)
#print("FWHM excitation 2:", utils.fwhm(i_ex2[idx_mid]))
#plot(i_ex2[idx_mid])

#i_sted2 = microscope.get_depletion_intensity(575e-9, 30e-3)
#idx_mid = int((i_sted2.shape[0]-1) / 2)
#print("FWHM STED 2:", utils.fwhm_donut(i_sted2[idx_mid]))
#plot(i_sted2[idx_mid])

#psf2 = microscope.get_detection_psf(535e-9)
#idx_mid = int((psf2.shape[0]-1) / 2)
#print("FWHM PSF 2:", utils.fwhm(psf2[idx_mid]))
#plot(psf2[idx_mid])

pyplot.show()

