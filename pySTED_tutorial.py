import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg


"""
This script will go over the basics of pySTED for simulation of confocal and STED acquisitions
on simulated samples.
In order to simulate an acquisition, we need a microscope and a sample. To build the STED microscope, we need an
excitation beam, a STED beam, a detector and the parameters of the fluorophores used in the sample. The class
code for the objects that make up the microscope and the sample are contained in pysted.base
Each object has parameters which can be tuned, which will affect the resulting acquisition
"""

print("Setting up the microscope...")
# Fluorophore properties
egfp = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {
        488: 0.08e-21,
        575: 0.02e-21
    },
    "sigma_ste": {
        575: 3.0e-22,
    },
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 1.2e-6,
    "k1": 1.3e-15, # Atto640N, Oracz2017
    "b":1.4, # Atto640N, Oracz2017
    "triplet_dynamics_frac": 0,
}

pixelsize = 20e-9
# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)

# These are the parameter ranges our RL agents can select from when playing actions
action_spaces = {
    "p_sted" : {"low" : 0., "high" : 175e-3}, # Similar to the sted in our lab
    "p_ex" : {"low" : 0., "high" : 150e-6}, # Similar to the sted in our lab
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

# Example values of parameters used when doing a STED acquisition
sted_params = {
    "pdt": action_spaces["pdt"]["low"] * 2,
    "p_ex": action_spaces["p_ex"]["high"] * 0.6,
    "p_sted": action_spaces["p_sted"]["high"] * 0.6
}

# Example values of parameters used when doing a Confocal acquisition. Confocals always have p_sted = 0
conf_params = {
    "pdt": action_spaces["pdt"]["low"],
    "p_ex": action_spaces["p_ex"]["high"] * 0.6,
    "p_sted": 0.0   # params have to be floats to pass the C function
}

# generate the microscope from its constituent parts
# if load_cache is true, it will load the previously generated microscope. This can save time if a
# microscope was previsously generated and used the same pixelsize we are using now
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=True)
psf_conf = microscope.get_effective(pixelsize, action_spaces["p_ex"]["high"], 0.0)
psf_sted = microscope.get_effective(pixelsize, action_spaces["p_ex"]["high"], action_spaces["p_sted"]["high"] * 0.25)

# You can uncomment these lines to visualize the simulated excitation and STED beams, as well as the
# detection PSFs when using certain excitation / STED power combinations
# fig, axes = plt.subplots(2, 2)
#
# axes[0, 0].imshow(i_ex)
# axes[0, 0].set_title(f"Excitation beam")
#
# axes[0, 1].imshow(i_sted)
# axes[0, 1].set_title(f"STED beam")
#
# axes[1, 0].imshow(psf_conf)
# axes[1, 0].set_title(f"Detection PSF in confocal modality")
#
# axes[1, 1].imshow(psf_sted)
# axes[1, 1].set_title(f"Detection PSF in STED modality")
#
# plt.tight_layout()
# plt.show()

# we now need a sample on to which to do our acquisition, which we call the datamap
# I will show how to build a simple datamap, along with a more complex one which includes nanostructures and a
# temporal element
# First, we use the Synapse class in exp_data_gen to simulate a synapse-like structure and add nanostructures to it
# You could use any integer-valued array as a Datamap
shroom1 = dg.Synapse(5, mode="mushroom", seed=42)

n_molecs_in_domain1, min_dist1 = 135, 50
shroom1.add_nanodomains(10, min_dist_nm=min_dist1, n_molecs_in_domain=n_molecs_in_domain1, valid_thickness=7)

# create the Datamap and set its region of interest
dmap = base.Datamap(shroom1.frame, pixelsize)
dmap.set_roi(i_ex, "max")

shroom2 = dg.Synapse(5, mode="mushroom", seed=42)
n_molecs_in_domain2, min_dist2 = 0, 50
shroom2.add_nanodomains(10, min_dist_nm=min_dist2, n_molecs_in_domain=n_molecs_in_domain2, valid_thickness=7)

# create a temporal Datamap which will also contain information on the positions of nanodomains
# We create a temporal element by making the nanostructures flash
# We then set its temporal index to be at the flash peak
time_idx = 2
temp_dmap = base.TemporalSynapseDmap(shroom2.frame, pixelsize, shroom2)
temp_dmap.set_roi(i_ex, "max")
temp_dmap.create_t_stack_dmap(2000000)
temp_dmap.update_whole_datamap(time_idx)
temp_dmap.update_dicts({"flashes": time_idx})

# you can uncomment this code to see both datamaps, which should look similar
# fig, axes = plt.subplots(1, 2)
#
# axes[0].imshow(dmap.whole_datamap[dmap.roi])
# axes[0].set_title(f"Base Datamap")
#
# axes[1].imshow(temp_dmap.whole_datamap[temp_dmap.roi])
# axes[1].set_title(f"Datamap with temporal element")
#
# plt.show()

# uncomment this code to run through the flash
# for t in range(temp_dmap.flash_tstack.shape[0]):
#     temp_dmap.update_whole_datamap(t)
#     temp_dmap.update_dicts({"flashes": t})
#
#     plt.imshow(temp_dmap.whole_datamap[temp_dmap.roi])
#     plt.title(f"Time idx = {t}")
#     plt.show()

# Now let's show a confocal acquisition and a STED acquisition on the datamaps
# The returns are :
# (1) The acquired image signal
# (2) The bleached datamaps
# (3) The acquired intensity. This is only useful when working in a temporal exeperiment setting, in which
#     an acquisition could be interrupted by the flash happening through it.


conf_acq, conf_bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                              bleach=True, update=True)
conf_acq2, conf_bleached2, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                              bleach=True, update=True)
sted_acq, sted_bleached, _ = microscope.get_signal_and_bleach(temp_dmap, temp_dmap.pixelsize, **sted_params,
                                                              bleach=True, update=True)
sted_acq2, sted_bleached2, _ = microscope.get_signal_and_bleach(temp_dmap, temp_dmap.pixelsize, **sted_params,
                                                              bleach=True, update=True)




fig, axes = plt.subplots(2, 2)

vmax = conf_acq.max()
axes[0,0].imshow(conf_acq, vmax=vmax)
axes[0,0].set_title(f"Confocal 1")

axes[0,1].imshow(conf_acq2, vmax=vmax)
axes[0,1].set_title(f"Confocal 2")

vmax = sted_acq.max()
axes[1,0].imshow(sted_acq, vmax=vmax)
axes[1,0].set_title(f"STED 1")


axes[1,1].imshow(sted_acq2, vmax=vmax)
axes[1,1].set_title(f"STED 2")

plt.suptitle("The four images where acquired sequentially. \nSame normalization on each row")

plt.show()

# I have set the bleaching to false in these acquisitions for speed. You can set it to True to see its effects
# on the acquired signal and the datamaps. You can also of course modify other parameters to see their effects
# on the acquired images. :)
