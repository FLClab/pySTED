"""
------------------------- BUG DESCRIPTION -------------------------
I would expect that changing the decay_time_us would change the slope of the flash
The lower the day_time_us, the faster the flash should go from peak back to normal,
however, changing this value does not seem to change anything
-------------------------------------------------------------------
"""

import numpy as np
from matplotlib import pyplot as plt
from pysted import base, utils


print("Setting up the microscope ...")
# Microscope stuff
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
        "phy_react": {488: 1e-6,   # 1e-4
                      575: 1e-10},   # 1e-8
        "k_isc": 0.26e6}
pixelsize = 20e-9
bleach = False
p_ex_val = 1e-6
p_sted_val = 30e-3
pdt_val = 10e-6
# pdt_val = np.random.randint(0, 100, size=(64, 64)) * 1e-6

# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, _, _ = microscope.cache(pixelsize, save_cache=True)
molec_disp = np.ones((64, 64))
molec_disp[29:34, 29:34] = 3
temporal_dmap_1 = base.TestTemporalDmap(molec_disp, pixelsize)
temporal_dmap_1.set_roi(i_ex, intervals="max")
decay_time_us_1 = 1000000  # 1 seconde
temporal_dmap_1.create_t_stack_dmap(decay_time_us_1)

for i in range(temporal_dmap_1.flash_tstack.shape[0]):
    temporal_dmap_1.update_whole_datamap(i)
    plt.imshow(temporal_dmap_1.whole_datamap)
    plt.title(f"i = {i}, max = {np.max(temporal_dmap_1.whole_datamap)}")
    plt.show()

temporal_dmap_2 = base.TestTemporalDmap(molec_disp, pixelsize)
temporal_dmap_2.set_roi(i_ex, intervals="max")
decay_time_us_2 = 100  # 1 seconde
temporal_dmap_2.create_t_stack_dmap(decay_time_us_1)

for i in range(temporal_dmap_2.flash_tstack.shape[0]):
    temporal_dmap_2.update_whole_datamap(i)
    plt.imshow(temporal_dmap_2.whole_datamap)
    plt.title(f"i = {i}, max = {np.max(temporal_dmap_2.whole_datamap)}")
    plt.show()


