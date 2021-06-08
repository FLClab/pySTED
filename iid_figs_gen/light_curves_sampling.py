import numpy as np
import tqdm
from pysted import base, utils
import os
import argparse
from matplotlib import pyplot as plt


curves_path = "flash_files/events_curves.npy"
event_curves = np.load(curves_path)

avg_light_curve, std_light_curve = utils.get_avg_lightcurve(event_curves)

x = np.linspace(0, avg_light_curve.shape[0] * 0.15, 40)

plt.style.use('dark_background')
plt.plot(x, avg_light_curve)
plt.fill_between(x, avg_light_curve + std_light_curve, avg_light_curve - std_light_curve, alpha=0.4)
plt.ylabel('Photon count [-]')
plt.xlabel('Times [s]')
plt.show()
