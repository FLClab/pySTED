
import time
import numpy

from matplotlib import pyplot
from tqdm import tqdm

from pysted import base, utils, raster, exp_data_gen

############################################
############################################
# Defaults
############################################
############################################

P_STED = 35.0e-3
P_EX = 25.0e-6
PDT = 10.0e-6

LASER_EX = {"lambda_" : 488e-9}
LASER_STED = {"lambda_" : 575e-9, "zero_residual" : 0.01}
DETECTOR = {"noise" : True}
OBJECTIVE = {}
FLUO = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {488: 4.5e-21,   # was 1.15e-20
                  575: 6e-21},
    "sigma_ste": {560: 1.2e-20,
                  575: 3.0e-22,   # was 6.0e-21
                  580: 5.0e-21},
    "sigma_tri": 1e-21,
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 5e-6,
    "phy_react": {488: 0.25e-7,   # 1e-4
                  575: 25.0e-11},   # 1e-8
    "k_isc": 0.26e6
}
# Best for now
FLUO = {
    "lambda_": 535e-9,
    "qy": 0.6,
    "sigma_abs": {
        488: 0.08e-21,   # was 1.15e-20
        575: 0.02e-21
    },
    "sigma_ste": {
        575: 3.0e-22,   # was 6.0e-21
    },
    "sigma_tri": 10.14e-21, # 1e-21
    "tau": 3e-09,
    "tau_vib": 1.0e-12,
    "tau_tri": 1.2e-6,
    "phy_react": {488: 0.008e-5,   # 1e-4, 12e-7
                  575: 0.008e-8},   # 1e-8
    "k_isc": 0.48e+6
}

# default
# "phy_react": {488: 0.008e-5,
#               575: 0.008e-8},
# min 488
# "phy_react": {488: 0.001e-5,
#               575: 0.008e-8},
# max 488
# "phy_react": {488: 0.015e-5,
#               575: 0.008e-8},
# min 575
# "phy_react": {488: 0.008e-5,
#               575: 0.004e-8},
# max 575
# "phy_react": {488: 0.008e-5,
#               575: 0.012e-8},

# FLUO = {
#     "lambda_": 535e-9,
#     "qy": 0.6,
#     "sigma_abs": {
#         488: 0.2e-21,   # was 1.15e-20
#         575: 0.08e-21
#     },
#     "sigma_ste": {
#         575: 3.0e-22,   # was 6.0e-21
#     },
#     "sigma_tri": 10.14e-21, # 1e-21
#     "tau": 3e-09,
#     "tau_vib": 1.0e-12,
#     "tau_tri": 1.2e-6,
#     "phy_react": {488: 0.075e-26 ** 2,   # 1e-4, 12e-7
#                   575: 0.000015e-26 ** 1.5},   # 1e-8
#     "k_isc": 0.48e+6
# }

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    # "p_ex" : {"low" : 0., "high" : 100.0e-6},
    "p_ex" : {"low" : 0., "high" : 250.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

percents_p_sted = [0.00, 0.10, 0.30, 0.50]
percents_p_ex = [0.05, 0.25, 0.5, 0.75, 0.95]
percents_pdt = [0., 0.1, 0.25, 0.5, 0.75, 1.0][:3]

def generate_example():
    for percent_pdt in percents_pdt:
        pdt = percent_pdt * (action_spaces["pdt"]["high"] - action_spaces["pdt"]["low"]) + action_spaces["pdt"]["low"]
        print(f"pdt : {pdt * 1e+6:0.4f} µs")

        fig, axes = pyplot.subplots(len(percents_p_sted), len(percents_p_ex), sharex=True, sharey=True, tight_layout=True, figsize=(10,10))

        acquisitions = []
        for percent_p_sted in tqdm(percents_p_sted):
            for percent_p_ex in tqdm(percents_p_ex, leave=False):

                pixelsize = 20e-9
                bleach = True
                p_ex = percent_p_ex * (action_spaces["p_ex"]["high"] - action_spaces["p_ex"]["low"]) + action_spaces["p_ex"]["low"]
                p_sted = percent_p_sted * (action_spaces["p_sted"]["high"] - action_spaces["p_sted"]["low"]) + action_spaces["p_sted"]["low"]
                pdt = percent_pdt * (action_spaces["pdt"]["high"] - action_spaces["pdt"]["low"]) + action_spaces["pdt"]["low"]
                roi = 'max'

                shroom = exp_data_gen.Synapse(5, mode="mushroom", seed=42)
                n_molecs_in_domain = 125
                min_dist = 100
                shroom.add_nanodomains((5, 10), min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain, seed=42, valid_thickness=3)
                molecules_disposition = shroom.frame.astype(int)
                molecules_disposition = molecules_disposition[:, :]

                # Generating objects necessary for acquisition simulation
                laser_ex = base.GaussianBeam(**LASER_EX)
                laser_sted = base.DonutBeam(**LASER_STED)
                detector = base.Detector(**DETECTOR)
                objective = base.Objective(**OBJECTIVE)
                fluo = base.Fluorescence(**FLUO)
                datamap = base.Datamap(molecules_disposition, pixelsize)
                microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
                i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
                datamap.set_roi(i_ex, roi)

                # print(f'starting acq with phy_react = {FLUO["phy_react"]}')
                # print(f"p_sted : {p_sted * 1e+3:0.4f} mW")
                # print(f"p_ex : {p_ex * 1e+6:0.4f} µW")
                # print(f"pdt : {pdt * 1e+6:0.4f} µs")
                time_start = time.time()
                acquisition, bleached, intensity = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                                                    bleach=True, update=True, seed=42)
                # print("Acquisition : ", acquisition[molecules_disposition > 0].mean())
                bleach = (molecules_disposition.sum() - bleached["base"].sum()) / molecules_disposition.sum()

                # print(f"ran in {time.time() - time_start} s")
                axes.ravel()[len(acquisitions)].set_title("{:0.2f} - {:0.2f}".format(bleach, acquisition.max()))
                acquisitions.append(acquisition)

        acquisitions = numpy.array(acquisitions)

        for ax, acquisition in zip(axes.ravel(), acquisitions):
            ax.imshow(acquisition, cmap="hot")#, vmin=0, vmax=0.1 * acquisitions.max(), cmap="hot")
        # for ax in axes.ravel():
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        if axes.ndim == 1:
            axes = axes[numpy.newaxis, :]
        for ax, p in zip(axes[:, 0], percents_p_sted):
            ax.set(ylabel="sted : {:0.2f}".format(p))
        for ax, p in zip(axes[-1, :], percents_p_ex):
            ax.set(xlabel="ex : {:0.2f}".format(p))

        # fig.savefig(f"./panels/bleachConstants/previous-{pdt * 1e+6:0.0f}.png", transparent=True, dpi=300)
        # pyplot.close("all")
    pyplot.show()

def generate_random_curve():
    while True:

        numpy.random.seed(None)
        FLUO = {
            "lambda_": 535e-9,
            "qy": 0.6,
            "sigma_abs": {
                488: 0.2e-21,   # was 1.15e-20
                575: 0.01e-21
            },
            "sigma_ste": {
                575: 3.0e-22,   # was 6.0e-21
            },
            "sigma_tri": numpy.random.uniform(0.01e-21, 0.1e-21), # 1e-21
            "tau": 3e-09,
            "tau_vib": 1.0e-12,
            "tau_tri": 5e-6,
            "phy_react": {488: numpy.random.uniform(1.0e-5, 10e-5),   # 1e-4, 12e-7
                          575: numpy.random.uniform(1.0e-8, 10e-8)},   # 1e-8
            "k_isc": 0.26e+6
        }

        laser_ex = base.GaussianBeam(**LASER_EX)
        laser_sted = base.DonutBeam(**LASER_STED)
        detector = base.Detector(**DETECTOR)
        objective = base.Objective(**OBJECTIVE)
        fluo = base.Fluorescence(**FLUO)
        microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)

        i_ex, i_sted, psf_det = microscope.cache(20e-9)

        percents_p_sted = [0]

        acquisitions = []
        for percent_p_sted in tqdm(percents_p_sted):
            for percent_p_ex in tqdm(percents_p_ex, leave=False):
                p_ex = percent_p_ex * (action_spaces["p_ex"]["high"] - action_spaces["p_ex"]["low"]) + action_spaces["p_ex"]["low"]
                p_sted = percent_p_sted * (action_spaces["p_sted"]["high"] - action_spaces["p_sted"]["low"]) + action_spaces["p_sted"]["low"]
                pdt = percent_pdt * (action_spaces["pdt"]["high"] - action_spaces["pdt"]["low"]) + action_spaces["pdt"]["low"]

                photons_ex = fluo.get_photons(i_ex * p_ex)
                k_ex = fluo.get_k_bleach(laser_ex.lambda_, photons_ex)
                duty_cycle = laser_sted.tau * laser_sted.rate
                photons_sted = fluo.get_photons(i_sted * p_sted * duty_cycle)
                k_sted = fluo.get_k_bleach(laser_sted.lambda_, photons_sted)

                prob_sted = numpy.ones(1)
                prob_ex = numpy.ones(1)

                prob_ex = prob_ex * numpy.prod(numpy.exp(-1. * k_ex * pdt))
                prob_sted = prob_sted * numpy.prod(numpy.exp(-1. * k_sted * pdt))
                prob = prob_ex * prob_sted

                initial_value = 125
                numpy.random.seed(42)
                current_value = numpy.random.binomial(initial_value, prob ** k_sted.size)

                acquisitions.append((initial_value - current_value) / initial_value)
        acquisitions = numpy.array(acquisitions)
        acquisitions = acquisitions.reshape(len(percents_p_sted), len(percents_p_ex))

        fig, ax = pyplot.subplots()
        ax.plot(percents_p_ex, acquisitions[0])
        ax.set(ylim=(-0.1, 1.1), xlim=(-0.1, 1.1))


        print(FLUO)

        pyplot.show()

if __name__ == "__main__":

    print(0.008e-5, 0.008e-5 - 0.007e-5, 0.008e-5 + 0.007e-5)
    samples = numpy.random.normal(
        loc=0.008e-5, scale=0.007e-5 / 2.576, size=10000
    )
    print(samples.mean(), samples.min(), samples.max())

    print(0.008e-8, 0.008e-8 - 0.004e-8, 0.008e-8 + 0.004e-8)
    samples = numpy.random.normal(
        loc=0.008e-8, scale=0.004e-8 / 2.576, size=10000
    )
    print(samples.mean(), samples.min(), samples.max())

    generate_example()
