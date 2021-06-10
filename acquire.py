
import numpy
import time
import random
import argparse

from pysted import base, utils, raster, bleach_funcs
from pysted.microscopes import DyMINMicroscope, RESCueMicroscope
from matplotlib import pyplot

numpy.random.seed(42)
random.seed(42)

def datamap_generator(shape, sources, molecules, shape_sources=(1, 1), random_state=None):
    """
    Function to generate a datamap with randomly located molecules.
    :param shape: A tuple representing the shape of the datamap. If only 1 number is passed, a square datamap will be
                  generated.
    :param sources: Number of molecule sources to be randomly placed on the datamap.
    :param molecules: Average number of molecules contained on each source. The actual number of molecules will be
                      determined by poisson sampling.
    :param shape_sources : A `tuple` of the shape of the sources
    :param random_state: Sets the seed of the random number generator.
    :returns: A datamap containing the randomly placed molecules
    """
    numpy.random.seed(random_state)
    if type(shape) == int:
        shape = (shape, shape)
    datamap = numpy.zeros(shape)
    pos = []
    for i in range(sources):
        row, col = numpy.random.randint(0, shape[0] - shape_sources[0]), numpy.random.randint(0, shape[1] - shape_sources[1])
        datamap[row : row + shape_sources[0], col : col + shape_sources[1]] += numpy.random.poisson(molecules)
        pos.append([row + shape_sources[0] // 2, row + shape_sources[1] // 2])
    return datamap, numpy.array(pos)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--microscope", type=str, default="STED",
                        help="Choose which microscope to acquire from")
    args = parser.parse_args()

    START = time.time()

    delta = 1
    num_mol = 2
    # molecules_disposition = numpy.zeros((50, 50))
    # molecules_disposition[
    #     molecules_disposition.shape[0]//2 - delta : molecules_disposition.shape[0]//2+delta+1,
    #     molecules_disposition.shape[1]//2 - delta : molecules_disposition.shape[1]//2+delta+1] = num_mol
    molecules_disposition, positions = datamap_generator(50, 10, num_mol, shape_sources=(2,2), random_state=42)
    # for j in range(1,4):
    #     for i in range(1,4):
    # #         molecules_disposition[
    # #             i * molecules_disposition.shape[0]//4,
    # #             j * molecules_disposition.shape[1]//4] = num_mol
    #         molecules_disposition[
    #             j * molecules_disposition.shape[0]//4 - delta : j * molecules_disposition.shape[0]//4 + delta + 1,
    #             i * molecules_disposition.shape[1]//4 - delta : i * molecules_disposition.shape[1]//4 + delta + 1] = num_mol

    print("Setting up the microscope ...")
    # Microscope stuff
    # egfp = {"lambda_": 535e-9,
    #         "qy": 0.6,
    #         "sigma_abs": {488: 1.15e-20,
    #                       575: 6e-21},
    #         "sigma_ste": {560: 1.2e-20,
    #                       575: 6.0e-21,
    #                       580: 5.0e-21},
    #         "sigma_tri": 1e-21,
    #         "tau": 3e-09,
    #         "tau_vib": 1.0e-12,
    #         "tau_tri": 5e-6,
    #         "phy_react": {488: 1e-7,   # 1e-4
    #                       575: 1e-11},   # 1e-8
    #         "k_isc": 0.26e6}
    egfp = {"lambda_": 535e-9,
            "qy": 0.6,
            "sigma_abs": {488: 3e-20,
                          575: 6e-21},
            "sigma_ste": {560: 1.2e-20,
                          575: 6.0e-21,
                          580: 5.0e-21},
            "sigma_tri": 1e-21,
            "tau": 3e-09,
            "tau_vib": 1.0e-12,
            "tau_tri": 5e-6,
            "phy_react": {488: 0.25e-7,   # 1e-4
                          575: 25.e-11},   # 1e-8
            "k_isc": 0.26e6}
    pixelsize = 20e-9
    bleach = True
    p_ex = 2e-6
    p_ex_array = numpy.ones(molecules_disposition.shape) * p_ex
    p_sted = 2.5e-3
    p_sted_array = numpy.ones(molecules_disposition.shape) * p_sted
    pdt = 100e-6
    pdt_array = numpy.ones(molecules_disposition.shape) * pdt
    roi = 'max'

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0.1)
    detector = base.Detector(noise=True, background=0, pcef=0.1)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    datamap = base.Datamap(molecules_disposition, pixelsize)

    if args.microscope == "DyMIN":
        # DyMIN microscope
        opts = {
            "scale_power" : [0., 0.25, 1.],
            "decision_time" : [10e-6, 10e-6, -1],
            "threshold_count" : [10, 8, 0]
        }
        microscope = DyMINMicroscope(laser_ex, laser_sted, detector, objective, fluo, opts=opts)
    elif args.microscope == "RESCue":
        # RESCue microscope
        opts = {
            "lower_threshold" : [2, -1],
            "upper_threshold" : [6, -1],
            "decision_time" : [25e-6, -1]
        }
        microscope = RESCueMicroscope(laser_ex, laser_sted, detector, objective, fluo, opts=opts)
    elif args.microscope == "STED":
        # STED microscope
        microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
    else:
        print("Microscope <{}> does not exist...".format(args.microscope))
        print("Please try again...".format(args.microscope))
        exit()

    start = time.time()
    i_ex, i_sted, _ = microscope.cache(datamap.pixelsize, save_cache=True)

    datamap.set_roi(i_ex, roi)
    print("Setup done...")

    time_start = time.time()
    # acquisition, bleached, scaled_power = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
    #                                                                     bleach=bleach, update=False, seed=42)

    experiment = utils.Experiment()
    experiment.add("microscope", microscope, datamap, params={"pdt":pdt, "p_ex":p_ex, "p_sted":p_sted})
    histories = experiment.acquire_all(1, bleach=bleach, processes=0)
    history = histories["microscope"]

    acquisition, bleached, scaled_power = history["acquisition"], history["bleached"], history["other"]

    print(f"ran in {time.time() - time_start} s")

    fig, axes = pyplot.subplots(1, 4, figsize=(10,3), sharey=True, sharex=True)

    axes[0].imshow(history["datamap"][-1])
    axes[0].set_title(f"Datamap roi")

    axes[1].imshow(bleached[-1], vmin=0, vmax=history["datamap"][-1].max())
    axes[1].set_title(f"Bleached datamap")

    axes[2].imshow(acquisition[-1])
    axes[2].set_title(f"Acquired signal (photons)")

    axes[3].imshow(scaled_power[-1])
    axes[3].set_title(f"Scaled power")

    print("Average molecules left : ", bleached[-1][molecules_disposition != 0].mean(axis=-1))

    print("Total run time : {}".format(time.time() - START))

    pyplot.show()
