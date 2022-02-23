import os
import tifffile

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from gym_sted.utils import get_foreground


def line_step_pixel_list_builder(dmap, line_step=1):
    """
    Builds a pixel_list with line_repetitions
    :param dmap: The datamap that will be acquired on (Datamap object)
    :param line_sted: The number of line repetitions. If 1, a normal raster scan pixel_list will be returned
    :returns: The pixel_list with appropriate number of line repetitions
    """
    # might be a more efficient way to this without such loops but I do not care for now :)
    n_rows, n_cols = dmap.whole_datamap[dmap.roi].shape
    pixel_list = []
    for row in range(n_rows):
        row_pixels = []
        for col in range(n_cols):
            pixel_list.append((row, col))
            row_pixels.append((row, col))
        if line_step > 1:
            for i in range(line_step - 1):   # - 1 cause the row is already there once at this stage
                for pixel in row_pixels:
                    pixel_list.append(pixel)
    return pixel_list


def microscopy_random_data_loader(paths):
    """
    Load a random imaging sequence, along with the imaging parameters used, from a list of folders containing bead
    images.
    :param paths: A list of the paths from which to randomly select an image and its params. Can also be a string
                  if user wants to load from only 1 specific folder
    :returns: A dictionary containing both confocals (acquired before and after the STED), the STED img and the
              parameters with which the STED was acquired (line_step, pdt, p_ex, p_sted)
    """
    if type(paths) is str:
        paths = [paths]

    load_path = np.random.choice(paths) + "/"
    results_table = pd.read_csv(load_path + "results_df.csv")

    img_idx = np.random.randint(0, results_table.shape[0])

    conf1 = tifffile.imread(load_path + f"conf1/{img_idx}.tiff")
    sted = tifffile.imread(load_path + f"sted/{img_idx}.tiff")
    conf2 = tifffile.imread(load_path + f"conf2/{img_idx}.tiff")

    # load line_step, dwelltime, p_ex and p_sted for the loaded img
    returns_dict = {
        "conf1": conf1,
        "sted": sted,
        "conf2": conf2,
        "line_step": results_table.at[img_idx, "line_step"],
        "pdt": results_table.at[img_idx, "dwelltime"],
        "p_ex": results_table.at[img_idx, "p_ex"],
        "p_sted": results_table.at[img_idx, "p_sted"],
        "load_path": load_path,
        "img_idx": img_idx
    }

    return returns_dict


def generate_dmap_from_real_img(sted_img, **kwargs):
    """
    Generates an array of the same shape as the image with molecules at the estimated bead positions, extracted from
    the input image.
    :param sted_img: The STED image from which a similar datamap is generated
    :param **kwargs: Thresholding parameters used to extract bead positions (min_distance and threshold_rel for
                     peak_local_max function), number of fluorophores to put in each bead, if to randomize
                     this number for each bead, ± value for random range
    :returns: An array of molecule dispositions which should approximately be the bead positions of the real sample
    """
    threshold_rel = kwargs.get("threshold_rel", 0.1)
    min_distance = kwargs.get("min_distance", 5)
    n_fluorophores = kwargs.get("n_fluorophores", 150)
    rand_range = kwargs.get("rand_range", 0)

    # use the foreground img of the sted_img to extract bead positions
    sted_fg_bool = get_foreground(sted_img)
    sted_fg = sted_img * sted_fg_bool
    bead_positions = peak_local_max(sted_fg, min_distance=min_distance, threshold_rel=threshold_rel)

    molecules_disposition = np.zeros(sted_img.shape)
    for col, row in bead_positions:
        molecules_disposition[row, col] += np.random.randint(n_fluorophores - rand_range,
                                                             n_fluorophores + rand_range + 1)

    return molecules_disposition



if __name__ == "__main__":
    np.random.seed(2)   # seed for testing

    data_main_dir = os.path.expanduser(os.path.join("~", "Documents", "research", "h22_code", "albert_beads_exps",
                                       "bandit-optimization-experiments", "2021-09-14_grid_articacts"))
    data_path = [data_main_dir + "/four_params_five_reps_2timesDwell"]
    data_paths = [
        data_main_dir + "/four_params_five_reps_2timesDwell", data_main_dir + "/four_params_five_reps_smallerDwell",
        data_main_dir + "/four_params_five_reps_smallerDwell_day2", data_main_dir + "/four_params_five_reps_tetraspec",
        data_main_dir + "/four_params_five_reps_tetraspec_smaller_dwell"
    ]

    testing_returns = microscopy_random_data_loader(data_paths)
    molecules_disposition = generate_dmap_from_real_img(testing_returns["sted"])

    plt.imshow(molecules_disposition)
    plt.show()
