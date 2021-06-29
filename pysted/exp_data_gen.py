import numpy as np
from matplotlib import pyplot as plt
from skimage import draw


class Synapse():
    """
    Synapse class
    Implemented with the intent of using a datamap_pixelsize of 20nm, to generate synapses with width of 500nm and
    height of 200nm inside a 64px by 64px datamap (1080nm x 1080nm).
    The synapse will have nanodomains along its upper edge
    """
    def __init__(self, n_molecs, datamap_pixelsize_nm=20, width_nm=(500, 1000), height_nm=(300, 500),
                 img_shape=(64, 64), dendrite_thickness=(1, 10), mode='rand', seed=None):
        np.random.seed(seed)

        modes = {0: 'mushroom', 1: 'bump', 2: 'rand'}
        if mode not in modes.values():
            raise ValueError(f"mode {mode} is not valid, valid modes are {modes.values}")
        if mode == 'rand':
            mode_key = np.random.randint(0, 2)
            mode = modes[mode_key]

        width_nm = np.random.randint(width_nm[0], width_nm[1])
        height_nm = np.random.randint(height_nm[0], height_nm[1])
        width_px = int(np.round(width_nm / datamap_pixelsize_nm))
        height_px = int(np.round(height_nm / datamap_pixelsize_nm))

        if mode == 'mushroom':
            center = (int(img_shape[0] / 2), int(img_shape[1] / 2))

            ellipse_rows, ellipse_cols = draw.ellipse(center[0], center[1], int(height_px / 2), int(width_px / 2))
            ellipse_pixels = np.stack((ellipse_rows, ellipse_cols), axis=-1)

            ellipse_r_perimeter, ellipse_c_perimeter = draw.ellipse_perimeter(center[0], center[1],
                                                                              int(height_px / 2), int(width_px / 2))
            # keep the perimeter of the ellipse as an attribute for latter addition of the nanodomains
            self.ellipse_perimeter = np.stack((ellipse_r_perimeter, ellipse_c_perimeter), axis=-1)

            # add a trapezoid shaped neck that joins the synapse to the bottom
            lowest_row = np.max(self.ellipse_perimeter[:, 0])
            lowest_row_pixels = np.squeeze(self.ellipse_perimeter[np.argwhere(self.ellipse_perimeter[:, 0] == lowest_row)])
            left_lowest_px = np.min(lowest_row_pixels[:, 1])
            right_lowest_px = np.max(lowest_row_pixels[:, 1])
            poly_width = np.random.randint(0, 3)
            polygon_corners_rows = [lowest_row, lowest_row, img_shape[0] - 1, img_shape[0] - 1]
            polygon_corners_cols = [left_lowest_px, right_lowest_px,
                                    right_lowest_px + poly_width,
                                    left_lowest_px - poly_width]
            polygon_rows, polygon_cols = draw.polygon(polygon_corners_rows, polygon_corners_cols)
            polygon_pixels = np.stack((polygon_rows, polygon_cols), axis=-1)

            img = np.zeros(img_shape)
            # fill the bottom of the image as if it was the dendrite
            img[img_shape[0] - np.random.randint(dendrite_thickness[0], dendrite_thickness[1]): img_shape[0]] = n_molecs
            img[ellipse_pixels[:, 0], ellipse_pixels[:, 1]] = n_molecs
            img[polygon_pixels[:, 0], polygon_pixels[:, 1]] = n_molecs
            self.frame = img
        elif mode == 'bump':
            dendrite_thickness = np.random.randint(dendrite_thickness[0], dendrite_thickness[1])
            center = (img_shape[0] - dendrite_thickness, int(img_shape[1] / 2))

            ellipse_rows, ellipse_cols = draw.ellipse(center[0], center[1], int(height_px / 2), int(width_px / 2))
            ellipse_pixels = np.stack((ellipse_rows, ellipse_cols), axis=-1)

            ellipse_r_perimeter, ellipse_c_perimeter = draw.ellipse_perimeter(center[0], center[1],
                                                                              int(height_px / 2), int(width_px / 2))
            # keep the perimeter of the ellipse as an attribute for latter addition of the nanodomains
            self.ellipse_perimeter = np.stack((ellipse_r_perimeter, ellipse_c_perimeter), axis=-1)
            # remove pixels on the bottom of the ellipse that are outside the ROI
            row_too_low_all = np.argwhere(ellipse_pixels[:, 0] >= img_shape[0])
            ellipse_pixels = np.delete(ellipse_pixels, row_too_low_all, axis=0)
            row_too_low_perimeter = np.argwhere(self.ellipse_perimeter[:, 0] >= img_shape[0])
            self.ellipse_perimeter = np.delete(self.ellipse_perimeter, row_too_low_perimeter, axis=0)

            img = np.zeros(img_shape)
            img[img_shape[0] - dendrite_thickness: img_shape[0]] = n_molecs
            img[ellipse_pixels[:, 0], ellipse_pixels[:, 1]] = n_molecs
            self.frame = img



class Nanodomain():
    """
    Nanodomain class
    For now I don't think this will do much other than say where the nanodomains are situated. For later exps, we will
    add flash routines and such to this class :)
    """