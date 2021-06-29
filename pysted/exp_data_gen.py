import numpy as np
from matplotlib import pyplot as plt
from skimage import draw
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial.distance import cdist


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
        self.img_shape = img_shape
        self.datamap_pixelsize_nm = datamap_pixelsize_nm

        modes = {0: 'mushroom', 1: 'bump', 2: 'rand'}
        if mode not in modes.values():
            raise ValueError(f"mode {mode} is not valid, valid modes are {modes.values}")
        if mode == 'rand':
            mode_key = np.random.randint(0, 2)
            mode = modes[mode_key]

        width_nm = np.random.randint(width_nm[0], width_nm[1])
        height_nm = np.random.randint(height_nm[0], height_nm[1])
        width_px = int(np.round(width_nm / self.datamap_pixelsize_nm))
        height_px = int(np.round(height_nm / self.datamap_pixelsize_nm))

        if mode == 'mushroom':
            center = (int(self.img_shape[0] / 2), int(self.img_shape[1] / 2))

            # ellipse_rows, ellipse_cols = draw.ellipse(center[0], center[1], int(height_px / 2), int(width_px / 2))
            # ellipse_pixels = np.stack((ellipse_rows, ellipse_cols), axis=-1)

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
            polygon_corners_rows = [lowest_row, lowest_row, self.img_shape[0] - 1, self.img_shape[0] - 1]
            polygon_corners_cols = [left_lowest_px, right_lowest_px,
                                    right_lowest_px + poly_width,
                                    left_lowest_px - poly_width]
            polygon_rows, polygon_cols = draw.polygon(polygon_corners_rows, polygon_corners_cols)
            polygon_pixels = np.stack((polygon_rows, polygon_cols), axis=-1)

            img = np.zeros(self.img_shape)
            # fill the bottom of the image as if it was the dendrite
            img[self.img_shape[0] - np.random.randint(dendrite_thickness[0], dendrite_thickness[1]):
                self.img_shape[0]] = 1
            img[self.ellipse_perimeter[:, 0], self.ellipse_perimeter[:, 1]] = 1
            img = binary_fill_holes(img)
            img[polygon_pixels[:, 0], polygon_pixels[:, 1]] = 1

        elif mode == 'bump':
            dendrite_thickness = np.random.randint(dendrite_thickness[0], dendrite_thickness[1])
            center = (self.img_shape[0] - dendrite_thickness, int(self.img_shape[1] / 2))

            ellipse_rows, ellipse_cols = draw.ellipse(center[0], center[1], int(height_px / 2), int(width_px / 2))
            ellipse_pixels = np.stack((ellipse_rows, ellipse_cols), axis=-1)

            ellipse_r_perimeter, ellipse_c_perimeter = draw.ellipse_perimeter(center[0], center[1],
                                                                              int(height_px / 2), int(width_px / 2))
            # keep the perimeter of the ellipse as an attribute for latter addition of the nanodomains
            self.ellipse_perimeter = np.stack((ellipse_r_perimeter, ellipse_c_perimeter), axis=-1)
            # remove pixels on the bottom of the ellipse that are outside the ROI
            row_too_low_all = np.argwhere(ellipse_pixels[:, 0] >= self.img_shape[0])
            ellipse_pixels = np.delete(ellipse_pixels, row_too_low_all, axis=0)
            row_too_low_perimeter = np.argwhere(self.ellipse_perimeter[:, 0] >= self.img_shape[0])
            self.ellipse_perimeter = np.delete(self.ellipse_perimeter, row_too_low_perimeter, axis=0)

            img = np.zeros(self.img_shape)
            img[self.img_shape[0] - dendrite_thickness: self.img_shape[0]] = 1
            # img[ellipse_pixels[:, 0], ellipse_pixels[:, 1]] = n_molecs
            img[self.ellipse_perimeter[:, 0], self.ellipse_perimeter[:, 1]] = 1
            img = binary_fill_holes(img)

        self.frame = np.where(img, n_molecs, 0)

    def filter_valid_nanodomain_pos(self):
        """
        Returns a list of the valid nanodomain positions. The valid positions are on the upper half of the perimeter
        of the synapse
        :return:
        """
        # row_too_low_perimeter = np.argwhere(self.ellipse_perimeter[:, 0] >= self.img_shape[0])
        # self.ellipse_perimeter = np.delete(self.ellipse_perimeter, row_too_low_perimeter, axis=0)
        ellipsis_min_row = np.min(self.ellipse_perimeter[:, 0])
        ellipsis_max_row = np.max(self.ellipse_perimeter[:, 0])
        ellipsis_height = ellipsis_max_row - ellipsis_min_row

        lower_half_perimeter = np.argwhere(self.ellipse_perimeter[:, 0] >= ellipsis_min_row + int(ellipsis_height / 2))
        valid_nanodomains_pos = np.delete(self.ellipse_perimeter, lower_half_perimeter, axis=0)
        self.valid_nanodomains_pos = valid_nanodomains_pos

        return valid_nanodomains_pos


    def add_nanodomains(self, n_nanodmains, min_dist_nm=200, molecs_in_domain=5):
        """

        :param n_nanodmains:
        :param min_dist:
        :return:
        """
        self.nanodomains = []
        self.nanodomains_coords = []
        for i in range(n_nanodmains):
            if self.valid_nanodomains_pos.shape[0] == 0:
                # no more valid positions, simply stop
                break
            self.nanodomains.append(Nanodomain(self.img_shape, self.valid_nanodomains_pos))
            self.nanodomains_coords.append(self.nanodomains[i].coords)
            distances = cdist(np.array(self.nanodomains_coords), self.valid_nanodomains_pos)
            distances *= self.datamap_pixelsize_nm
            invalid_positions_idx = np.argwhere(distances < min_dist_nm)[:, 1]
            self.valid_nanodomains_pos = np.delete(self.valid_nanodomains_pos, invalid_positions_idx, axis=0)
        for row, col in self.nanodomains_coords:
            self.frame[row, col] += 5


class Nanodomain():
    """
    Nanodomain class
    For now I don't think this will do much other than say where the nanodomains are situated. For later exps, we will
    add flash routines and such to this class :)
    """
    def __init__(self, img_shape, valid_positions=None):
        """
        Spawns a Nanodomain
        :param valid_positions: List of valid pixels for the nanodomain to spawn in. If none, randomly spawn in the img
        """
        if valid_positions is not None:
            valid_positions = np.array(valid_positions)
            self.coords = np.array(valid_positions[np.random.randint(0, valid_positions.shape[0])])
        else:
            self.coords = np.array([np.random.randint(0, img_shape[0]), np.random.randint(0, img_shape[1])])

