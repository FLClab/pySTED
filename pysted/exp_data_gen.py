import numpy as np
import warnings
from matplotlib import pyplot as plt
from skimage import draw
from skimage import transform as sktr
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial.distance import cdist
from pysted import utils


def degrees_to_radians(angle_deg):
    return angle_deg * np.pi / 180


def rotate_nds(nd_coords, rot_angle, frame_shape=(64, 64)):
    """
    rotates the nanodomain coords
    :param nd_coords: list or np.array of the nanodomain coords in the unrotated frame
    :param rot_angle: rotation angle in degrees
    :param frame_shape: shape of the image in which the NDs reside. Should be (64, 64) in most cases
    """
    nd_coords_to_rot = np.asarray(nd_coords) - \
                       np.array([int(frame_shape[0] / 2) - 0.5, int(frame_shape[1] / 2) - 0.5])   # faut tu jfasse - 0.5
    angle_rad = degrees_to_radians(rot_angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_nd_coords = np.asarray(np.rint(nd_coords_to_rot @ rotation_matrix.T), dtype=np.int8) + \
                        np.array([int(frame_shape[0] / 2) - 0.5, int(frame_shape[1] / 2) - 0.5])
    rotated_nd_coords = np.rint(rotated_nd_coords).astype(int)

    return rotated_nd_coords


class Beads():
    def __init__(self, n_molecs, n_beads, datamap_pixelsize_nm=20, img_shape=(64, 64), seed=None):
        np.random.seed(seed)
        self.img_shape = img_shape
        self.datamap_pixelsize_nm = datamap_pixelsize_nm
        if type(n_molecs) is int or type(n_molecs) is tuple:
            self.n_molecs = n_molecs
        else:
            raise TypeError("Class attribute n_molecs has to be an int or a tuple")

        self.n_beads = n_beads

        self.frame = self.generate_frame()

    def generate_frame(self, seed=None):
        np.random.seed(seed)
        self.frame = np.zeros(self.img_shape)

        beads_pos = np.random.randint(self.img_shape, size=(self.n_beads, 2))

        for row, col in beads_pos:
            if type(self.n_molecs) is int:
                self.frame[row, col] = self.n_molecs
            elif type(self.n_molecs) is tuple:
                self.frame[row, col] = np.random.randint(np.min(self.n_molecs), np.max(self.n_molecs))
            else:
                raise TypeError("Class attribute n_molecs has to be an int or a tuple")

        return self.frame






class Synapse():
    """
    Synapse class
    Implemented with the intent of using a datamap_pixelsize of 20nm, to generate synapses with width of 500nm and
    height of 200nm inside a 64px by 64px datamap (1080nm x 1080nm).
    The synapse will have nanodomains along its upper edge
    :param n_molecs: The number of molecules that will be placed in every pixel of the synapse
    :param datamap_pixelsize_nm: The pixelsize (in nm) of the datamap
    :param width_nm: Interval for the desired width (in nm) of the synapse. The width will be sampled from this interval
    :param height_nm: Interval for the desired height (in nm) of the synapse. The height will be sampled from this
                      interval
    :param img_shape: The shape of the image (tuple)
    :param dendrite_thickness: Interval for the desired thickness of the dendrite (in pixels) to which the synapse is
                               connected. The height will be sampled from this interval. Basically, this just determines
                               how much of the dendrite we see in the ROI, drawn as a line on the bottom of the image.
    :param mode: The mode used for the random synapse generation. 'mushroom' will produce a mushroom-like synapse
                 (i.e., an ellipse connected to the dendrite by a recangular-ish neck). 'bump' will produce a bump-like
                 synapse (i.e., a have ellipse protruding from the dendrite). 'rand' will randomly select one of these
                 2 modes.
    :param seed: Sets the seed for the randomness
    """
    def __init__(self, n_molecs, datamap_pixelsize_nm=20, width_nm=(400, 800), height_nm=(300, 600),
                 img_shape=(64, 64), dendrite_thickness=(1, 10), mode='rand', seed=None):
        np.random.seed(seed)
        self.img_shape = img_shape
        self.datamap_pixelsize_nm = datamap_pixelsize_nm
        self.n_molecs_base = n_molecs
        self.flash_tstep = 0
        self.nanodomains = []
        self.nanodomains_coords = []

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
            sampled_thickness = np.random.randint(dendrite_thickness[0], dendrite_thickness[1])
            self.unrotated_dendrite_top = self.img_shape[0] - sampled_thickness
            img[self.img_shape[0] - sampled_thickness: self.img_shape[0]] = 1
            img[self.ellipse_perimeter[:, 0], self.ellipse_perimeter[:, 1]] = 1
            img = binary_fill_holes(img)
            img[polygon_pixels[:, 0], polygon_pixels[:, 1]] = 1

        elif mode == 'bump':
            sampled_thickness = np.random.randint(dendrite_thickness[0], dendrite_thickness[1])
            self.unrotated_dendrite_top = self.img_shape[0] - sampled_thickness
            center = (self.img_shape[0] - sampled_thickness, int(self.img_shape[1] / 2))

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
            img[self.img_shape[0] - sampled_thickness: self.img_shape[0]] = 1
            # img[ellipse_pixels[:, 0], ellipse_pixels[:, 1]] = n_molecs
            img[self.ellipse_perimeter[:, 0], self.ellipse_perimeter[:, 1]] = 1
            img = binary_fill_holes(img)

        self.frame = np.where(img, n_molecs, 0)

    def filter_valid_nanodomain_pos(self, thickness=0):
        """
        Returns a list of the valid nanodomain positions. The valid positions are on the upper half of the perimeter
        of the synapse
        :param thickness: The thickness of the valid region for the nanodomains. This value is 0 by default, meaning the
                          nanodomains will only be placed on the upper perimeter of the synapse
        :return: list of the valid positions for the nanodomains.
        """
        ellipsis_min_row = np.min(self.ellipse_perimeter[:, 0])
        ellipsis_max_row = np.max(self.ellipse_perimeter[:, 0])
        ellipsis_height = ellipsis_max_row - ellipsis_min_row

        lower_half_perimeter = np.argwhere(self.ellipse_perimeter[:, 0] >= ellipsis_min_row + int(ellipsis_height / 2))
        pixels_in_top_half_perimeter = np.delete(self.ellipse_perimeter, lower_half_perimeter, axis=0)

        synapse_pixels = np.argwhere(self.frame > 0)
        distances = cdist(synapse_pixels, pixels_in_top_half_perimeter)
        valid_pixels_idx = np.argwhere(distances <= thickness)[:, 0]
        valid_pixels = synapse_pixels[valid_pixels_idx]
        self.valid_nanodomains_pos = valid_pixels

        return valid_pixels


    def add_nanodomains(self, n_nanodmains, min_dist_nm=200, n_molecs_in_domain=5, seed=None, valid_thickness=0):
        """
        Adds nanodomains on the periphery of the synapse.
        :param n_nanodmains: The number of nanodomains that will be attempted to be added. If n_nanodomains is too
                             high and the min_dist_nm is too high, it may not be able to place all the nanodomains,
                             in which case a warning will be raised to tell the user not all nanodomains have been
                             placed
        :param min_dist_nm: The minimum distance (in nm) separating the nanodomains.
        :param n_molecs_in_domain: The number of molecules to be added at the nanodomain positions
        :param seed: Sets the seed for the random placement of nanodomains
        :param valid_thickness: The thickness of the valid region for the nanodomains. This value is 0 by default,
                                meaning the nanodomains will only be placed on the upper perimeter of the synapse
        """
        if type(n_nanodmains) is tuple:
            n_nanodmains = np.random.randint(n_nanodmains[0], n_nanodmains[1])
        if type(min_dist_nm) is tuple:
            min_dist_nm = np.random.randint(min_dist_nm[0], min_dist_nm[1])
        if type(valid_thickness) is tuple:
            valid_thickness = np.random.randint(valid_thickness[0], valid_thickness[1])
        if type(n_molecs_in_domain) is tuple:
            n_molecs_in_domain_list = [np.random.randint(n_molecs_in_domain[0], n_molecs_in_domain[1])
                                       for i in range(n_nanodmains)]

        np.random.seed(seed)
        self.nanodomains = []
        self.nanodomains_coords = []
        n_nanodmains_placed = 0
        self.filter_valid_nanodomain_pos(thickness=valid_thickness)

        for i in range(n_nanodmains):
            if self.valid_nanodomains_pos.shape[0] == 0:
                # no more valid positions, simply stop
                warnings.warn(f"Attempted to place {n_nanodmains} nanodomains, but only {n_nanodmains_placed} could"
                              f"be placed due to the minimum distance of {min_dist_nm} separating them")
                break
            self.nanodomains.append(Nanodomain(self.img_shape, self.valid_nanodomains_pos))
            self.nanodomains_coords.append(self.nanodomains[i].coords)
            n_nanodmains_placed += 1
            distances = cdist(np.array(self.nanodomains_coords), self.valid_nanodomains_pos)
            distances *= self.datamap_pixelsize_nm
            invalid_positions_idx = np.argwhere(distances < min_dist_nm)[:, 1]
            self.valid_nanodomains_pos = np.delete(self.valid_nanodomains_pos, invalid_positions_idx, axis=0)

        counter = 0
        for row, col in self.nanodomains_coords:
            if type(n_molecs_in_domain) is tuple:
                self.frame[row, col] += n_molecs_in_domain_list[counter]
            else:
                self.frame[row, col] += n_molecs_in_domain
            counter += 1
        self.n_molecs_in_domains = n_molecs_in_domain

    def fatten_nanodomains(self):
        """
        Fattens the nanodomains by 1 pixel on each side (if the side is within the synapse)
        """
        nanodomains_px = np.argwhere(self.frame == self.n_molecs_base + self.n_molecs_in_domains)
        for i in range(nanodomains_px.shape[0]):
            # look on all sides of the pixel, if it is within the synapse and not already part of a nanodomain,
            # add molecules to make it part of the nanodomain
            row, col = nanodomains_px[i, :]
            if (self.frame[row - 1, col] == self.n_molecs_base):   # look on top
                self.frame[row - 1, col] += self.n_molecs_in_domains
            if (self.frame[row + 1, col] == self.n_molecs_base):   # look on bot
                self.frame[row + 1, col] += self.n_molecs_in_domains
            if (self.frame[row, col - 1] == self.n_molecs_base):   # look right
                self.frame[row, col - 1] += self.n_molecs_in_domains
            if (self.frame[row, col + 1] == self.n_molecs_base):   # look left
                self.frame[row, col + 1] += self.n_molecs_in_domains

    def flash_nanodomains(self, current_time, time_quantum_us, delay=0, fwhm_step_usec_correspondance=(10, 1500000)):
        """
        Verifies if the nanodomains need to be updated for the flash, updates them if it is the case
        """
        n_time_quantums_us_per_flash_step = utils.time_quantum_to_flash_tstep_correspondance(
            fwhm_step_usec_correspondance,
            time_quantum_us)
        if (current_time % n_time_quantums_us_per_flash_step == 0):
            self.flash_tstep += 1
            for nanodomain in self.nanodomains:
                if self.flash_tstep >= nanodomain.flash_curve.shape[0]:
                    # if the flash is done, return the nanodomains to their initial value
                    self.flash_tstep = nanodomain.flash_curve.shape[0]
                self.frame[nanodomain.coords[0], nanodomain.coords[1]] = int(nanodomain.flash_curve[self.flash_tstep] *
                                                                             self.n_molecs_in_domains)
            self.frame = self.frame.astype(int)
            plt.imshow(self.frame)
            plt.title(f"flash_tstep = {self.flash_tstep} \n"
                      f"current_time = {current_time} \n"
                      f"flash value = {np.max(self.frame)}")
            plt.show()

    def rotate_and_translate(self, rot_angle=None, translate=True):
        # pad the frame to allow us to lengthen the dendrite and translate and stuff :)
        pad_width = 100
        padded_frame = np.pad(self.frame, pad_width)

        # stretch the dendrite
        padded_frame[pad_width + self.unrotated_dendrite_top: pad_width + self.frame.shape[0]] = self.n_molecs_base

        # rotate around the center, keep the center crop of img shape (64, 64), the rotate the nds
        if rot_angle is None:
            rot_angle = np.random.randint(0, 360)
        rot8_padded = sktr.rotate(padded_frame, rot_angle, resize=False, order=1, preserve_range=True)

        rot8_roi = rot8_padded[pad_width: pad_width + self.frame.shape[0],
                               pad_width: pad_width + self.frame.shape[1]]

        # set the translation limits to make sure no ND is outside the roi set by the frame
        if len(self.nanodomains) != 0:
            rotated_nd_coords = rotate_nds(self.nanodomains_coords, rot_angle, frame_shape=self.frame.shape)
        else:
            rotated_nd_coords = np.asarray(self.nanodomains_coords)
        dist_to_far_edge = np.min((np.min(self.frame.shape) - 1) * np.ones(rotated_nd_coords.shape) - rotated_nd_coords)
        dist_to_close_edge = np.min(rotated_nd_coords)
        translate_lim = np.min([dist_to_close_edge, dist_to_far_edge])

        if translate and (translate_lim > 0):
            translate_rows = np.random.randint(-translate_lim, translate_lim)
            translate_cols = np.random.randint(-translate_lim, translate_lim)
            rot8_roi = rot8_padded[pad_width + translate_rows: pad_width + translate_rows + self.frame.shape[0],
                                   pad_width + translate_cols: pad_width + translate_cols + self.frame.shape[1]]
        else:
            translate_rows, translate_cols = 0, 0
            rot8_roi = rot8_padded[pad_width: pad_width + self.frame.shape[0],
                                   pad_width: pad_width + self.frame.shape[1]]

        if len(self.nanodomains) != 0:
            rotated_nd_coords[:, 0] -= translate_rows
            rotated_nd_coords[:, 1] -= translate_cols
        self.nanodomains_coords = rotated_nd_coords
        for idx, nd in enumerate(self.nanodomains):
            nd.coords = self.nanodomains_coords[idx]
        self.frame = np.rint(rot8_roi).astype(int)




class Nanodomain():
    """
    Nanodomain class

    :param img_shape: The shape of the image in which the nanodomains will be placed
    :param valid_positions: A list of the valid positions from which to randomly sample a nanodomain position.
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

    def add_flash_curve(self, events_curves_path, seed=None):
        """
        Adds a list of molecule values for a certain number of frames
        """
        sampled_light_curve = utils.flash_generator(events_curves_path, seed=seed)
        normalized_light_curve = utils.rescale_data(sampled_light_curve, to_int=False, divider=2)
        # smoothed_light_curve = utils.savitzky_golay(normalized_light_curve, 11, 5)
        # plt.plot(normalized_light_curve, label="unsmoothed")
        # plt.plot(smoothed_light_curve, label="smoothed")
        # plt.legend()
        # plt.show()
        # exit()

        self.flash_curve = np.append(normalized_light_curve, [0])


if __name__ == "__main__":
    from pysted import base

    # generate a synapse with nanodomains
    # create molecules disposition
    n_molecs_base = 5
    min_dist = 100
    n_molecs_in_domain = 100
    n_nanodomains = 7
    valid_thickness = 3
    shroom = Synapse(n_molecs_base, mode="mushroom", seed=42)
    shroom.add_nanodomains(n_nanodomains, min_dist, seed=42, n_molecs_in_domain=n_molecs_in_domain, valid_thickness=3)

    # microscope stuff
    egfp = {
                "lambda_": 535e-9,
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
                "phy_react": {488: 0.25e-7,   # 1e-4
                              575: 25.0e-11},   # 1e-8
                "k_isc": 0.26e+6
            }
    pixelsize = 20e-9
    bleach = False
    p_ex = 5.0e-6
    p_sted = 5.0e-3
    pdt = 100.0e-6

    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0)
    detector = base.Detector(noise=True, background=0)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
    i_ex, _, _ = microscope.cache(pixelsize, save_cache=True)

    datamap = base.Datamap(shroom.frame, pixelsize)
    datamap.set_roi(i_ex, "max")

    # confocal acquisition
    confocal_acq, _, _ = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, 0.0,
                                                          bleach=False, update=False)

    # STED acquisition
    sted_acq, _, _ = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
                                                      bleach=False, update=False)

    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(datamap.whole_datamap[datamap.roi])
    axes[0].set_title(f"Datamap")
    axes[1].imshow(confocal_acq)
    axes[1].set_title(f"Confocal")
    axes[2].imshow(sted_acq)
    axes[2].set_title(f"STED")

    plt.show()
