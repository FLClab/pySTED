
'''
This module contains utilitary functions that used across pySTED.

Code written by Benoit Turcotte, benoit.turcotte.4@ulaval.ca, October 2020
For use by FLClab (@CERVO) authorized people
'''

import numpy
import numpy as np
import scipy, scipy.constants, scipy.integrate

import math
import random
import warnings
import tifffile
import os

from matplotlib import pyplot
import time
from pysted import temporal, raster
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm, trange


def approx_binomial(n, p, size=None):
    '''Sample (64-bit) from a binomial distribution using the normal approximation.

    :param n: The number of trials (int or array of ints).
    :param p: The probability of success (float).
    :param size: The shape of the output (int or tuple of ints, optional).

    :return: 64-bit int or array of 64-bit ints.
    '''
    if not isinstance(n, numpy.ndarray):
        n = numpy.array(n, dtype=numpy.int64)
    is_0 = n == 0
    n[is_0] = 1
    gaussian = numpy.random.normal(n*p, numpy.sqrt(n*p*(1-p)), size=size)
    gaussian[is_0] = 0
    gaussian[gaussian < 0] = 0
    # add the continuity correction to sample at the midpoint of each integral bin
    gaussian += 0.5
    if size is not None:
        binomial = gaussian.astype(numpy.int64)
    else:
        # scalar
        binomial = int(gaussian)
    return binomial


def cart2pol(x, y):
    '''Convert the polar coordinates corresponding to the given cartesian
    coordinates.

    :param x: The :math:`x` cartesian coordinate.
    :param y: The :math:`y` cartesian coordinate.

    :return: A tuple of the angle :math:`\\theta` and the lenght :math:`rho`.
    '''
    theta = numpy.arctan2(y, x)
    rho = numpy.sqrt(x**2 + y**2)
    return theta, rho


def complex_quadrature(func, a, b, args):
    '''Integrate a complex integrand using the Gauss-Kronrod quadrature.

    :param func: The function to integrate.
    :param a: The lower bound of the integration.
    :param b: The upper bound of the integration.
    :param args: Additionals arguments of the function to integrate.

    :return: The integration result as a complex number.
    '''
    def real_func(x, args):
        return scipy.real(func(x, args))
    def imag_func(x, args):
        return scipy.imag(func(x, args))
    real_integral = scipy.integrate.quad(real_func, a, b, args)
    imag_integral = scipy.integrate.quad(imag_func, a, b, args)
    return real_integral[0] + 1j * imag_integral[0]


def fwhm(values):
    '''Compute the full width at half maximum of the Gaussian-shaped values.

    :param values: An array of values describing a Gaussian shape.

    :return: The full width at half maximum.
    '''
    hm = numpy.max(values) / 2
    idx_max = numpy.argmax(values)
    for i, _ in enumerate(values[:-1]):
        if values[i+1] >= hm:
            return (idx_max - i) * 2
    raise Exception("The shape is not Gaussian, FWHM cannot be computed.")


def fwhm_donut(values):
    '''Compute the full width at half maximum of the donut-shaped values.

    :param values: An array of values describing a donut shape.

    :return: A tuple of the outer and inner width at half maximum.
    '''
    hm = numpy.max(values) / 2
    idx_max = numpy.argmax(values)
    idx_min = numpy.argmin(values)
    for i, _ in enumerate(values[:-1]):
        if values[i+1] >= hm:
            # return big ray and small ray
            return idx_min - i, idx_min - (idx_max + idx_max - i)
    raise Exception("The shape of the donut is wrong, FWHM cannot be computed.")


def pinhole(radius, pixelsize, n_pixels=None):
    '''Return a pinhole mask.

    :param radius: The radius of the pinhole (m).
    :param pixelsize: The size of a pixel (m).
    :param n_pixels: The (optional) number of pixels (default: size of the
                     pinhole).
    '''
    if n_pixels is None:
        n_pixels = int(radius / pixelsize) * 2 + 1 # odd number of pixels
    center = int(n_pixels / 2)
    # n_pixels x n_pixels arrays containing the x and y coordinates as values
    xx, yy = numpy.mgrid[:n_pixels, :n_pixels]
    # a circle is the squared distance to the center point
    circle = numpy.sqrt((xx - center)**2 + (yy - center)**2)
    return (circle <= radius / pixelsize)


def rescale(data, factor):
    '''Rescale the *data* container (and content) given the ratio between the
    current container unit size and the new container unit size.


    Example::

        >>> data_10nm = numpy.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])
        >>> rescale(data_10nm, 2) # rescale to 20 nm size
        numpy.array([[2, 0],
                     [0, 0]])

    :param data: A 2D array.
    :param factor: The ratio between the original container units and the new
                   container units.

    :return: A 2D array.
    '''
    assert factor > 0, "The rescale factor must be positive!"
    new_data_h = int(data.shape[0] // factor)
    new_data_w = int(data.shape[1] // factor)
    new_data = numpy.zeros((new_data_h, new_data_w), dtype=data.dtype)
    for y in range(new_data_h):
        y_old_start = int(y * factor)
        y_old_end = int(y_old_start + factor)
        for x in range(new_data_w):
            x_old_start = int(x * factor)
            x_old_end = int(x_old_start + factor)
            new_data[y, x] = numpy.sum(data[y_old_start:y_old_end,
                                            x_old_start:x_old_end])
    return new_data


def resize(*images):
    '''Resize images to the shape of the largest (pad with zeros).

    :param images: Square shaped images.

    :return: A tuple of copies of the given *images* resized to the size of the
              largest input image.
    '''
    def fit(small_image, large_image):
        '''Pad with zeros and return a small image to fit a larger image,
        assuming square shaped images.
        '''
        half_small = int(small_image.shape[0] / 2)
        half_large = int(large_image.shape[0] / 2)
        pad = half_large - half_small
        return numpy.pad(small_image, ((pad, pad), (pad, pad)), "constant")

    sizes = [image.shape[0] for image in images]
    idx = numpy.argmax(sizes)
    largest_image = images[idx]
    new_images = [fit(image, largest_image) if i != idx else largest_image
                  for i, image in enumerate(images)]
    return tuple(new_images)


def inverse(x, a=1):
    '''Evaluate the inverse function :math:`1 / (a x + 1)`.

    :param x: An integer or array.
    :param a: The scale of *x*.

    :return: The result of the function, same shape as *x*.
    '''
    return 1 / (a * x + 1)


def inverse_exponential(x, a=1):
    '''Evaluate the inverse function :math:`1 / e^{ax}`.

    :param x: An integer or array.
    :param a: The scale of *x*.

    :return: The result of the function, same shape as *x*.
    '''
    return 1 / (numpy.exp(a * x))


def exponential(x, a):
    return numpy.exp(-a * x)


def stack(datamap, data):
    '''Compute a new frame consisting in a replication of the given *data*
    centered at every positions and multiplied by the factors given in the
    *datamap*.

    Example::

        >>> datamap = numpy.array([[2, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])
        >>> data = numpy.array([[1, 2, 1],
                                [2, 3, 2],
                                [1, 2, 1]])
        >>> utils.stack(datamap, data)
        numpy.array([[6, 4, 0, 0],
                     [4, 2, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

    :param datamap: A 2D array indicating how many data are positioned in every
    :param data: A 2D array containing the data to replicate.

    :return: A 2D array shaped like *datamap*.
    '''
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2
    frame = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))
    positions = numpy.where(datamap > 0)
    numbers = datamap[positions]
    for nb, y, x in zip(numbers, *positions):
        frame[y:y+h_pad+1, x:x+w_pad+1] += data * nb
    return frame[int(h_pad/2):-int(h_pad/2), int(w_pad/2):-int(w_pad/2)]


def stack_btmod_definitive(datamap, data, data_pixelsize, img_pixelsize, pixel_list):
    '''Compute a new frame consisting in a replication of the given *data*
    centered at every positions and multiplied by the factors given in the
    *datamap*.

    Example::

        >>> datamap = numpy.array([[2, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])
        >>> data = numpy.array([[1, 2, 1],
                                [2, 3, 2],
                                [1, 2, 1]])
        >>> utils.stack(datamap, data)
        numpy.array([[6, 4, 0, 0],
                     [4, 2, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

    :param datamap: A 2D array indicating how many data are positioned in every
    :param data: A 2D array containing the data to replicate.
    :param data_pixelsize: Length of a pixel in the datamap (m)
    :param img_pixelsize: Distance the laser moves in between each application of data. Must be a multiple of the
                          data_pixelsize (m)
    :param pixel_list: List of pixels on which we want to do the acquisition.

    :return: A 2D array shaped like *datamap*.
    '''
    filtered_pixel_list = pixel_list_filter(datamap, pixel_list, img_pixelsize, data_pixelsize)
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2
    modif_returned_array = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))
    padded_datamap = numpy.pad(numpy.copy(datamap), h_pad // 2, mode="constant", constant_values=0)

    for (row, col) in filtered_pixel_list:
        modif_returned_array[row:row + h_pad + 1, col:col + w_pad + 1] += data * padded_datamap[row:row + h_pad + 1,
                                                                                                col:col + w_pad + 1]

        # modif_returned_array[row:row + h_pad + 1, col:col + w_pad + 1] += data * datamap[row, col]
    return modif_returned_array[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]


def pixel_sampling(datamap, mode="all"):
    '''
    Function to test different pixel sampling methods, instead of simply imaging pixel by pixel

    :param datamap: A 2D array of the data to be imaged, used for its shape.
    :param mode: A keyword to determine the order of pixels in the returned list. By default, all pixels are added in a
                 raster scan (left to right, row by row) order.

    :return: A list containing all the pixels in the order in which we want them to be imaged.
    '''
    pixel_list = []
    if mode == "all":
        for row in range(datamap.shape[0]):
            for col in range(datamap.shape[1]):
                pixel_list.append((row, col))
    elif mode == "checkers":
        # TODO: ajouter checker_size comme param au lieu de hard codé ici
        # TODO: regarder s'il y a une manière plus efficace de faire ça
        checkers = numpy.zeros((datamap.shape[0], datamap.shape[1]))
        cell_size = 8

        even_row = True
        cell_white = False
        for row in range(0, checkers.shape[0], cell_size):
            for col in range(0, checkers.shape[0], cell_size):
                cell_white = not cell_white
                if even_row:
                    if cell_white:
                        checkers[row:row + cell_size, col: col + cell_size] = 1
                if not even_row:
                    if not cell_white:
                        checkers[row:row + cell_size, col: col + cell_size] = 1
            even_row = not even_row

        for row in range(datamap.shape[0]):
            for col in range(datamap.shape[1]):
                if checkers[row, col] == 1:
                    pixel_list.append((row, col))
    elif mode == "forsenCD":
        positions = numpy.where(datamap > 0)
        pixel_list = list(zip(positions[0], positions[1]))
    elif mode == "besides":
        pixel_list = []
        positions = numpy.where(datamap > 0)
        molecules = list(zip(positions[0], positions[1]))
        padded_datamap = numpy.pad(numpy.zeros(datamap.shape), 1, mode="constant")
        verif_matrix = numpy.zeros(datamap.shape)
        for (row, col) in molecules:
            interim_pixel_list = []
            xd = numpy.where(datamap[row-1:row+2, col-1:col+2] == 0)
            interim_pixel_list.append(list(zip(xd[0], xd[1])))
            interim_pixel_list = interim_pixel_list[0]
            for pixel in interim_pixel_list:
                if verif_matrix[pixel[0] + row - 1, pixel[1] + col - 1] == 0:
                    pixel_list.append((pixel[0] + row - 1, pixel[1] + col - 1))
                verif_matrix[pixel[0] + row - 1, pixel[1] + col - 1] = 1
    else:
        print(f"list_mode = {mode} is not valid, retard")

    return pixel_list


def pxsize_comp2(img_pixelsize, data_pixelsize):
    """
    Try number 2 for my float comparison function that hopefully will give the right values this time :)

    :param img_pixelsize: Acquisition pixel size. Has to be a multiple of data_pixelsize (m).
    :param data_pixelsize: Raw data pixelsize (m).

    :return: Integer values of the pixelsizes which can later be used to compute ratios and stuff :)
    """
    # test = img_pixelsize / data_pixelsize
    # test_int = int(img_pixelsize / data_pixelsize)
    # test3 = img_pixelsize % data_pixelsize
    # if img_pixelsize < data_pixelsize or not math.isclose(test3, 0):
    #     raise Exception("img_pixelsize has to be a multiple of data_pixelsize")
    img_pixelsize_int = float(str(img_pixelsize)[0: str(img_pixelsize).find('e')])
    data_pixelsize_int = float(str(data_pixelsize)[0: str(data_pixelsize).find('e')])
    img_pixelsize_exp = int(str(img_pixelsize)[str(img_pixelsize).find('e') + 1:])
    data_pixelsize_exp = int(str(data_pixelsize)[str(data_pixelsize).find('e') + 1:])
    exp = img_pixelsize_exp - data_pixelsize_exp
    img_pixelsize_int *= 10 ** exp
    # img_pixelsize_int = int(img_pixelsize_int)
    # data_pixelsize_int = int(data_pixelsize_int)
    test3 = img_pixelsize_int % data_pixelsize_int
    if img_pixelsize < data_pixelsize or not math.isclose(test3, 0):
        raise Exception("img_pixelsize has to be a multiple of data_pixelsize")
    return img_pixelsize_int, data_pixelsize_int


def pxsize_comp_array_maker(img_pixelsize, data_pixelsize, datamap):
    """
    Compare the pixel sizes of the image and the datamap and return the appropriate pixel sizes for the acquisition.

    Function which utilizes the ratio between the image pixelsize and the datamap pixelsize to create an appropriatly
    sized output datamap for a normal raster scan acquisition with ratio jumps between laser applications. This assures
    that the laser application is placed in the appropriate acquisition output pixel.
    
    :param img_pixelsize: The image pixelsize (m)
    :param data_pixelsize: The datamap pixelsize (m)
    :param datamap: The datamap on which the acquisition is made
    
    :return: An empty datamap of shape (ceil(datamap.shape[0] / ratio), ceil(datamap.shape[1] / ratio))
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = img_pixelsize_int / data_pixelsize_int
    nb_rows = int(numpy.ceil(datamap.shape[0] / ratio))
    nb_cols = int(numpy.ceil(datamap.shape[1] / ratio))
    datamap_to_fill = numpy.zeros((nb_rows, nb_cols))
    return datamap_to_fill


def pxsize_grid(img_pixelsize, data_pixelsize, datamap):
    """
    Function which creates a grid of the pixels. 
    
    This can be iterated on based on the ratio between img_pixelsize and
    data_pixelsize. Imagine the laser is fixed on a grid and can only make discrete movements, and this grid size is
    determined by the ratio

    :param img_pixelsize: Size of the minimum distance the laser must do between acquisitions (m). Must be a multiple of
                          data_pixelsize.
    :param data_pixelsize: Size of a pixel of the datamap (m).
    :param datamap: Raw molecule dispotion on which we wish to do an acquisition.
    
    :return: A list of the pixels which can be iterated on (?)
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = int(img_pixelsize_int / data_pixelsize_int)

    valid_pixels = []
    for row in range(0, datamap.shape[0], ratio):
        for col in range(0, datamap.shape[1], ratio):
            valid_pixels.append((row, col))

    return valid_pixels


def pxsize_ratio(img_pixelsize, data_pixelsize):
    """
    Computes the ratio between the acquisition pixel size and the datamap pixel size
    
    :param img_pixelsize: Minimum distance the laser must move during application. Multiple of data_pixelsize (m).
    :param data_pixelsize: Size of a pixel in the datamap (m).
    
    :return: the ratio between pixel sizes
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = int(img_pixelsize_int / data_pixelsize_int)
    return ratio


def mse_calculator(array1, array2):
    """
    Compute the RMS between two arrays. Must be of same size
    
    :param array1: First array
    :param array2: Second array
    
    :return: Mean squarred error of the 2 arrays
    """
    array_diff = numpy.absolute(array1 - array2)
    array_diff_squared = numpy.square(array_diff)
    mean_squared_error = float(numpy.sum(array_diff_squared) / (array1.shape[0] * array1.shape[1]))
    return mean_squared_error


def pixel_list_filter(datamap, pixel_list, img_pixelsize, data_pixelsize, output_empty=False):
    """
    Function to pre-filter a pixel list. Depending on the ratio between the data_pixelsize and acquisition pixelsize,
    a certain number of pixels must be skipped between laser applications.
    
    :param pixel_list: The list of pixels passed to the acquisition function, which needs to be filtered
    :param img_pixelsize: The acquisition pixelsize (m)
    :param data_pixelsize: The data pixelsize (m)
    :param output_empty: Bool to allow (or not) this function to return an empty pixel list
    
    :return: A filtered version of the input pixel_list, from which the pixels which can't be iterated over due to the
              pixel sizes have been removed
    """
    # figure out valid pixels to iterate on based on ratio between pixel sizes
    # imagine the laser is fixed on a grid, which is determined by the ratio
    valid_pixels_grid = pxsize_grid(img_pixelsize, data_pixelsize, datamap)

    # if no pixel_list is passed, use valid_pixels_grid to figure out which pixels to iterate on
    # if pixel_list is passed, keep only those which are also in valid_pixels_grid
    if pixel_list is None:
        pixel_list = valid_pixels_grid
    else:
        valid_pixels_grid_matrix = numpy.zeros(datamap.shape)
        nb_valid_pixels = 0
        for (row, col) in valid_pixels_grid:
            valid_pixels_grid_matrix[row, col] = 1
            nb_valid_pixels += 1
        pixel_list_matrix = numpy.zeros(datamap.shape)
        for idx, (row, col) in enumerate(pixel_list):
            pixel_list_matrix[row, col] += idx + 1
            if row == datamap.shape[0] - 1 and col == datamap.shape[1] - 1:
                break
        final_valid_pixels_matrix = pixel_list_matrix * valid_pixels_grid_matrix
        if numpy.array_equal(final_valid_pixels_matrix, numpy.zeros(datamap.shape)):
            if output_empty:
                return pixel_list
            else:
                warnings.warn(" \nNo pixels in the list passed is valid given the ratio between pixel sizes, \n"
                              "Iterating on valid pixels in a raster scan instead.")
                pixel_list = valid_pixels_grid  # itérer sur les pixels valides seulement
        else:
            pixel_list_interim = numpy.argsort(final_valid_pixels_matrix, axis=None)
            pixel_list_interim = numpy.unravel_index(pixel_list_interim, datamap.shape)
            pixel_list = [(pixel_list_interim[0][i], pixel_list_interim[1][i])
                          for i in range(len(pixel_list_interim[0]))]
            pixel_list = pixel_list[-numpy.count_nonzero(final_valid_pixels_matrix):]

    return pixel_list


def symmetry_verifier(array, direction="vertical", plot=False):
    """
    Verifies if the given array is symmetrical along the vertical or horizontal direction
    
    :param array: Array to be verified for symmetry
    :param direction: Direction along which to verify the symmetry. Vertical to see upper vs lower half, Horizontal to
                      see left vs right half.
    :param plot: Determines whether or not graphs of array and its symmetry will be displayed.
    
    :return: Array(s) displaying the symmetry
    """

    direction_lc = direction.lower()
    valid_directions = ["vertical", "horizontal"]
    if direction_lc not in valid_directions:
        raise Exception(f"{direction} is not a valid direction, valid directions are {valid_directions}")

    nb_rows, nb_cols = array.shape
    if direction_lc == "vertical":
        if nb_rows % 2 == 0:
            halfway = nb_rows // 2
            upper_half = array[0:halfway, :]
            lower_half = array[halfway:, :]
        else:
            halfway = nb_rows // 2 + 1
            upper_half = array[0:halfway, :]
            lower_half = array[halfway - 1:, :]
        symmetry = upper_half - numpy.flip(lower_half, 0)
    elif direction_lc == "horizontal":
        if nb_cols % 2 == 0:
            halfway = nb_cols // 2
            left_half = array[:, 0:halfway]
            right_half = array[:, halfway:]
        else:
            halfway = nb_cols // 2 + 1
            left_half = array[:, 0:halfway]
            right_half = array[:, halfway - 1:]
        symmetry = left_half - numpy.flip(right_half, 1)
    else:
        # forbidden zone, shouldn't go there ever because of previous error handling
        raise Exception(f"Forbidden zone, call BT if you get here")

    if plot:
        fig, axes = pyplot.subplots(1, 2)

        base_imshow = axes[0].imshow(array)
        axes[0].set_title(f"Base input array")
        fig.colorbar(base_imshow, ax=axes[0], fraction=0.04, pad=0.05)

        symmetry_imshow = axes[1].imshow(symmetry)
        axes[1].set_title(f"{direction_lc.capitalize()} symmetry verification")
        fig.colorbar(symmetry_imshow, ax=axes[1], fraction=0.04, pad=0.05)

        pyplot.show()

    return symmetry


def array_padder(base, laser, pad_value=0):
    """
    Function used to pad an array (base) according to the size of the secondary array being iterated over it (laser).
    
    :param base: Base array on which we wish to iterate another array.
    :param laser: Secondary array which will be iterated over the base array. Axes have to be of odd lengths in order
                  for it to have a well defined single pixel center.
    :param pad_value: Value of the padded region.
    
    :return: Padded version of the base array, along with the number of added rows and columns
    """

    laser_rows, laser_cols = laser.shape
    if laser_rows % 2 == 0 or laser_cols % 2 == 0:
        raise Exception(f"Laser shape has to be odd in order to have a well defined single pixel center")
    rows_pad, cols_pad = laser_rows // 2, laser_cols // 2
    padded_base = numpy.pad(base, ((rows_pad, rows_pad), (cols_pad, cols_pad)), 'constant', constant_values=pad_value)
    return padded_base, rows_pad, cols_pad


def pad_values(laser):
    """
    Pad values for the array_padder function

    Return the minimum necessary rows, cols to pad an array with if we want to iterated over all of it with a laser
    
    :param laser: Array of the shape of the laser which we want to iterated over a datamap.
    
    :return: rows_pad, cols_pad, the number of rows and columns we need to pad the array with.
    """
    rows_pad, cols_pad = laser.shape[0] // 2, laser.shape[1] // 2
    return rows_pad, cols_pad


def array_unpadder(padded_base, laser):
    """
    Unpads an array according to the size of the secondary array being iterated over it.

    Function used to unpad a padded array (padded_base) according to the size of the secondary array being iterated over
    it (laser).
    
    :param padded_base: Padded Base array which we wish to unpad.
    :param laser: Secondary array which has been iterated over padded_base. Axes have to be of odd lengths in order for
                  it to have a well defined single pixel center.
    
    :return: An unpadded version of the padded_base.
    """

    laser_rows, laser_cols = laser.shape
    if laser_rows % 2 == 0 or laser_cols % 2 == 0:
        raise Exception(f"Laser shape has to be odd in order to have a well defined single pixel center")
    rows_pad, cols_pad = laser_rows // 2, laser_cols // 2
    # laser_received[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]
    unpadded_base = padded_base[rows_pad:-rows_pad, cols_pad:-cols_pad]
    return unpadded_base


def datamap_generator(shape, sources, molecules, random_state=None):
    """
    Function to generate a datamap with randomly located molecules.

    :param shape: A tuple representing the shape of the datamap. If only 1 number is passed, a square datamap will be
                  generated.
    :param sources: Number of molecule sources to be randomly placed on the datamap.
    :param molecules: Average number of molecules contained on each source. The actual number of molecules will be
                      determined by poisson sampling.
    :param random_state: Sets the seed of the random number generator.
    
    :return: A datamap containing the randomly placed molecules
    """
    numpy.random.seed(random_state)
    if type(shape) == int:
        shape = (shape, shape)
    datamap = numpy.zeros(shape)
    for i in range(sources):
        row, col = numpy.random.randint(0, shape[0]), numpy.random.randint(0, shape[1])
        datamap[row, col] = numpy.random.poisson(molecules)

    return datamap


def molecules_symmetry(pre_bleach, post_bleach):
    """
    Function to compare the ratio of surviving molecules in the upper vs lower half of a datamap.
    
    :param pre_bleach: The datamap before bleaching it.
    :param post_bleach: The datamap after bleaching it.
    
    :return: Ratio of molecules surviving bleach, split between upper half and lower half
    """
    # We have to compare the same datamap before and after applying lasers on it, so the shape has to be the same
    if pre_bleach.shape != post_bleach.shape:
        raise ValueError("Both pre and post bleach datamaps need to be of the same shape")

    # if there is an odd number of rows, the last row of the upper half will be the first row of the lower half
    if pre_bleach.shape[0] % 2 == 0:
        pre_bleach_uhm = numpy.sum(pre_bleach[0:pre_bleach.shape[0] // 2, :])
        pre_bleach_lhm = numpy.sum(pre_bleach[pre_bleach.shape[0] // 2:, :])
        post_bleach_uhm = numpy.sum(post_bleach[0:post_bleach.shape[0] // 2, :])
        post_bleach_lhm = numpy.sum(post_bleach[post_bleach.shape[0] // 2:, :])
    else:
        pre_bleach_uhm = numpy.sum(pre_bleach[0:pre_bleach.shape[0] // 2 + 1, :])
        pre_bleach_lhm = numpy.sum(pre_bleach[pre_bleach.shape[0] // 2:, :])
        post_bleach_uhm = numpy.sum(post_bleach[0:post_bleach.shape[0] // 2 + 1, :])
        post_bleach_lhm = numpy.sum(post_bleach[post_bleach.shape[0] // 2:, :])

    uh_ratio = post_bleach_uhm / pre_bleach_uhm
    lh_ratio = post_bleach_lhm / pre_bleach_lhm
    return uh_ratio, lh_ratio


def molecules_survival(pre_bleach, post_bleach):
    """
    Return the ratio of surviving molecules
    
    :param pre_bleach: The datamap before bleaching it.
    :param post_bleach: The datamap after bleaching it.
    
    :return: Ratio of molecules surviving bleach
    """
    return numpy.sum(post_bleach) / numpy.sum(pre_bleach)


def float_to_array_verifier(float_or_array, shape):
    """
    Verify if a given input is a float or an array. 
    
    If it is a float, it will return an array of shape (shape) filled. 
    If it is an array, it will verify if it is of the appropriate shape.
    If it is neither, it will return an error
    
    :param float_or_array: Either a float or an array containing floats
    :param shape: The shape we want for our array (tuple)
   
    :return: An array of the appropriate shape
    """
    if isinstance(float_or_array, (float, numpy.floating)):
        returned_array = numpy.ones(shape) * float_or_array
    elif type(float_or_array) is numpy.ndarray and shape == float_or_array.shape:
        returned_array = numpy.copy(float_or_array)
    else:
        raise TypeError("Has to be either a float or an array of same shape as the ROI")
    return returned_array


def dict_write_func(file, dictio):
    """
    Write a dict to a text file in a good way :)

    :param file: path of the file to write to
    :param dictio: the dictionnary we wish to write to a txt file
    """
    f = open(file, 'a')
    f.write(str(dictio))
    f.write("\n")
    f.close()


def event_reader(file):
    """
    Read events from a file containing event dictionaries and return the dicts to a list

    :param file: Path to the txt file containing the dict for the events
    
    :return: A list containing the dicts of every identified event in a video
    """
    events_list = []
    f = open(file, 'r')
    lines = f.readlines()
    for line_idx, line in enumerate(lines):
        events_list.append(eval(line.strip()))
    return events_list


def add_event(file, start_frame, end_frame, start_row, end_row, start_col, end_col):
    """
    Function that allows a user to easily store an event in a file, which can later be read with the event_reader func.
    
    :param file: File to write the dict to. The goal is to use 1 text file to which we will write all the vents for
                 1 video.
    :param start_frame: Frame number for the start of the event
    :param start_row: Frame number for the end of the event
    :param start_col: Starting column for the left of the event
    :param end_frame: Ending column for the right of the event
    :param end_row: Starting row for the top of the event
    :param end_col: Ending row for the bottom of the event
    """
    event = {"start frame": start_frame,
             "end frame": end_frame,
             "start col": start_col,
             "end col": end_col,
             "start row": start_row,
             "end row": end_row}
    dict_write_func(file, event)


def get_light_curve(video_path, event):
    """
    Use a tif video of Ca2+ flashes along with a ROI (spatial and temporal) to convert the data to an intensity/photon
    count curve.
    
    :param video_path: The path to the video file from which we want to extract an event light curve (str)
    :param event: A dictionary containing the start and end info for frames, rows, columns of an event (dict)
    
    :return: A vector representing the mean intensity accros the frames of the event
    """
    data_vid = tifffile.imread(video_path)
    event_data = data_vid[event["start frame"]: event["end frame"],
                          event["start row"]: event["end row"],
                          event["start col"]: event["end col"]]
    mean_photons = numpy.mean(event_data, axis=(1, 2))

    return mean_photons


def rescale_data(data, to_int=True, divider=1):
    """
    Function to rescale the data (made for light curves, might be of use elsewhere) between 1 and max-min
    
    :param data: data to rescale.
    :param to_int: Determines whether the data is truncated to ints after being normalized. Useful for using the
                   fast acquisition function.
    
    :return: The data rescaled between 1 and max(data) - min(data)
    """
    b, a = numpy.max(data) - numpy.min(data), 1
    normalized = ((b - a)/divider) * ((data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))) + a
    if to_int:
        normalized = normalized.astype(int)
    return normalized


def shift_data(data, peak_idx=5):
    """
    Function to shift the data (made for light curves, might be of use elsewhere) so the peak is on idx 5
    
    :param data: data to shift
    :param peak_idx: idx at which we want the peak to be
    
    :return: The shifted data
    """
    peak_arg = numpy.argmax(data)
    if peak_arg >= peak_idx:
        shifted_curve = data[peak_arg - peak_idx:]
        while len(shifted_curve) != 40:
            shifted_curve = numpy.append(shifted_curve, shifted_curve[-1])
    else:
        shifted_curve = numpy.zeros(data.shape)
        shifted_curve[:peak_idx - peak_arg] = data[0]
        shifted_curve[peak_idx - peak_arg: peak_idx] = data[:peak_arg]
        shifted_curve[peak_idx:] = data[peak_arg: - (peak_idx - peak_arg)]
    return shifted_curve


def get_avg_lightcurve(light_curves):
    """
    This function takes as input a list of light curves and processes them so they are rescaled and shifted to align
    their peaks. It then return the avg light curve as well as its standard deviation, so we can sample a light curve
    for event simulation.
    
    :param light_curves: list of light curves of Ca2+ flash events
    
    :return: the avg light curve and std of the light curve
    """
    shifted_curves = []
    for curve in light_curves:
        rescaled_curve = rescale_data(curve, to_int=False, divider=1)
        shifted_curve = shift_data(rescaled_curve, peak_idx=5)
        shifted_curves.append(shifted_curve)

    avg_shifted_curves = numpy.mean(shifted_curves, axis=0)
    std_shifted_curves = numpy.std(shifted_curves, axis=0)
    return avg_shifted_curves, std_shifted_curves


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Return
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    ----------
    """

    # try:
    #     window_size = numpy.abs(numpy.int(window_size))
    #     order = numpy.abs(numpy.int(order))
    # except ValueError, msg:
    #     raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = numpy.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = numpy.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - numpy.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + numpy.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = numpy.concatenate((firstvals, y, lastvals))
    return numpy.convolve( m[::-1], y, mode='valid')


def sample_light_curve(light_curves):
    """
    This function allows to sample from a distribution of light curves and handles the smoothing and correcting values.
    
    :param light_curves: list containing the light curves we wish to do stats and sample from
    
    :return: A smoothed curve sampled from the distribution
    """
    avg_curve, std_curve = get_avg_lightcurve(light_curves)
    sampled_curve = numpy.random.normal(avg_curve, std_curve)
    sampled_curve = numpy.where(sampled_curve >= 1, sampled_curve, 1)
    smoothed_sampled = savitzky_golay(sampled_curve, 5, 2)   # These params give nice curves :)
    return smoothed_sampled


def flash_generator(events_curves_path, seed=None):
    """
    Generates a flash by sampling from statistics built from save light curves
    
    :param events_curves_path: Path to the .npy file containing the light curves
    :param seed: Sets the seed for random sampling :)
    
    :return: A sampled light curve
    """
    numpy.random.seed(seed)
    events_curves = numpy.load(events_curves_path)

    sampled_light_curve = sample_light_curve(events_curves)

    return sampled_light_curve


def sampled_flash_manipulations(events_curves_path, delay, rescale=True, seed=None):
    """
    Samples a light curve and modifies it to make it more prettier for training (i.e. more like the hand crafted light
    curves)
    
    - converts the values to ints
    - add variable delay at the start of the curve to delay the flash
    - (optional) rescales the values between [1, 28]
    
    :param events_curves_path: Path to the .npy file containing the light curves
    :param delay: Number of steps where the flash value stays ctw at 1 before the flash starts
    :param rescale: Whether or not the light curve will be rescaled. For now, if true, simply rescales between [1, 28]
                    because this is the value range for
    :param seed: Sets the seed for random sampling :)
    """
    numpy.random.seed(seed)
    events_curves = numpy.load(events_curves_path)

    sampled_light_curve = sample_light_curve(events_curves)

    if rescale:
        sampled_light_curve = (28 - 1) * (sampled_light_curve - sampled_light_curve.min()) / \
                                       (sampled_light_curve.max() - sampled_light_curve.min()) + 1

    sampled_light_curve = numpy.round(sampled_light_curve).astype(int)

    if type(delay) is tuple:
        delay = numpy.random.randint(delay[0], delay[1])
    if delay > 0:
        delay = numpy.ones(delay)
        sampled_light_curve = numpy.append(delay, sampled_light_curve)

    return sampled_light_curve


def smooth_ramp_hand_crafted_light_curve_2(n_steps_rise=10, n_steps_decay=10, delay=0, n_molecules_multiplier=None,
                                           end_pad=0):
    if n_molecules_multiplier is None:
        n_molecules_multiplier = numpy.random.randint(20, 35)
    if type(n_molecules_multiplier) is tuple:
        n_molecules_multiplier = numpy.random.randint(n_molecules_multiplier[0], n_molecules_multiplier[1])

    mean = 0
    std = 1
    variance = np.square(std)
    x = np.arange(-5, 5, .01)
    f = np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))

    gaussian_rise_full_res = f[:int(f.shape[0]/2)]
    # first ~third of the gaussian rise are values close to 0, dont need them
    gaussian_rise_full_res = gaussian_rise_full_res[int(gaussian_rise_full_res.shape[0] / 3):]
    gaussian_rise_full_res = gaussian_rise_full_res / np.max(gaussian_rise_full_res)

    rise_indices = np.linspace(0, gaussian_rise_full_res.shape[0] - 1, num=n_steps_rise, dtype=int)
    rise_values = gaussian_rise_full_res[rise_indices]

    tau = 3
    tmax = 10
    t = np.linspace(0, tmax, n_steps_decay)
    y = n_molecules_multiplier * np.exp(-t / tau)

    light_curve = np.ones(delay + n_steps_rise + n_steps_decay + end_pad + 1)
    light_curve[delay + n_steps_rise - 1:delay + n_steps_rise - 1 + y.shape[0]] *= y
    # light_curve = numpy.where(light_curve < 1, 1, light_curve)
    light_curve = light_curve / np.max(light_curve)
    light_curve[delay:delay + n_steps_rise] = rise_values
    light_curve[:delay] = 0
    light_curve *= n_molecules_multiplier

    return light_curve


def hand_crafted_light_curve(delay=2, n_decay_steps=10, n_molecules_multiplier=28, end_pad=0):
    """
    Hand crafted light curve that has a more abrupt rise than sampling a light curve from real data.
    
    :param delay: The number of steps before the peak of the light curve.
    :param n_decay_steps: The number of steps for the light curve to return to 1
    :param n_molecules_multiplier: The value of the light curve at it's peak
    :param end_pad: The number of steps where the curve stays flat at 1 after the end of the exponential decay.
    
    :return: The hand crafted light curve, which is flat at 1 until t = delay, where it peaks to n_molecs_multiplier,
              then decays back to 1 over t = n_decay_steps steps, and stays flat at 1 for end_pad + 1 steps
    """
    tau = 3
    tmax = 10
    t = np.linspace(0, tmax, n_decay_steps)
    y = n_molecules_multiplier * np.exp(-t / tau)

    # light_curve = np.ones(20)
    light_curve = np.ones(delay + n_decay_steps + end_pad)
    light_curve[delay:delay + y.shape[0]] *= y
    light_curve = numpy.where(light_curve < 1, 1, light_curve)

    return light_curve


def smooth_ramp_hand_crafted_light_curve(delay=2, n_decay_steps=10, n_molecules_multiplier=None, end_pad=0):
    """
    Hand crafted light curve that has a more abrupt rise than sampling a light curve from real data.
    
    :param delay: The number of steps before the peak of the light curve.
    :param n_decay_steps: The number of steps for the light curve to return to 1
    :param n_molecules_multiplier: The value of the light curve at it's peak
    :param end_pad: The number of steps where the curve stays flat at 1 after the end of the exponential decay.
    
    :return: The hand crafted light curve, which is flat at 1 until t = delay, where it peaks to n_molecs_multiplier,
              then decays back to 1 over t = n_decay_steps steps, and stays flat at 1 for end_pad + 1 steps
    """
    if n_molecules_multiplier is None:
        n_molecules_multiplier = numpy.random.randint(20, 35)
    if type(n_molecules_multiplier) is tuple:
        n_molecules_multiplier = numpy.random.randint(n_molecules_multiplier[0], n_molecules_multiplier[1])

    tau = 3
    tmax = 10
    t = np.linspace(0, tmax, n_decay_steps)
    y = n_molecules_multiplier * np.exp(-t / tau)

    # light_curve = np.ones(20)
    light_curve = np.ones(delay + n_decay_steps + end_pad + 1)
    light_curve[delay] = int(0.2 * np.max(y))
    light_curve[delay + 1:delay + 1 + y.shape[0]] *= y
    light_curve = numpy.where(light_curve < 1, 1, light_curve)

    return light_curve


def generate_fiber_with_synapses(datamap_shape, fibre_min, fibre_max, n_synapses, min_dist, polygon_scale=(5, 10)):
    """
    This func allows a user to generate a fiber object and synapses attached to it.
    
    :param datamap_shape: shape of the image to which we will add the fiber and synapses
    :param fibre_min: min position for the fibre to start at
    :param fibre_max: max position for the fibre to start at
    :param n_synapses: number of synapses to put on the fiber
    :param min_dist: min distance we want to have between the synapses. this is to make sure we don't have overlapping
                     synapses
    :param polygon_scale: values from which the polygon size will be sampled
    
    :return: the fiber object and the polygon objects representing the synapses
    """
    min_array, max_array = numpy.asarray((fibre_min, fibre_min)), numpy.asarray((fibre_max, fibre_max))
    fibre = temporal.Fiber(random_params={"num_points": (fibre_min, fibre_max),
                                                    "pos": [numpy.zeros((1, 2)) + min_array,
                                                            datamap_shape - max_array],
                                                    "scale": (1, 5)})
    n_added = 0
    synapse_positions = numpy.empty((0, 2))
    n_loops = 0
    while n_added != n_synapses:
        # sometimes we get infinite loops if we cant place the synapses far enough appart
        if n_loops > n_synapses * 100:
            break
        sampled_node = numpy.asarray(random.sample(list(fibre.nodes_position), 1)[0].astype(int))
        if numpy.less_equal(sampled_node, 0).any() or \
                numpy.greater_equal(sampled_node, datamap_shape - numpy.ones((1, 1))).any():
            continue
        if n_added == 0:
            synapse_positions = numpy.append(synapse_positions, sampled_node)
            synapse_positions = numpy.expand_dims(synapse_positions, 0).astype(int)
            n_added += 1
            continue
        else:
            sample_to_verify = numpy.expand_dims(numpy.copy(sampled_node), axis=0).astype(int)
            synapse_positions = numpy.append(synapse_positions, sample_to_verify, axis=0).astype(int)
            distances = cdist(synapse_positions, synapse_positions)
            distances[n_added, n_added] = min_dist + 1
            if numpy.less_equal(distances[n_added, :], min_dist).any():
                # at least 1 elt is closer than 10 pixels to an already present elt so remove it :)
                synapse_positions = numpy.delete(synapse_positions, n_added, axis=0)
            else:
                # good to add to the list
                n_added += 1
        n_loops += 1

    polygon_list = []
    for node in synapse_positions:
        polygon = temporal.Polygon(random_params={"pos": [node, node],
                                                  "scale": polygon_scale})
        polygon_list.append(polygon)

    return fibre, polygon_list


def generate_secondary_fibers(datamap_shape, main_fiber, n_sec, min_dist=10, sec_len=(2, 6), seed=None):
    """
    This function allows to spawn secondary fibers branching from a main fiber
    
    :param datamap_shape: The shape of the datamap in which the main fiber resides
    :param main_fiber: The main fiber object to which we will add secondary fiber branches
    :param n_sec: The interval for the number of secondary branches we wish to spawn (tuple)
    :param min_dist: The min distance between spawned secondary fiber, to ensure they are not all clumped
    :param sec_len: The interval for the length of the secondary fibers (tuple)
    :param seed: Random number generator seed
    
    :return: a list containing the secondary fiber objects
    """
    n_added = 0
    sec_fiber_positions = numpy.empty((0, 2))
    angle_at_position = []
    n_secondary = int(random.uniform(*n_sec))
    n_loops = 0
    while n_added != n_secondary:
        n_loops += 1
        if n_loops >= 100 * n_secondary:
            break
        # sampled_node = numpy.asarray(random.sample(list(main_fiber.nodes_position), 1)[0].astype(int))
        sample_idx = numpy.random.randint(len(main_fiber.nodes_position))
        sampled_node = main_fiber.nodes_position[sample_idx, :].astype(int)
        if numpy.less_equal(sampled_node, 0).any() or \
                numpy.greater_equal(sampled_node, datamap_shape - numpy.ones((1, 1))).any():
            continue
        if n_added == 0:
            sec_fiber_positions = numpy.append(sec_fiber_positions, sampled_node)
            sec_fiber_positions = numpy.expand_dims(sec_fiber_positions, 0).astype(int)
            angle_at_position.append(main_fiber.angles[sample_idx])
            n_added += 1
            continue
        else:
            sample_to_verify = numpy.expand_dims(numpy.copy(sampled_node), axis=0).astype(int)
            sec_fiber_positions = numpy.append(sec_fiber_positions, sample_to_verify, axis=0).astype(int)
            distances = cdist(sec_fiber_positions, sec_fiber_positions)
            distances[n_added, n_added] = min_dist + 1
            if numpy.less_equal(distances[n_added, :], min_dist).any():
                # at least 1 elt is closer than 10 pixels to an already present elt so remove it :)
                sec_fiber_positions = numpy.delete(sec_fiber_positions, n_added, axis=0)
            else:
                # good to add to the list
                angle_at_position.append(main_fiber.angles[sample_idx])
                n_added += 1

    sec_fibers_list = []
    for node in sec_fiber_positions:
        sec_fiber = temporal.Fiber(random_params={"num_points": sec_len,
                                                  "pos": [node, node],
                                                  "scale": (1, 3),
                                                  "angle": (- 0.25, 0.25)}, seed=seed)
        sec_fibers_list.append(sec_fiber)

    return sec_fibers_list


def generate_synapses_on_fiber(datamap_shape, main_fiber, n_syn, min_dist, synapse_scale=(5, 10)):
    """
    Generates polygon objects (representing synapses) on the main fiber
    
    :param datamap_shape: The shape of the datamap on which the main_fiber lies
    :param main_fiber: Fiber object representing the main branch
    :param n_syn: The interval from which we will sample the number of synapses to spawn (tuple)
    :param min_dist: The minimal distance between 2 synapses, to prevent clumping
    :param synapse_scale: The interval form which we will sample each synapses' size
    
    :return: A list containing all the synapses on the main fiber
    """
    n_added = 0
    synapse_positions = numpy.empty((0, 2))
    n_synapses = int(random.uniform(*n_syn))
    n_loops = 0
    while n_added != n_synapses:
        n_loops += 1
        if n_loops >= 100 * n_synapses:
            break
        sampled_node = numpy.asarray(random.sample(list(main_fiber.nodes_position), 1)[0].astype(int))
        if numpy.less_equal(sampled_node, 0).any() or \
                numpy.greater_equal(sampled_node, datamap_shape - numpy.ones((1, 1))).any():
            continue
        if n_added == 0:
            synapse_positions = numpy.append(synapse_positions, sampled_node)
            synapse_positions = numpy.expand_dims(synapse_positions, 0).astype(int)
            n_added += 1
            continue
        else:
            sample_to_verify = numpy.expand_dims(numpy.copy(sampled_node), axis=0).astype(int)
            synapse_positions = numpy.append(synapse_positions, sample_to_verify, axis=0).astype(int)
            distances = cdist(synapse_positions, synapse_positions)
            distances[n_added, n_added] = min_dist + 1
            if numpy.less_equal(distances[n_added, :], min_dist).any():
                # at least 1 elt is closer than 10 pixels to an already present elt so remove it :)
                synapse_positions = numpy.delete(synapse_positions, n_added, axis=0)
            else:
                # good to add to the list
                n_added += 1

    synapse_list = []
    for node in synapse_positions:
        polygon = temporal.Polygon(random_params={"pos": [node, node],
                                                  "scale": synapse_scale})
        synapse_list.append(polygon)

    return synapse_list


def generate_synaptic_fibers(image_shape, main_nodes, n_sec_fibers, n_synapses, min_fiber_dist=3, min_synapse_dist=1,
                             sec_fiber_len=(10, 20), synapse_scale=(5, 5), seed=None):
    """
    This function wraps up the generation of fibers with secondary branches and synapses in a stand-alone function

    TODO:
    - Add variable number of synapses, distances, 
    - Add "position identifiers" to the synapses so I can easily make them flash after

    :param image_shape: The shape of the ROI in which we want to spawn stuff
    :param main_nodes: ???
    :param n_sec_fibers: The interval for the number of secondary fibers branching from the main fiber (tuple)
    :param n_synapses: The interval for the number of synapses (tuple)
    :param min_fiber_dist: The minimum distance separating the secondary fibers
    :param min_synapse_dist: The minimum distance separating the synapses
    :param sec_fiber_len: The interval for the lengths of the secondary fibers
    :param synapse_scale: The interval for the size of the synapses
    :param seed: Random number generator seed
    
    :return: An array containing the disposition of molecules corresponding to the generated shape and a list
             containing all the synapses (Polygon objects)
    """
    # generate an empty image
    image = numpy.zeros(image_shape)

    # generate the main fiber
    min_nodes, max_nodes = main_nodes[0], main_nodes[1]
    min_array, max_array = numpy.asarray((min_nodes, min_nodes)), numpy.asarray((max_nodes, max_nodes))
    fibre_rand = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                               "pos": [numpy.zeros((1, 2)) + min_array,
                                                       image.shape - max_array],
                                               "scale": (1, 5)}, seed=seed)

    # generate secondary fibers
    sec_fibers = generate_secondary_fibers(image_shape, fibre_rand, n_sec_fibers, min_fiber_dist,
                                           sec_len=sec_fiber_len, seed=seed)

    # generate synapses attached to the secondary fibers
    synapses_lists = []
    for secondary_fiber in sec_fibers:
        ith_fiber_synapses = generate_synapses_on_fiber(image_shape, secondary_fiber, n_synapses, min_synapse_dist,
                                                        synapse_scale=synapse_scale)
        synapses_lists.append(ith_fiber_synapses)

    roi = ((0, 0), image_shape)
    ensemble_test = temporal.Ensemble(roi=roi)
    ensemble_test.append(fibre_rand)
    for idx, sec_fiber in enumerate(sec_fibers):
        ensemble_test.append(sec_fiber)
        for synapse in synapses_lists[idx]:
            ensemble_test.append(synapse)

    return ensemble_test, synapses_lists


def generate_synapse_flash_dicts(synapses_list, roi_shape):
    """
    This function is used to generate the dictionnaries needed to keep track of which synapses are flashing and where
    they are situatied in the flash and such
    
    :param synapses_list: The list of all the synapses in the frame (flattenened)
    :param roi_shape: The shape of the frame
    
    :return: synapse_flashing_dict, a dict corresponding the synapses to whether they are currently flashing or not,
             synapse_flash_idx_dict, a dict corresponding each flash to where in the light curve they are at,
             synapse_flash_curve_dict, a dict containing the light curve sampled for the flash of this synapse,
             isolated_synapses_frames, a dict containing a frame for each synapse in which only it appears and the rest
                                       is 0
    """
    synpase_flashing_dict, synapse_flash_idx_dict, synapse_flash_curve_dict, isolated_synapses_frames = {}, {}, {}, {}

    for idx_syn in range(len(synapses_list)):
        synpase_flashing_dict[idx_syn] = False
        synapse_flash_idx_dict[idx_syn] = 0
        rr, cc = synapses_list[idx_syn].return_shape(shape=roi_shape)
        isolated_synapses_frames[idx_syn] = numpy.zeros(roi_shape).astype(int)
        isolated_synapses_frames[idx_syn][rr.astype(int), cc.astype(int)] += 5

    return synpase_flashing_dict, synapse_flash_idx_dict, synapse_flash_curve_dict, isolated_synapses_frames


def generate_raster_pixel_list(n_pixels_to_add, starting_pixel, img):
    """
    Generates a pixel list of a raster scan of n_pixels_to_add pixels starting from starting_pixel
    
    :param n_pixels_to_add: The number of pixels for which we want to image
    :param starting_pixel: The starting point of our raster scan
    :param img: the img in which the raster scan occurs
    
    :return: A pixel list for the raster scan starting at starting_pixel
    """
    if starting_pixel[0] >= img.shape[0] or starting_pixel[1] >= img.shape[1]:
        raise ValueError(f"starting pixel {starting_pixel} must be within img bounds (of shape {img.shape})")
    return_list = []
    current_idx = starting_pixel
    for i in range(n_pixels_to_add):
        return_list.append(tuple(current_idx))
        current_idx[1] += 1
        if current_idx[1] >= img.shape[1]:
            current_idx[1] = 0
            current_idx[0] += 1
        if current_idx[0] >= img.shape[0]:
            current_idx[0] = 0

    return return_list


def set_starting_pixel(previous_pixel, image_shape, ratio=1):
    """
    Return a value 1 pixel further from the last pixel of an acquisition list in an normal raster scan fashion.
    
    :param previous_pixel: The pixel on which the previous raster scan stopped
    :param image_shape: the shape of the ROI on which the raster scan is occuring
    
    :return: The pixel on which the next raster scan should start
    """
    starting_pixel = list(previous_pixel)
    starting_pixel[1] += ratio
    if starting_pixel[1] >= image_shape[1]:
        starting_pixel[1] = 0
        starting_pixel[0] += ratio
    if starting_pixel[0] >= image_shape[0]:
        starting_pixel[0] = 0

    return starting_pixel


def compute_time_correspondances(fwhm_step_sec_correspondance, acquisition_time_sec, pixel_dwelltime, mode="flash"):
    """
    This function computes how many loop steps will occur and how many pixels can be imaged for each loop step.
    So far this only works for static pixel_dwelltime, need to figure out how to make it work for varying dwell times
    per pixel or varying dwell times as in RESCue.
    
    :param fwhm_step_sec_correspondance: a tuple containing how large in time steps the FWHM of the mean flash is at
                                         index 0 and how large in seconds we want the FWHM to be at index 1
    :param acquisition_time_sec: How long we want to acquire on the same datamap, in seconds. This will be used to
                                 determine how many loops we need to do (float? int?)
    :param pixel_dwelltime: The pixel dwell time used by the microscope (float)
    
    :return: The number of pixels that can be imaged per loop and the number of loop iterations
    """

    legal_modes = ["flash", "pdt"]
    if mode.lower() not in legal_modes:
        raise ValueError(f"Mode '{mode}' is not valid")

    fwhm_time_steps, fwhm_time_secs = fwhm_step_sec_correspondance[0], fwhm_step_sec_correspondance[1]
    sec_per_time_step = fwhm_time_secs / fwhm_time_steps
    if mode == "flash":
        n_time_steps = int(acquisition_time_sec / sec_per_time_step)
        n_pixels_per_tstep = sec_per_time_step / pixel_dwelltime
        return n_time_steps, n_pixels_per_tstep
    elif mode == "pdt":
        n_time_steps = int(acquisition_time_sec / pixel_dwelltime) 
        x_pixels_for_flash_ts = round(sec_per_time_step / pixel_dwelltime)
        return n_time_steps, x_pixels_for_flash_ts


def time_quantum_to_flash_tstep_correspondance(fwhm_step_sec_corresnpondance, time_quantum_us):
    """
    Computes the correspondance between the time quantum value (in us) and the flash time steps
    
    :param fwhm_step_sec_corresnpondance: A tuple containing how large in time steps the FWHM of the mean flash is at
                                          index 0 and how large in useconds we want the FWHM to be at index 1
    :param time_quantum_us: Value of the time quantum used by the master clock (in us)
    
    :return: The number of time quantums per flash time step
    """
    fwhm_time_usecs, fwhm_time_steps = fwhm_step_sec_corresnpondance[1], fwhm_step_sec_corresnpondance[0]
    usec_per_time_step = fwhm_time_usecs / fwhm_time_steps
    # I think in most cases, the time_quantum will be 1 us
    # I also need to make sure the fwhm_time_usecs is in usecs and is a multiple of the time_quantum_us ?
    n_time_quantums_per_flash_ts = int(usec_per_time_step / time_quantum_us)
    return n_time_quantums_per_flash_ts




def flash_routine(synapses, probability, synapse_flashing_dict, synapse_flash_idx_dict, curves_path,
                  synapse_flash_curve_dict, isolated_synapses_frames, datamap):
    """
    This function makes 1 step in a flash routine. It loops through all the synapses in a frame to determine whether
    they will start flashing (if they aren't already), or move the flash forward 1 time step if they are flashing, or
    reset the synapse if its flash is over.
    
    :param synapses: A list of all the synapses in the datamap
    :param probability: The probability with which a synapse will start flashing
    :param synapse_flashing_dict: The dict listing whether each synapse is flashing or not
    :param synapse_flash_idx_dict: The dict listing where in their flash each synapse is
    :param curves_path: Path to the .npy file of the light curves being sampled
    :param synapse_flash_curve_dict: The dict listing the sampled flash curve for every synapse
    :param isolated_synapses_frames: The dict listing the isolated synapse frames
    :param datamap: The datamap on which the synapses lie
    
    :return: The updated dicts and datamap
    """
    for idx_syn in range(len(synapses)):
        if numpy.random.binomial(1, probability) and synapse_flashing_dict[idx_syn] is False:
            # can start the flash
            synapse_flashing_dict[idx_syn] = True
            synapse_flash_idx_dict[idx_syn] = 1
            sampled_curve = flash_generator(curves_path)
            synapse_flash_curve_dict[idx_syn] = rescale_data(sampled_curve, to_int=True, divider=3)
            # pyplot.plot(synapse_flash_curve_dict[idx_syn])
            # pyplot.show()
            # exit()

        if synapse_flashing_dict[idx_syn]:
            datamap.whole_datamap[datamap.roi] -= isolated_synapses_frames[idx_syn]
            datamap.whole_datamap[datamap.roi] += isolated_synapses_frames[idx_syn] * \
                                                  synapse_flash_curve_dict[idx_syn][synapse_flash_idx_dict[idx_syn]]
            synapse_flash_idx_dict[idx_syn] += 1
            if synapse_flash_idx_dict[idx_syn] >= 40:
                synapse_flash_idx_dict[idx_syn] = 0
                synapse_flashing_dict[idx_syn] = False

    return synapse_flashing_dict, synapse_flash_idx_dict, synapse_flash_curve_dict, datamap.whole_datamap


def action_execution(action_selected, frame_shape, starting_pixel, pxsize, datamap, frozen_datamap, microscope, pdt,
                     p_ex, p_sted, intensity_map, bleach):
    """
    Executes the selected action. Handles matching the starting_pixel with the number of pixels for which we can image.
    Combines this acquisition with the previously computed intensity_map in the case where a full scan was interupted
    by a flash, for example.
    
    :param action_selected: The selected action (for now, either a full confocal scan (at lower resolution) or a full
                            sted scan).
    :param frame_shape: The shape of the ROI
    :param starting_pixel: The pixel at which the scan starts
    :param pxsize: The acquisition pixel size
    :param datamap: The datamap being imaged
    :param frozen_datamap: A static version of the datamap roi, NOT SURE WHY THIS IS USED IN THE WAY IT IS
    :param microscope: The microscope imageing the datamap
    :param pdt: The pixel dwelltime (either scalar or array of size frame_shape)
    :param p_ex: The excitation power (either scalar or array of size frame_shape)
    :param p_sted: The STED power (either scalar or array of size frame_shape)
    :param intensity_map: The intensity map for the previous acquisition, in case it was interrupted
    :param bleach: Bool determining whether bleaching occurs or not
    
    :return: acq, the acquisition (photons),
             bleached, the bleached datamap,
             datamap, the updated datamap,
             pixel_list, the pixel_list on which the acquisition was occuring, useful to figure out where the next
             acquisition should start
    """
    valid_actions = ["confocal", "sted"]

    if action_selected == "confocal":
        pixel_list = generate_raster_pixel_list(frame_shape[0] * frame_shape[1], starting_pixel, frozen_datamap)
        pixel_list = pixel_list_filter(frozen_datamap, pixel_list, pxsize, datamap.pixelsize, output_empty=True)

        # Cut elements before the starting pixel from the list
        start_idx = pixel_list.index(tuple(starting_pixel))
        pixel_list = pixel_list[start_idx:]
        pixel_list = pixel_list[:microscope.pixel_bank]

    elif action_selected == "sted":
        pixel_list = generate_raster_pixel_list(microscope.pixel_bank, starting_pixel, frozen_datamap)
        pixel_list = pixel_list_filter(frozen_datamap, pixel_list, datamap.pixelsize, datamap.pixelsize,
                                       output_empty=True)

    acq, bleached, intensity_map = microscope.get_signal_and_bleach_fast(datamap, pxsize, pdt, p_ex, p_sted,
                                                                         acquired_intensity=intensity_map,
                                                                         pixel_list=pixel_list,  bleach=bleach,
                                                                         update=False, filter_bypass=True)

    if bleach:
        datamap.whole_datamap = numpy.copy(bleached)

    return acq, intensity_map, datamap, pixel_list


def action_execution_2(action_selected, frame_shape, starting_pixel, pxsize, datamap, frozen_datamap, microscope,
                       pdt, p_ex, p_sted, intensity_map, bleach, t_stack_idx):
    """
    Executes the selected action. Handles matching the starting_pixel with the number of pixels for which we can image.
    Combines this acquisition with the previously computed intensity_map in the case where a full scan was interupted
    by a flash, for example.
    :param action_selected: The selected action (for now, either a full confocal scan (at lower resolution) or a full
                            sted scan).
    :param frame_shape: The shape of the ROI
    :param starting_pixel: The pixel at which the scan starts
    :param pxsize: The acquisition pixel size
    :param datamap: The datamap being imaged
    :param frozen_datamap: A static version of the datamap roi, NOT SURE WHY THIS IS USED IN THE WAY IT IS
    :param microscope: The microscope imageing the datamap
    :param pdt: The pixel dwelltime (either scalar or array of size frame_shape)
    :param p_ex: The excitation power (either scalar or array of size frame_shape)
    :param p_sted: The STED power (either scalar or array of size frame_shape)
    :param intensity_map: The intensity map for the previous acquisition, in case it was interrupted
    :param bleach: Bool determining whether bleaching occurs or not
    :param t_stack_idx: The time step at which we are in our experiment
    :return: acq, the acquisition (photons),
             bleached, the bleached datamap,
             datamap, the updated datamap,
             pixel_list, the pixel_list on which the acquisition was occuring, useful to figure out where the next
             acquisition should start
    """
    valid_actions = ["confocal", "sted"]

    if action_selected == "confocal":
        pixel_list = generate_raster_pixel_list(frame_shape[0] * frame_shape[1], starting_pixel, frozen_datamap)
        pixel_list = pixel_list_filter(frozen_datamap, pixel_list, pxsize, datamap.pixelsize, output_empty=True)

        # Cut elements before the starting pixel from the list
        start_idx = pixel_list.index(tuple(starting_pixel))
        pixel_list = pixel_list[start_idx:]
        pixel_list = pixel_list[:microscope.pixel_bank]

    elif action_selected == "sted":
        pixel_list = generate_raster_pixel_list(microscope.pixel_bank, starting_pixel, frozen_datamap)
        pixel_list = pixel_list_filter(frozen_datamap, pixel_list, datamap.pixelsize, datamap.pixelsize,
                                       output_empty=True)

    acq, bleached_dict, intensity_map = microscope.get_signal_and_bleach_fast_2(datamap, pxsize, pdt, p_ex, p_sted,
                                                                                acquired_intensity=intensity_map,
                                                                                pixel_list=pixel_list, bleach=bleach,
                                                                                update=True, filter_bypass=True,
                                                                                indices=t_stack_idx,
                                                                                raster_func=raster.raster_func_c_self_bleach_split)

    return acq, intensity_map, datamap, pixel_list


def action_execution_g(action_selected, frame_shape, starting_pixel, pxsize, datamap, frozen_datamap, microscope,
                       pdt, p_ex, p_sted, intensity_map, bleach, t_stack_idx):
    """
    Executes the selected action. 
    
    Handles matching the starting_pixel with the number of pixels for which we can image.
    Combines this acquisition with the previously computed intensity_map in the case where a full scan was interupted
    by a flash, for example.
    
    :param action_selected: The selected action (for now, either a full confocal scan (at lower resolution) or a full
                            sted scan).
    :param frame_shape: The shape of the ROI
    :param starting_pixel: The pixel at which the scan starts
    :param pxsize: The acquisition pixel size
    :param datamap: The datamap being imaged
    :param frozen_datamap: A static version of the datamap roi, NOT SURE WHY THIS IS USED IN THE WAY IT IS
    :param microscope: The microscope imageing the datamap
    :param pdt: The pixel dwelltime (either scalar or array of size frame_shape)
    :param p_ex: The excitation power (either scalar or array of size frame_shape)
    :param p_sted: The STED power (either scalar or array of size frame_shape)
    :param intensity_map: The intensity map for the previous acquisition, in case it was interrupted
    :param bleach: Bool determining whether bleaching occurs or not
    :param t_stack_idx: The time step at which we are in our experiment
    :return: acq, the acquisition (photons),
             bleached, the bleached datamap,
             datamap, the updated datamap,
             pixel_list, the pixel_list on which the acquisition was occuring, useful to figure out where the next
             acquisition should start
    """
    valid_actions = ["confocal", "sted"]
    # vérifier si action_selected est dans valid_actions, sinon lancer une erreur

    if action_selected == "confocal":
        pixel_list = generate_raster_pixel_list(frame_shape[0] * frame_shape[1], starting_pixel, frozen_datamap)
        pixel_list = pixel_list_filter(frozen_datamap, pixel_list, pxsize, datamap.pixelsize, output_empty=True)

        # Cut elements before the starting pixel from the list
        start_idx = pixel_list.index(tuple(starting_pixel))
        pixel_list = pixel_list[start_idx:]
        pixel_list = pixel_list[:microscope.pixel_bank]

    elif action_selected == "sted":
        pixel_list = generate_raster_pixel_list(microscope.pixel_bank, starting_pixel, frozen_datamap)
        pixel_list = pixel_list_filter(frozen_datamap, pixel_list, datamap.pixelsize, datamap.pixelsize,
                                       output_empty=True)

    acq, bleached_dict, intensity_map = microscope.get_signal_and_bleach(datamap, pxsize, pdt, p_ex, p_sted,
                                                                         acquired_intensity=intensity_map,
                                                                         pixel_list=pixel_list, bleach=bleach,
                                                                         update=True, filter_bypass=True,
                                                                         indices=t_stack_idx)

    return acq, intensity_map, datamap, pixel_list

def make_path_sane(p):
    """Function to uniformly return a real, absolute filesystem path."""
    # ~/directory -> /home/user/directory
    p = os.path.expanduser(p)
    # A/.//B -> A/B
    p = os.path.normpath(p)
    # Resolve symbolic links
    p = os.path.realpath(p)
    # Ensure path is absolute
    p = os.path.abspath(p)
    return p
