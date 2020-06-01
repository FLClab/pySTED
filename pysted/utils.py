
'''This module contains utilitary functions that used across pySTED.
'''

import numpy
import scipy, scipy.constants, scipy.integrate

# import mis par BT
import random

# import mis par BT pour des tests :)
from matplotlib import pyplot
import time


def approx_binomial(n, p, size=None):
    '''Sample (64-bit) from a binomial distribution using the normal approximation.
    
    :param n: The number of trials (int or array of ints).
    :param p: The probability of success (float).
    :param size: The shape of the output (int or tuple of ints, optional).
    :returns: 64-bit int or array of 64-bit ints.
    '''
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
    :returns: A tuple of the angle :math:`\\theta` and the lenght :math:`rho`.
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
    :returns: The integration result as a complex number.
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
    :returns: The full width at half maximum.
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
    :returns: A tuple of the outer and inner width at half maximum.
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
    :returns: A 2D array.
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
    :returns: A tuple of copies of the given *images* resized to the size of the
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
    :returns: The result of the function, same shape as *x*.
    '''
    return 1 / (a * x + 1)


def inverse_exponential(x, a=1):
    '''Evaluate the inverse function :math:`1 / e^{ax}`.
    
    :param x: An integer or array.
    :param a: The scale of *x*.
    :returns: The result of the function, same shape as *x*.
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
    :returns: A 2D array shaped like *datamap*.
    '''
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2
    frame = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))
    positions = numpy.where(datamap > 0)
    numbers = datamap[positions]
    for nb, y, x in zip(numbers, *positions):
        frame[y:y+h_pad+1, x:x+w_pad+1] += data * nb
    return frame[int(h_pad/2):-int(h_pad/2), int(w_pad/2):-int(w_pad/2)]


def stack_btmod(datamap, data):
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
        >>> utils.stack_btmod(datamap, data)
        numpy.array([[6, 4, 0, 0],
                     [4, 2, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

    :param datamap: A 2D array indicating how many data are positioned in every
    :param data: A 2D array containing the data to replicate.
    :returns: A 2D array shaped like *datamap*.
    ******* VERSION QUE BT MODIFIE POUR DES TESTS DE PRISE D'IMAGE NON-RASTER SCANNED *******
    '''
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2   # garde ça
    frame = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))   # garde ça
    print("in stack_btmod")
    for y in range(datamap.shape[0]):
        for x in range(datamap.shape[1]):
            frame[y:y + h_pad + 1, x:x + w_pad + 1] += data * datamap[y, x]
    return frame[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]


def stack_btmod_list(datamap, data):
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
        >>> utils.stack_btmod(datamap, data)
        numpy.array([[6, 4, 0, 0],
                     [4, 2, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

    :param datamap: A 2D array indicating how many data are positioned in every
    :param data: A 2D array containing the data to replicate.
    :returns: A 2D array shaped like *datamap*.
    ******* VERSION QUE BT MODIFIE POUR DES TESTS DE PRISE D'IMAGE NON-RASTER SCANNED *******
    '''
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2   # garde ça
    frame = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))   # garde ça
    pixel_list = pixel_sampling(datamap)
    print("in stack_btmod_list")
    for pixel in pixel_list:
        frame[pixel[0]:pixel[0] + h_pad + 1, pixel[1]:pixel[1] + w_pad + 1] += data * datamap[pixel[0], pixel[1]]
    return frame[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]


def stack_btmod_list_shuffle(datamap, data):
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
        >>> utils.stack_btmod(datamap, data)
        numpy.array([[6, 4, 0, 0],
                     [4, 2, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

    :param datamap: A 2D array indicating how many data are positioned in every
    :param data: A 2D array containing the data to replicate.
    :returns: A 2D array shaped like *datamap*.
    ******* VERSION QUE BT MODIFIE POUR DES TESTS DE PRISE D'IMAGE NON-RASTER SCANNED + SHUFFLING *******
    '''
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2   # garde ça
    frame = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))   # garde ça
    pixel_list = pixel_sampling(datamap)
    random.shuffle(pixel_list)
    print("in stack_btmod_list_shuffle")
    for pixel in pixel_list:
        frame[pixel[0]:pixel[0] + h_pad + 1, pixel[1]:pixel[1] + w_pad + 1] += data * datamap[pixel[0], pixel[1]]
    return frame[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]


def stack_btmod_checkers(datamap, data):
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
        >>> utils.stack_btmod(datamap, data)
        numpy.array([[6, 4, 0, 0],
                     [4, 2, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

    :param datamap: A 2D array indicating how many data are positioned in every
    :param data: A 2D array containing the data to replicate.
    :returns: A 2D array shaped like *datamap*.
    ******* VERSION QUI SÉLECTIONNE JUSTE UNE RÉGION CHECKERS *******
    '''
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2   # garde ça
    frame = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))   # garde ça
    pixel_list = pixel_sampling(datamap, mode="checkers")
    print("in stack_btmod_list")
    for pixel in pixel_list:
        frame[pixel[0]:pixel[0] + h_pad + 1, pixel[1]:pixel[1] + w_pad + 1] += data * datamap[pixel[0], pixel[1]]
    return frame[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]


def pixel_sampling(datamap, mode="all"):
    '''
    Function to test different pixel sampling methods, instead of simply imaging pixel by pixel
    :param datamap: A 2D array of the data to be imaged, used for its shape
    :returns: A list (?) containing all the pixels in the order in which we want them to be imaged

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
        cell_size = 10

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

    return pixel_list

