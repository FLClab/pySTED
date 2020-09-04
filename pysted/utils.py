
'''This module contains utilitary functions that used across pySTED.
'''

import numpy
import scipy, scipy.constants, scipy.integrate

# import mis par BT
import math
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


def stack_btmod_pixsize(datamap, data, data_pixelsize, img_pixelsize):
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
    *** VERSION QUI TIENT EN COMPTE LE PIXELSIZE DES DONNÉES BRUTES :)
    '''
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = img_pixelsize_int / data_pixelsize_int
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2
    datamap_to_fill = pxsize_comp_array_maker(img_pixelsize, data_pixelsize, datamap)
    modif_returned_array = numpy.pad(datamap_to_fill, h_pad // 2, mode="constant", constant_values=0)
    row_idx = 0
    col_idx = 0
    for row in range(0, datamap.shape[0], int(ratio)):
        for col in range(0, datamap.shape[1], int(ratio)):
            # j'essaie qqchose qui je pense devrait donner la même chose pour des pixelsize identique,
            # pas certain de comment ça fonctionnerait pour différents pixelsize, mais c'est un bon starting point
            modif_returned_array[row_idx:row_idx+h_pad+1, col_idx:col_idx+w_pad+1] += data * datamap[row, col]
            col_idx += 1
            if col_idx >= int(numpy.ceil(datamap.shape[0] / ratio)):
                col_idx = 0
        row_idx += 1
        if row_idx >= int(numpy.ceil(datamap.shape[1] / ratio)):
            row_idx = 0
    modif_returned_array = modif_returned_array[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]
    return modif_returned_array


def stack_btmod_pixsize_list(datamap, data, data_pixelsize, img_pixelsize, pixel_list):
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
    *** VERSION QUI TIENT EN COMPTE LE PIXELSIZE DES DONNÉES BRUTES :)
    - je veux juste vérifier que le saut entre 2 acquisitions de pixels est d'au moins un ratio
    - je place le résultat dans une matrice de la même taille que les données brutes, peu importe quel autre facteur
    - ceci veut donc dire que même si j'ai un ratio de 2, si je passe une liste de tous les pixels, l'image résultante
      aura la même taille que les données brutes, et non les dimensions divisées par 2 :)
    - si je passe une pixel_list d'un seul pixel, je pars alors en mode "trouver un poil"
        ce mode part du pixel de la pixel_list, vérifie s'il y a des poils proches, embarque sur ceux si si oui, passe
        au prochain pixel du raster scan si non (mieux décrit dans mes notes sur iPad)
    '''
    if pixel_list is None:
        raise Exception("No pixel_list passed bruh, programmer error if we ever get here :)")
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = img_pixelsize_int / data_pixelsize_int
    # est-ce que je dois aussi m'assurer que le pixelsize fit avec la shape de l'image? i.e. que les 2 shapes soient
    # divisibles par le pixelsize? ou le ratio? ???
    h_pad, w_pad = int(data.shape[0] / 2) * 2, int(data.shape[1] / 2) * 2
    modif_returned_array = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))
    if len(pixel_list) == 1:
        # mode cabochon pour imager uniquement les pixels avec des molécules dedans, mais genre de façon quon le sait
        # pas d'Avance, tkt c'était plus pour mon personnal curiosity qu'autre chose :)
        print("YES")
        print(f"0 + int(ratio) - 1 = {0 + int(ratio) - 1}")
        # Verification matrix to see if a pixel has already been imaged
        pixels_added_to_list = numpy.zeros(datamap.shape)

        # Creating a list for a raster scan starting at the passed pixel
        # The goal is to continue raster scanning until we find a molecule
        raster_scan_list = pixel_sampling(datamap, mode="all")
        start_pos_idx = raster_scan_list.index(pixel_list[0])
        raster_scan_list = raster_scan_list[start_pos_idx + 1:]

        # Apply laser to the pixels, still have to watch out for pixel jumping due to ratio
        previous_pixel = None
        went_in_if = 0
        for pixel in pixel_list:
            pixels_to_add_to_list = numpy.zeros(datamap.shape)
            row = pixel[0]
            col = pixel[1]
            if previous_pixel is not None:
                if abs(row - previous_pixel[0]) < ratio and abs(col - previous_pixel[1]) < ratio:  # absolues?
                    continue

            modif_returned_array[row:row + h_pad + 1, col:col + w_pad + 1] += data * datamap[row, col]

            # Finding the area covered by the gaussian in the datamap
            if row - int(data.shape[0] / 2) < 0:
                upper_edge = 0
            else:
                upper_edge = row - int(data.shape[0] / 2)
            if row + int(data.shape[0] / 2) >= datamap.shape[0]:
                lower_edge = datamap.shape[0] - 1
            else:
                lower_edge = row + int(data.shape[0] / 2)
            if col - int(data.shape[1] / 2) < 0:
                left_edge = 0
            else:
                left_edge = col - int(data.shape[1] / 2)
            if col + int(data.shape[1] / 2) >= datamap.shape[1]:
                right_edge = datamap.shape[1] - 1
            else:
                right_edge = col + int(data.shape[1] / 2)

            # basically il me faut un if datamap[upper:lower, left:right] > 0 -> mettre ces pixels dans la liste,
            # sinon ajouter le prochain pixel du raster scan
            if numpy.any(datamap[upper_edge:lower_edge, left_edge:right_edge] > 0):
                went_in_if = 1
                pixels_to_add_to_list[upper_edge:lower_edge, left_edge:right_edge] = \
                    datamap[upper_edge:lower_edge, left_edge:right_edge] > 0
                xd = numpy.where(pixels_to_add_to_list > 0)
                for pixel_to_add in zip(xd[0], xd[1]):
                    if pixels_added_to_list[pixel_to_add[0], pixel_to_add[1]] == 0:
                        pixel_list.append(pixel_to_add)
                        pixels_added_to_list[pixel_to_add[0], pixel_to_add[1]] = 1
            elif went_in_if == 0:
                # ajouter le prochain pixel du raster scan :)
                pixel_list.append(raster_scan_list[0 + int(ratio) - 1])   # 0 + int(ratio) - 1 ?
                # raster_scan_list.pop(0)
                for i in range(0, int(ratio)):
                    raster_scan_list.pop(0)

            previous_pixel = pixel

    else:
        previous_pixel = None
        invalid_px_in_sequence = 0
        valid_px_in_sequence = 0
        for pixel in pixel_list:
            row = pixel[0]
            col = pixel[1]
            if previous_pixel is not None:
                # if row - previous_pixel[0] < ratio and col - previous_pixel[1] < ratio:  # absolues?
                if abs(row - previous_pixel[0]) < ratio and abs(col - previous_pixel[1]) < ratio:  # absolues?
                    invalid_px_in_sequence += 1
                    continue
            valid_px_in_sequence += 1
            modif_returned_array[row:row + h_pad + 1, col:col + w_pad + 1] += data * datamap[row, col]
            # uh oh, si j'itère à côté, est-ce que je devrais acquérir du signal? j'imagine que oui, mais ça va à
            # l'encontre de comment le calcul est décrit dans l'exemple de la doc...
            # ce que je crois quil faudrait faire :
            # modif_returned_array[row:row + h_pad + 1, col:col + w_pad + 1] += data * padded_datamap[row:row + h_pad + 1,
            #                                                                                         col:col + w_pad + 1]
            previous_pixel = pixel

        # print(f"invalid_px_in_sequence = {invalid_px_in_sequence}")
        # print(f"valid_px_in_sequence = {valid_px_in_sequence}")
    return modif_returned_array[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]


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
    :returns: A 2D array shaped like *datamap*.
    '''
    filtered_pixel_list = pixel_list_filter(pixel_list, img_pixelsize, data_pixelsize)
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
        cell_size = 100

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


def pxsize_comp(img_pixelsize, data_pixelsize):
    """
    Function to compare the pixelsize of the raw data to the acquisition pixelsize in order to verify compatibility.
    :param image_pixelsize: acquisition pixelsize. Has to be a multiple of data_pixelsize
    :param data_pixelsize: raw data pixelsize.
    :returns: Integer values of the pixelsizes which can later be used to compute ratios and stuff :)
    *** OBSOLETE FUNCTION DUE TO BUGS, USE pxsize_comp2 UNTIL I MAKE SURE THE 2ND VERSION WORKS PERFECTLY SO I CAN
    DELETE THIS ONE ***
    """
    # VÉRIFIER ET TESTER TOUT CELA, DEVRAIT MARCHER :)
    # je pense qu'il faut une condition de plus sur la forme de l'image...
    if data_pixelsize is None:
        print(f"no data pixelsize, set as image pixelsize = {img_pixelsize}...")
        data_pixelsize = img_pixelsize
    img_pixelsize_int = float(str(img_pixelsize)[0: str(img_pixelsize).find('e')])
    data_pixelsize_int = float(str(data_pixelsize)[0: str(data_pixelsize).find('e')])
    pixelsize_exp = int(str(img_pixelsize)[str(img_pixelsize).find('e') + 1:])
    data_pixelsize_exp = int(str(data_pixelsize)[str(data_pixelsize).find('e') + 1:])
    exp = pixelsize_exp - data_pixelsize_exp
    img_pixelsize_int *= 10 ** exp
    if img_pixelsize < data_pixelsize or img_pixelsize_int % data_pixelsize_int != 0:
        # lancer une erreur ou qqchose si j'arrive ici
        raise Exception("pixelsize has to be a multiple of data_pixelsize")
    return img_pixelsize_int, data_pixelsize_int

def pxsize_comp2(img_pixelsize, data_pixelsize):
    """
    Try number 2 for my float comparison function that hopefully will give the right values this time :)
    :param img_pixelsize: Acquisition pixel size. Has to be a multiple of data_pixelsize (m).
    :param data_pixelsize: Raw data pixelsize (m).
    :returns: Integer values of the pixelsizes which can later be used to compute ratios and stuff :)
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
    img_pixelsize_int = int(img_pixelsize_int)
    data_pixelsize_int = int(data_pixelsize_int)
    test3 = img_pixelsize_int % data_pixelsize_int
    if img_pixelsize < data_pixelsize or not math.isclose(test3, 0):
        raise Exception("img_pixelsize has to be a multiple of data_pixelsize")
    return img_pixelsize_int, data_pixelsize_int


def pxsize_comp_array_maker(img_pixelsize, data_pixelsize, datamap):
    """
    Function which utilizes the ratio between the image pixelsize and the datamap pixelsize to create an appropriatly
    sized output datamap for a normal raster scan acquisition with ratio jumps between laser applications. This assures
    that the laser application is placed in the appropriate acquisition output pixel.
    :param img_pixelsize: The image pixelsize (m)
    :param data_pixelsize: The datamap pixelsize (m)
    :param datamap: The datamap on which the acquisition is made
    :returns: An empty datamap of shape (ceil(datamap.shape[0] / ratio), ceil(datamap.shape[1] / ratio))
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = img_pixelsize_int / data_pixelsize_int
    nb_rows = int(numpy.ceil(datamap.shape[0] / ratio))
    nb_cols = int(numpy.ceil(datamap.shape[1] / ratio))
    datamap_to_fill = numpy.zeros((nb_rows, nb_cols))
    return datamap_to_fill


def pxsize_grid(img_pixelsize, data_pixelsize, datamap):
    """
    Function which creates a grid of the pixels which can be iterated on based on the ratio between img_pixelsize and
    data_pixelsize. Imagine the laser is fixed on a grid and can only make discrete movements, and this grid size is
    determined by the ratio
    :param img_pixelsize: Size of the minimum distance the laser must do between acquisitions (m). Must be a multiple of
                          data_pixelsize.
    :param data_pixelsize: Size of a pixel of the datamap (m).
    :param datamap: Raw molecule dispotion on which we wish to do an acquisition.
    :returns: A list of the pixels which can be iterated on (?)
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
    :returns: the ratio between pixel sizes
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = int(img_pixelsize_int / data_pixelsize_int)
    return ratio


def image_squisher(datamap, data_pixelsize, img_pixelsize):
    """
    le but est d'essayer de squisher une image en fonction du ratio entre data_pixelsize et img_pixelsize :)
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = int(img_pixelsize_int / data_pixelsize_int)
    squished_datamap = numpy.zeros((int(datamap.shape[0] / ratio), int(datamap.shape[1] / ratio)))
    row_idx = 0
    col_idx = 0
    for row in range(0, datamap.shape[0], int(ratio)):
        for col in range(0, datamap.shape[1], int(ratio)):
            # j'essaie qqchose qui je pense devrait donner la même chose pour des pixelsize identique,
            # pas certain de comment ça fonctionnerait pour différents pixelsize, mais c'est un bon starting point
            squished_datamap[row_idx, col_idx] = numpy.sum(datamap[row:row+ratio, col:col+ratio])
            col_idx += 1
            if col_idx >= int(datamap.shape[0] / ratio):
                col_idx = 0
        row_idx += 1
        if row_idx >= int(datamap.shape[1] / ratio):
            row_idx = 0
    return squished_datamap


def mse_calculator(array1, array2):
    """
    Compute the RMS between two arrays. Must be of same size
    :param array1: First array
    :param array2: Second array
    :returns: Mean squarred error of the 2 arrays
    """
    # jpourrais mettre un ti qqchose ici pour vérifier si les 2 arrays ont la même forme, jsp si c'est nécessaire
    array_diff = numpy.absolute(array1 - array2)
    array_diff_squared = numpy.square(array_diff)
    mean_squared_error = float(numpy.sum(array_diff_squared) / (array1.shape[0] * array1.shape[1]))
    return mean_squared_error


def pixels_iterated_on(img_pxsz, data_pxsz, datamap, laser):
    """
    Le but ici est de vérifier sur quels pixels j'itère quand je fais un raster scan avec un ratio entre les pxsz
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pxsz, data_pxsz)
    ratio = img_pixelsize_int / data_pixelsize_int
    h_pad, w_pad = int(laser.shape[0] / 2) * 2, int(laser.shape[1] / 2) * 2
    datamap_to_fill = pxsize_comp_array_maker(img_pxsz, data_pxsz, datamap)
    print(f"datamap_to_fill.shape = {datamap_to_fill.shape}")
    modif_returned_array = numpy.pad(datamap_to_fill, h_pad // 2, mode="constant", constant_values=0)
    print(f"modif_returned_array.shape = {modif_returned_array.shape}")
    row_idx = 0
    col_idx = 0
    for row in range(0, datamap.shape[0], int(ratio)):
        for col in range(0, datamap.shape[1], int(ratio)):
            if datamap[row, col] > 0:
                modif_returned_array[(row_idx + row_idx + h_pad + 1) // 2, (col_idx + col_idx + w_pad + 1) // 2] += 1
            col_idx += 1
            if col_idx >= int(numpy.ceil(datamap.shape[0] / ratio)):
                col_idx = 0
        row_idx += 1
        if row_idx >= int(numpy.ceil(datamap.shape[1] / ratio)):
            row_idx = 0

    modif_returned_array = modif_returned_array[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]
    return modif_returned_array


def pixels_iterated_on_list_skipping(img_pxsz, data_pxsz, datamap, laser, pixel_list):
    """
    Fonction test pour trouver une bonne façon de faire le pixel skipping quand on est dans un cas autre que raster sur
    tous les pixels :)
    JUSTE POUR LES TESTS À DELETE
    """
    if pixel_list is None:
        raise Exception("No pixel_list passed bruh, programmer error if we ever get here :)")
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pxsz, data_pxsz)
    ratio = img_pixelsize_int / data_pixelsize_int
    # est-ce que je dois aussi m'assurer que le pixelsize fit avec la shape de l'image? i.e. que les 2 shapes soient
    # divisibles par le pixelsize? ou le ratio? ???
    h_pad, w_pad = int(laser.shape[0] / 2) * 2, int(laser.shape[1] / 2) * 2
    modif_returned_array = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))
    iterated_pixels_array = numpy.zeros((datamap.shape[0] + h_pad, datamap.shape[1] + w_pad))

    print("Test function running :)")
    padded_datamap = numpy.pad(numpy.copy(datamap), h_pad // 2, mode="constant")
    previous_pixel = None
    invalid_px_in_sequence = 0
    valid_px_in_sequence = 0
    for pixel in pixel_list:
        row = pixel[0]
        col = pixel[1]
        if previous_pixel is not None:
            # if row - previous_pixel[0] < ratio and col - previous_pixel[1] < ratio:  # absolues?
            if abs(row - previous_pixel[0]) < ratio and abs(col - previous_pixel[1]) < ratio:  # absolues?
                invalid_px_in_sequence += 1
                continue
        valid_px_in_sequence += 1
        modif_returned_array[row:row + h_pad + 1, col:col + w_pad + 1] += laser * datamap[row, col]
        iterated_pixels_array[(row + row + h_pad + 1) // 2, (col + col + w_pad + 1) // 2] += 1
        previous_pixel = pixel

    print(f"invalid_px_in_sequence = {invalid_px_in_sequence}")
    print(f"valid_px_in_sequence = {valid_px_in_sequence}")
    return modif_returned_array[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)], \
           iterated_pixels_array[int(h_pad / 2):-int(h_pad / 2), int(w_pad / 2):-int(w_pad / 2)]


def pixel_list_filter(pixel_list, img_pixelsize, data_pixelsize):
    """
    Function to pre-filter a pixel list. Depending on the ratio between the data_pixelsize and acquisition pixelsize,
    a certain number of pixels must be skipped between laser applications.
    :param pixel_list: The list of pixels passed to the acquisition function, which needs to be filtered
    :param img_pixelsize: The acquisition pixelsize (m)
    :param data_pixelsize: The data pixelsize (m)
    :returns: A filtered version of the input pixel_list, from which the pixels which can't be iterated over due to the
              pixel sizes have been removed
    """
    img_pixelsize_int, data_pixelsize_int = pxsize_comp2(img_pixelsize, data_pixelsize)
    ratio = int(img_pixelsize_int / data_pixelsize_int)

    previous_pixel = None
    new_pixel_list = []
    for (row, col) in pixel_list:
        if previous_pixel is not None:
            if abs(row - previous_pixel[0]) < ratio and abs(col - previous_pixel[1]) < ratio:
                continue

        new_pixel_list.append((row, col))
        previous_pixel = (row, col)
    return new_pixel_list
