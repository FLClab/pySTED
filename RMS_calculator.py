import numpy

def rms_calculator(array1, array2):
    # assume que les 2 arrays sont de la même taille
    array_diff = numpy.absolute(array1 - array2)
    array_diff_squared = numpy.square(array_diff)
    mean_squared_error = float(numpy.sum(array_diff_squared) / (array1.shape[0] * array1.shape[1]))

    """print(f"squared_min = {numpy.amin(array_diff_squared)}")
    print(f"squared_max = {numpy.amax(array_diff_squared)}")
    print(f"sum of array = {numpy.sum(array_diff_squared)}")
    print(f"sum of maxes = {numpy.amax(array_diff_squared) * array1.shape[0] * array1.shape[1]}")"""
    return mean_squared_error