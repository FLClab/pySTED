

import numpy

def _multiple_lines(coords):
    """
    Generates a semgented line from a list of coordinates
    :param coords: A (N, 2) `numpy.ndarray` of (y, x) coordinates
    :returns : rr, cc -> (N,) ndarray of int
               Indices of pixels that belong to the line.
               May be used to directly index into an array, e.g.
               ``img[rr, cc] = 1``.
    """

    cdef Py_ssize_t totlen = 0
    cdef Py_ssize_t r, c, dr, dc
    cdef Py_ssize_t sr, sc, d, i, j
    cdef Py_ssize_t itt

    for i in range(len(coords) - 1):
        r0, c0 = coords[i]
        r1, c1 = coords[i + 1]
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        totlen += max(dc, dr) + 1

    cdef Py_ssize_t[::1] rr = numpy.zeros(totlen, dtype=numpy.intp)
    cdef Py_ssize_t[::1] cc = numpy.zeros(totlen, dtype=numpy.intp)

    # Itterates through the pair of points
    itt = 0
    for i in range(len(coords) - 1):

        r, c = coords[i]
        r1, c1 = coords[i + 1]
        steep = 0

        dr = abs(r1 - r)
        dc = abs(c1 - c)

        if (c1 - c) > 0:
            sc = 1
        else:
            sc = -1
        if (r1 - r) > 0:
            sr = 1
        else:
            sr = -1
        if dr > dc:
            steep = 1
            c, r = r, c
            dc, dr = dr, dc
            sc, sr = sr, sc
        d = (2 * dr) - dc

        for j in range(dc):
            if steep:
                rr[itt] = c
                cc[itt] = r
            else:
                rr[itt] = r
                cc[itt] = c
            while d >= 0:
                r = r + sr
                d = d - (2 * dc)
            c = c + sc
            d = d + (2 * dr)

            itt += 1

        rr[itt] = r1
        cc[itt] = c1

        itt += 1

    return numpy.asarray(rr), numpy.asarray(cc)