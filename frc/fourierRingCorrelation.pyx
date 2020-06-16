
import numpy
from libc.math cimport sqrt, abs

cdef frc(complex[:,:] fimg1, complex[:,:] fimg2, int h, int w, int rmax):

    cdef int yc, xc
    cdef int r, n
    cdef double absA, absB
    cdef complex corr, im1, im2

    yc, xc = int((h + 1) / 2) + 1, int((w + 1) / 2) + 1
    rmax = min([w - xc, h - yc])
    fscList = numpy.zeros((rmax, ), dtype=numpy.dtype("double"))
    nPx = numpy.zeros((rmax, ), dtype=numpy.dtype("int"))
    for r in range(rmax):
        corr = 0
        absA = 0
        absB = 0
        n = 0
        for x in range(w):
            for y in range(h):
                if (((x - xc) * (x - xc) + (y - yc) * (y - yc)) >= (r - 0.5) * (r - 0.5)) \
                    & (((x - xc) * (x - xc) + (y - yc) * (y - yc)) < (r + 0.5) * (r + 0.5)):

                    im1 = fimg1[y][x]
                    im2 = fimg2[y][x]
                    corr = corr + im1 * im2.conjugate()
                    absA = absA + abs(im1 * im1)
                    absB = absB + abs(im2 * im2)

                    n += 1
        fscList[r] = abs(corr) / (sqrt(absA * absB) + 0.000001)
        nPx[r] = n

    return fscList, nPx

cpdef return_result(complex[:,:] fimg1, complex[:,:] fimg2, int h, int w, int rmax):
    return frc(fimg1, fimg2, h, w, rmax)
