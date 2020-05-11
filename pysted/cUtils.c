#include <Python.h>

#define _USE_MATH_DEFINES
#include <math.h>

static PyObject* calculate_amplitude(PyObject *self, PyObject* args) {
    /* Compute and return the amplitude of a 2D gaussian at the point (x, y) at
    the given radius.
    
    args : double, double, double
    return : double
    */
    double x, y, d_airy;
    PyArg_ParseTuple(args, "ddd", &x, &y, &d_airy);
    
    double d = sqrt(pow(x, 2) + pow(y, 2));
    double d_scaled = d / d_airy;
    double amplitude;
    if (d_scaled == 0) {
        amplitude = 1;
    } else {
        amplitude = pow(j1(M_PI * d_scaled) / (M_PI * d_scaled), 2);
    }
    return Py_BuildValue("d", amplitude);
}

static PyMethodDef cMethods[] = {
    {"calculate_amplitude", calculate_amplitude, METH_VARARGS, "Return the amplitude of a 2D gaussian at the point (x, y) at the given radius."},
    {NULL, NULL, 0, NULL} // sentinel (?!?)
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cUtils", // m_name
    "C version of some tools.", // m_doc
    -1, // m_size
    cMethods, // m_methods
    NULL, // m_reload
    NULL, // m_traverse
    NULL, // m_clear
    NULL, // m_free
};

PyMODINIT_FUNC
PyInit_cUtils(void) {
    PyObject *lModule = PyModule_Create(&moduledef);
    return lModule;
}

