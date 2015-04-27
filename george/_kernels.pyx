# distutils: language = c++
from __future__ import division

cimport cython
cimport kernels

import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool as bool_t

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def _rebuild(kernel_spec):
    return CythonKernel(kernel_spec)

# def _rebuild_kernderiv(kernel_spec):
#     # I am not sure what this does
#     return CythonDerivKernel(kernel_spec)



cdef class CythonKernel:

    cdef kernels.Kernel* kernel
    cdef object kernel_spec

    def __cinit__(self, kernel_spec):
        self.kernel_spec = kernel_spec
        self.kernel = kernels.parse_kernel(kernel_spec)

    def __reduce__(self):
        return _rebuild, (self.kernel_spec, )

    def __dealloc__(self):
        del self.kernel

    @cython.boundscheck(False)
    def value_symmetric(self, np.ndarray[DTYPE_t, ndim=2] x):
        cdef unsigned int n = x.shape[0], ndim = x.shape[1]
        if self.kernel.get_ndim() != ndim:
            raise ValueError("Dimension mismatch")

        # Build the kernel matrix.
        cdef double value
        cdef unsigned int i, j, delta = x.strides[0]
        cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((n, n), dtype=DTYPE)
        for i in range(n):
            print "calling value method in _pyx file"
            k[i, i] = self.kernel.value(<double*>(x.data + i*delta),
                                        <double*>(x.data + i*delta))
            for j in range(i + 1, n):
                value = self.kernel.value(<double*>(x.data + i*delta),
                                          <double*>(x.data + j*delta))
                k[i, j] = value
                k[j, i] = value

        return k

    @cython.boundscheck(False)
    def value_general(self, np.ndarray[DTYPE_t, ndim=2] x1,
                      np.ndarray[DTYPE_t, ndim=2] x2):
        # Parse the input kernel spec.
        cdef unsigned int n1 = x1.shape[0], ndim = x1.shape[1], n2 = x2.shape[0]
        if self.kernel.get_ndim() != ndim or x2.shape[1] != ndim:
            raise ValueError("Dimension mismatch")

        # Build the kernel matrix.
        cdef double value
        cdef unsigned int i, j, d1 = x1.strides[0], d2 = x2.strides[0]
        cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((n1, n2), dtype=DTYPE)
        for i in range(n1):
            for j in range(n2):
                k[i, j] = self.kernel.value(<double*>(x1.data + i*d1),
                                            <double*>(x2.data + j*d2))

        return k

    @cython.boundscheck(False)
    def gradient_symmetric(self, np.ndarray[DTYPE_t, ndim=2] x):
        # Check the input dimensions.
        cdef unsigned int n = x.shape[0], ndim = x.shape[1]
        if self.kernel.get_ndim() != ndim:
            raise ValueError("Dimension mismatch")

        # Get the number of parameters.
        cdef unsigned int size = self.kernel.size()

        # Build the gradient matrix.
        cdef double value
        cdef np.ndarray[DTYPE_t, ndim=3] g = np.empty((n, n, size), dtype=DTYPE)
        cdef unsigned int i, j, k, delta = x.strides[0]
        cdef unsigned int dx = g.strides[0], dy = g.strides[1]
        for i in range(n):
            self.kernel.gradient(<double*>(x.data + i*delta),
                                 <double*>(x.data + i*delta),
                                 <double*>(g.data + i*dx + i*dy))
            for j in range(i + 1, n):
                self.kernel.gradient(<double*>(x.data + i*delta),
                                     <double*>(x.data + j*delta),
                                     <double*>(g.data + i*dx + j*dy))
                for k in range(size):
                    g[j, i, k] = g[i, j, k]

        return g

    @cython.boundscheck(False)
    def gradient_general(self, np.ndarray[DTYPE_t, ndim=2] x1,
                         np.ndarray[DTYPE_t, ndim=2] x2):
        cdef unsigned int n1 = x1.shape[0], ndim = x1.shape[1], n2 = x2.shape[0]
        if self.kernel.get_ndim() != ndim or x2.shape[1] != ndim:
            raise ValueError("Dimension mismatch")

        # Get the number of parameters.
        cdef unsigned int size = self.kernel.size()

        # Build the gradient matrix.
        cdef double value
        cdef np.ndarray[DTYPE_t, ndim=3] g = np.empty((n1, n2, size),
                                                      dtype=DTYPE)
        cdef unsigned int i, j, k, d1 = x1.strides[0], d2 = x2.strides[0]
        cdef unsigned int dx = g.strides[0], dy = g.strides[1]
        for i in range(n1):
            for j in range(n2):
                self.kernel.gradient(<double*>(x1.data + i*d1),
                                     <double*>(x2.data + j*d2),
                                     <double*>(g.data + i*dx + j*dy))

        return g


# cdef class CythonDerivKernel(CythonKernel):
#     """new class to contain all commonly used derivative features
#
#     :note:
#     we do not have to call `CythonKernel.__cinit__` because
#     Cython calls it for us automatically
#
#     See first answer at
#     http://stackoverflow.com/questions/10298371/cython-and-c-inheritance
#     """
#     cdef np.ndarray __pairsOfBIndices__
#     cdef np.ndarray __pairsOfCIndices__
#
#     @cython.boundscheck(False)
#     def __cinit__(self):
#         self.__pairsOfBIndices__ = \
#             np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2],
#                      [2, 3, 0, 1], [1, 3, 0, 2], [1, 2, 0, 3]],
#                      dtype=DTYPE_int)
#
#         self.__pairsOfCIndices__ = \
#             np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]],
#                      dtype=DTYPE_int)
#
#     def __X__(self, np.ndarray[DTYPE_t, ndim=2] coords,
#               unsigned int m, unsigned int n, unsigned int spat_ix):
#         return coords[m, spat_ix] - coords[n, spat_ix]
#
#     def __termA__(self, np.ndarray[DTYPE_t, ndim=2] coords,
#                   np.ndarray[DTYPE_int_t, ndim=1] ix,
#                   unsigned int m, unsigned int n,
#                   bool_t debug=False):
#         """
#         # the constructor also needs the coordinates
#         Compute term 1 in equation (24) without leading factors of $\beta^4$
#
#         :params coords: numpy array,
#             shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions
#
#         :params ix: np array of 4 integers,
#             denote the spatial subscripts for the ker deriv
#
#         :beta: float,
#             inverse length
#
#         .. math:
#             X_i X_j X_h X_k
#         """
#         cdef double term = 1.
#
#         if debug:
#             print "---------------------------------"
#         for i in ix:
#             if debug:
#                 print "parts of term A = {0}".format(term)
#             term *= self.__X__(coords, m, n, i)
#
#         if debug:
#             print "m = {0}, n = {1}".format(m, n)
#             print "indices of term A ix = {0}".format(ix)
#             print "term A = {0}".format(term)
#             print "---------------------------------"
#
#         return term
#
#     def __termB__(self, np.ndarray[DTYPE_t, ndim=2] coords,
#                   np.ndarray[DTYPE_int_t, ndim=1] ix,
#                   unsigned int m, unsigned int n,
#                   np.ndarray[DTYPE_t, ndim=1] metric, debug=False):
#         """
#         Compute term 2 in equation (24) without leading factors of $\beta^3$
#
#         :param coords: numpy array,
#             shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions
#
#         :param ix: list of 4 integers,
#             denote the spatial subscripts for the ker deriv,
#             assumes to take the form [a, b, c, d]
#
#         :param m: integer
#             denote the row index of covariance matrix, or obs_no
#
#         :params n: integer
#             denote the col index of covariance matrix, or obs_no
#
#         :param metric: list of floats
#             should be of dimension 2, we assume diagonal metric here
#
#         .. math:
#             X_a X_b D_{cd} \delta_{cd}
#         """
#         if debug is True:
#             print "---------------------------------"
#             print "m = {0}, n = {1}".format(m, n)
#             print "indices of term B = {0}".format(ix)
#             print "termB = {0}".format(self.__X__(coords, m, n, ix[0]) *
#             self.__X__(coords, m, n, ix[1]) *
#             metric[ix[2]])
#
#         if ix[2] != ix[3]:
#             return 0
#
#         return self.__X__(coords, m, n, ix[0]) * \
#             self.__X__(coords, m, n, ix[1]) * \
#             metric[ix[2]]
#
#     def __termC__(self, np.ndarray[DTYPE_t, ndim=2] coords,
#                   np.ndarray[DTYPE_int_t, ndim=1] ix,
#                   np.ndarray[DTYPE_t, ndim=1] metric,
#                   bool_t debug=False):
#         """
#         Compute term 3 in equation (24) without leading factor of $\beta^2$
#
#         :param coords: numpy array,
#             shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions
#
#         :param ix: list of 4 integers,
#             denote the spatial subscripts for the ker deriv,
#             assumes to take the form [a, b, c, d]
#
#         :param metric: array of floats
#
#         .. math:
#             D_{ab} D_{cd} \delta_{ab} \delta_{cd}
#         """
#         if debug:
#             print "---------------------------------"
#             print "indices of term C = {0}".format(ix)
#             print "term C = {0}".format(metric[ix[2]] * metric[ix[0]])
#
#         if ix[0] != ix[1]:
#             return 0
#
#         if ix[2] != ix[3]:
#             return 0
#
#         return metric[ix[2]] * metric[ix[0]]
#
#     def __Sigma4thDeriv__(self, double beta,
#                           np.ndarray[DTYPE_t, ndim=2] coords,
#                           np.ndarray[DTYPE_int_t, ndim=1] ix,
#                           unsigned int m, unsigned int n,
#                           np.ndarray[DTYPE_t, ndim=1] metric,
#                           bool_t debug=False):
#         """
#         Gather the 10 terms for the 4th derivative of each Sigma
#         given the ix for each the derivatives are taken w.r.t.
#
#         :params corr: float
#             value of the correlation parameter in the ExpSquaredKernel
#
#         :params coords: 2D numpy array
#             with shape (nObs, ndim)
#
#         :params ix: list of 4 integers
#         """
#         allTermBs = 0
#         combBix = \
#             np.array([[ix[i] for i in self.__pairsOfBIndices__[j]]
#                       for j in range(6)])
#
#         # combBix is the subscript indices combination for B terms
#         for i in range(6):
#             allTermBs += self.__termB__(coords, combBix[i], m, n, metric)
#
#         allTermCs = 0
#         combCix = \
#             np.array([[ix[i] for i in self.__pairsOfCIndices__[j]]
#                       for j in range(3)])
#
#         for i in range(3):
#             allTermCs += self.__termC__(coords, combCix[i], metric)
#
#         termA = self.__termA__(coords, ix, m, n)
#
#         if debug:
#             print "---------------------------------"
#             print "m = {0}, n = {1}".format(m, n)
#             print "combBix is ", combBix
#             print "combCix is ", combCix
#             print "combined terms A, B, C are \n" + \
#             " {0}, {1}, {2}".format(termA, allTermBs, allTermCs)
#
#         return (beta ** 4. * termA -
#                 beta ** 3. * allTermBs +
#                 beta ** 2. * allTermCs) / 4.
#
#     def __compute_Sigma4derv_matrix__(self, np.ndarray[DTYPE_t, ndim=2] x,
#                                       double pars,
#                                       np.ndarray[DTYPE_int_t, ndim=1] ix,
#                                       np.ndarray[DTYPE_t, ndim=1] metric):
#         """
#         Compute the coefficients due to the derivatives - this
#         should result in a symmetric N x N matrix where N is the
#         number of observations
#
#         :params par: theta_2^2 according to George parametrization
#         :params ix: list or array
#             of 4 integers to indicate derivative subscripts
#         """
#         # should add a check to make sure that pars is a float not a list
#
#         return [[self.__Sigma4thDeriv__(pars, x, ix, m, n, metric,
#                                         debug=False)
#                 for m in range(x.shape[0])]
#                 for n in range(x.shape[0])
#                 ]
#
#     @cython.boundscheck(False)
#     def kernel_deriv_coeff(self, np.ndarray[DTYPE_t, ndim=2] x1,
#                            np.ndarray[DTYPE_int_t, ndim=2] ix_list,
#                            double pars,
#                            np.ndarray[DTYPE_int_t, ndim=1] terms_signs,
#                            np.ndarray[DTYPE_t, ndim=1] metric):
#         """equivalent to the value method in the `KernelDerivatives` class
#         in kern_deriv.py but implementation may be a bit different
#         :param x: 2D np array of floats
#             these are the spatial coordinates / features
#         :param ix_list: list or array of lists,
#             each embedded list contains 4 integers that denotes the subscripts
#             of the terms
#         :param pars:
#         :param terms_signs:
#         """
#         cdef np.ndarray[DTYPE_t, ndim=2] mat = \
#             np.zeros((x1.shape[0], x1.shape[0]))
#
#         for i in range(len(ix_list)):
#             mat += terms_signs[i] * \
#                 self.__compute_Sigma4derv_matrix__(x1, pars, ix_list[i],
#                                                    metric)
#
#         return mat
#
#     @cython.boundscheck(False)
#     def value_symmetric(self, np.ndarray[DTYPE_t, ndim=2] x,
#                         np.ndarray[DTYPE_int_t, ndim=2] ix_list,
#                         double pars,
#                         np.ndarray[DTYPE_int_t, ndim=1] terms_signs,
#                         np.ndarray[DTYPE_t, ndim=1] metric):
#         """
#         :param x: features
#         :param pars: list of floats, pars of the kernel
#         :params ix_list: list or array of 4 lists
#             each list contains 4 integers to indicate derivative subscripts
#         """
#         cdef unsigned int n = x.shape[0], ndim = x.shape[1]
#         if self.kernel.get_ndim() != ndim:
#             raise ValueError("Dimension mismatch")
#
#         # Build the kernel matrix.
#         cdef double value
#         cdef unsigned int i, j, delta = x.strides[0]
#         cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((n, n), dtype=DTYPE)
#
#         cdef np.ndarray[DTYPE_t, ndim=2] LambDa = \
#             self.kernel_deriv_coeff(x, ix_list, pars, terms_signs, metric)
#
#         for i in range(n):
#             # Compute diagonal values.
#             # x.data gives the pointers to the numpy array values.
#             k[i, i] = self.kernel.value(<double*>(x.data + i*delta),
#                                         <double*>(x.data + i*delta)) * \
#                 LambDa[i, i]
#
#             for j in range(i + 1, n):
#                 # Off diagonal values assuming symmetric matrix.
#                 value = self.kernel.value(<double*>(x.data + i*delta),
#                                           <double*>(x.data + j*delta)) * \
#                     LambDa[i, j]
#                 k[i, j] = value
#                 k[j, i] = value
#
#         return k
#
#     @cython.boundscheck(False)
#     def value_general(self, np.ndarray[DTYPE_t, ndim=2] x1,
#                       np.ndarray[DTYPE_t, ndim=2] x2,
#                       np.ndarray[DTYPE_int_t, ndim=2] ix_list,
#                       double pars,
#                       np.ndarray[DTYPE_int_t, ndim=1] terms_signs,
#                       np.ndarray[DTYPE_t, ndim=1] metric):
#         # Parse the input kernel spec.
#         cdef unsigned int n1 = x1.shape[0], ndim = x1.shape[1], n2 = x2.shape[0]
#         if self.kernel.get_ndim() != ndim or x2.shape[1] != ndim:
#             raise ValueError("Dimension mismatch")
#
#         cdef np.ndarray[DTYPE_t, ndim=2] LambDa = \
#             self.kernel_deriv_coeff(x1, ix_list, pars, terms_signs, metric)
#
#         # Build the kernel matrix.
#         cdef double value
#         cdef unsigned int i, j, d1 = x1.strides[0], d2 = x2.strides[0]
#         cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((n1, n2), dtype=DTYPE)
#         for i in range(n1):
#             for j in range(n2):
#                 k[i, j] = self.kernel.value(<double*>(x1.data + i*d1),
#                                             <double*>(x2.data + j*d2)) * \
#                     LambDa[i, j]
#
#         return k
#
#     @cython.boundscheck(False)
#     def gradient_symmetric(self, np.ndarray[DTYPE_t, ndim=2] x):
#         # have to think about what to do with the gradient function
#         # this should also be modified
#         # Check the input dimensions.
#         cdef unsigned int n = x.shape[0], ndim = x.shape[1]
#         if self.kernel.get_ndim() != ndim:
#             raise ValueError("Dimension mismatch")
#
#         # Get the number of parameters.
#         cdef unsigned int size = self.kernel.size()
#
#         # Build the gradient matrix.
#         cdef double value
#         cdef np.ndarray[DTYPE_t, ndim=3] g = np.empty((n, n, size), dtype=DTYPE)
#         cdef unsigned int i, j, k, delta = x.strides[0]
#         cdef unsigned int dx = g.strides[0], dy = g.strides[1]
#         for i in range(n):
#             self.kernel.gradient(<double*>(x.data + i*delta),
#                                  <double*>(x.data + i*delta),
#                                  <double*>(g.data + i*dx + i*dy))
#             for j in range(i + 1, n):
#                 self.kernel.gradient(<double*>(x.data + i*delta),
#                                      <double*>(x.data + j*delta),
#                                      <double*>(g.data + i*dx + j*dy))
#                 for k in range(size):
#                     g[j, i, k] = g[i, j, k]
#
#         return g
#
