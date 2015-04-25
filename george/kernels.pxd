# distutils: language = c++
from __future__ import division

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef extern from "metrics.h" namespace "george::metrics":

    cdef cppclass Metric:
        pass

    cdef cppclass OneDMetric(Metric):
        OneDMetric(const unsigned int ndim, const unsigned int dim)

    cdef cppclass IsotropicMetric(Metric):
        IsotropicMetric(const unsigned int ndim)

    cdef cppclass AxisAlignedMetric(Metric):
        AxisAlignedMetric(const unsigned int ndim)

cdef extern from "kernels.h" namespace "george::kernels":

    cdef cppclass Kernel:
        double value (const double* x1, const double* x2) const
        void gradient (const double* x1, const double* x2, double* grad) const
        unsigned int get_ndim () const
        unsigned int size () const
        void set_vector (const double*)


    cdef cppclass CustomKernel(Kernel):
        CustomKernel(const unsigned int ndim, const unsigned int size,
                     void* meta,
                     double (*f) (const double* pars, const unsigned int size,
                                  void* meta,
                                  const double* x1, const double* x2,
                                  const unsigned int ndim),
                     void (*g) (const double* pars, const unsigned int size,
                                void* meta,
                                const double* x1, const double* x2,
                                const unsigned int ndim, double* grad))

    # Operators.
    cdef cppclass Operator(Kernel):
        pass

    cdef cppclass Sum(Operator):
        Sum(const unsigned int ndim, Kernel* k1, Kernel* k2)

    cdef cppclass Product(Operator):
        Product(const unsigned int ndim, Kernel* k1, Kernel* k2)

    # Basic kernels.
    cdef cppclass ConstantKernel(Kernel):
        ConstantKernel(const unsigned int ndim)

    cdef cppclass WhiteKernel(Kernel):
        WhiteKernel(const unsigned int ndim)

    cdef cppclass DotProductKernel(Kernel):
        DotProductKernel(const unsigned int ndim)

    # Radial kernels.
    cdef cppclass ExpKernel[M](Kernel):
        ExpKernel(const unsigned int ndim, M* metric)

    cdef cppclass ExpSquaredKernel[M](Kernel):
        ExpSquaredKernel(const unsigned int ndim, M* metric)

    cdef cppclass Matern32Kernel[M](Kernel):
        Matern32Kernel(const unsigned int ndim, M* metric)

    cdef cppclass Matern52Kernel[M](Kernel):
        Matern52Kernel(const unsigned int ndim, M* metric)

    cdef cppclass RationalQuadraticKernel[M](Kernel):
        RationalQuadraticKernel(const unsigned int ndim, M* metric)

    # Periodic kernels.
    cdef cppclass CosineKernel(Kernel):
        CosineKernel(const unsigned int ndim, const unsigned int dim)

    cdef cppclass ExpSine2Kernel(Kernel):
        ExpSine2Kernel(const unsigned int ndim, const unsigned int dim)

    # Custom Cpp kernels for derivatives.
    cdef cppclass DerivativeExpSquaredKernel[M](ExpSquaredKernel):
        DerivativeExpSquaredKernel(const unsigned int ndim, M* metric)

        # cdef np.ndarray __pairsOfBIndices__
        # cdef np.ndarray __pairsOfCIndices__

        # @cython.boundscheck(False)
        # def __cinit__(self):
        #     self.__pairsOfBIndices__ = \
        #         np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2],
        #                 [2, 3, 0, 1], [1, 3, 0, 2], [1, 2, 0, 3]],
        #                 dtype=DTYPE_int)

        #     self.__pairsOfCIndices__ = \
        #         np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]],
        #                 dtype=DTYPE_int)

        # def __X__(self, np.ndarray[DTYPE_t, ndim=2] coords,
        #         unsigned int m, unsigned int n, unsigned int spat_ix):
        #     return coords[m, spat_ix] - coords[n, spat_ix]

        # def __termA__(self, np.ndarray[DTYPE_t, ndim=2] coords,
        #             np.ndarray[DTYPE_int_t, ndim=1] ix,
        #             unsigned int m, unsigned int n,
        #             bool_t debug=False):
        #     """
        #     # the constructor also needs the coordinates
        #     Compute term 1 in equation (24) without leading factors of $\beta^4$

        #     :params coords: numpy array,
        #         shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions

        #     :params ix: np array of 4 integers,
        #         denote the spatial subscripts for the ker deriv

        #     :beta: float,
        #         inverse length

        #     .. math:
        #         X_i X_j X_h X_k
        #     """
        #     cdef double term = 1.

        #     if debug:
        #         print "---------------------------------"
        #     for i in ix:
        #         if debug:
        #             print "parts of term A = {0}".format(term)
        #         term *= self.__X__(coords, m, n, i)

        #     if debug:
        #         print "m = {0}, n = {1}".format(m, n)
        #         print "indices of term A ix = {0}".format(ix)
        #         print "term A = {0}".format(term)
        #         print "---------------------------------"

        #     return term

        # def __termB__(self, np.ndarray[DTYPE_t, ndim=2] coords,
        #             np.ndarray[DTYPE_int_t, ndim=1] ix,
        #             unsigned int m, unsigned int n,
        #             np.ndarray[DTYPE_t, ndim=1] metric, debug=False):
        #     """
        #     Compute term 2 in equation (24) without leading factors of $\beta^3$

        #     :param coords: numpy array,
        #         shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions

        #     :param ix: list of 4 integers,
        #         denote the spatial subscripts for the ker deriv,
        #         assumes to take the form [a, b, c, d]

        #     :param m: integer
        #         denote the row index of covariance matrix, or obs_no

        #     :params n: integer
        #         denote the col index of covariance matrix, or obs_no

        #     :param metric: list of floats
        #         should be of dimension 2, we assume diagonal metric here

        #     .. math:
        #         X_a X_b D_{cd} \delta_{cd}
        #     """
        #     if debug is True:
        #         print "---------------------------------"
        #         print "m = {0}, n = {1}".format(m, n)
        #         print "indices of term B = {0}".format(ix)
        #         print "termB = {0}".format(self.__X__(coords, m, n, ix[0]) *
        #         self.__X__(coords, m, n, ix[1]) *
        #         metric[ix[2]])

        #     if ix[2] != ix[3]:
        #         return 0

        #     return self.__X__(coords, m, n, ix[0]) * \
        #         self.__X__(coords, m, n, ix[1]) * \
        #         metric[ix[2]]


        # def __termC__(self, np.ndarray[DTYPE_t, ndim=2] coords,
        #               np.ndarray[DTYPE_int_t, ndim=1] ix,
        #               np.ndarray[DTYPE_t, ndim=1] metric,
        #               bool_t debug=False):
        #     """
        #     Compute term 3 in equation (24) without leading factor of $\beta^2$

        #     :param coords: numpy array,
        #         shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions

        #     :param ix: list of 4 integers,
        #         denote the spatial subscripts for the ker deriv,
        #         assumes to take the form [a, b, c, d]

        #     :param metric: array of floats

        #     .. math:
        #         D_{ab} D_{cd} \delta_{ab} \delta_{cd}
        #     """
        #     if debug:
        #         print "---------------------------------"
        #         print "indices of term C = {0}".format(ix)
        #         print "term C = {0}".format(metric[ix[2]] * metric[ix[0]])

        #     if ix[0] != ix[1]:
        #         return 0

        #     if ix[2] != ix[3]:
        #         return 0

        #     return metric[ix[2]] * metric[ix[0]]

        # def __Sigma4thDeriv__(self, double beta,
        #                     np.ndarray[DTYPE_t, ndim=2] coords,
        #                     np.ndarray[DTYPE_int_t, ndim=1] ix,
        #                     unsigned int m, unsigned int n,
        #                     np.ndarray[DTYPE_t, ndim=1] metric,
        #                     bool_t debug=False):
        #     """
        #     Gather the 10 terms for the 4th derivative of each Sigma
        #     given the ix for each the derivatives are taken w.r.t.

        #     :params corr: float
        #         value of the correlation parameter in the ExpSquaredKernel

        #     :params coords: 2D numpy array
        #         with shape (nObs, ndim)

        #     :params ix: list of 4 integers
        #     """
        #     allTermBs = 0
        #     combBix = \
        #         np.array([[ix[i] for i in self.__pairsOfBIndices__[j]]
        #                 for j in range(6)])

        #     # combBix is the subscript indices combination for B terms
        #     for i in range(6):
        #         allTermBs += self.__termB__(coords, combBix[i], m, n, metric)

        #     allTermCs = 0
        #     combCix = \
        #         np.array([[ix[i] for i in self.__pairsOfCIndices__[j]]
        #                 for j in range(3)])

        #     for i in range(3):
        #         allTermCs += self.__termC__(coords, combCix[i], metric)

        #     termA = self.__termA__(coords, ix, m, n)

        #     if debug:
        #         print "---------------------------------"
        #         print "m = {0}, n = {1}".format(m, n)
        #         print "combBix is ", combBix
        #         print "combCix is ", combCix
        #         print "combined terms A, B, C are \n" + \
        #         " {0}, {1}, {2}".format(termA, allTermBs, allTermCs)

        #     return (beta ** 4. * termA -
        #             beta ** 3. * allTermBs +
        #             beta ** 2. * allTermCs) / 4.

        # def __compute_Sigma4derv_matrix__(self, np.ndarray[DTYPE_t, ndim=2] x,
        #                                 double pars,
        #                                 np.ndarray[DTYPE_int_t, ndim=1] ix,
        #                                 np.ndarray[DTYPE_t, ndim=1] metric):
        #     """
        #     Compute the coefficients due to the derivatives - this
        #     should result in a symmetric N x N matrix where N is the
        #     number of observations

        #     :params par: theta_2^2 according to George parametrization
        #     :params ix: list or array
        #         of 4 integers to indicate derivative subscripts
        #     """
        #     # should add a check to make sure that pars is a float not a list

        #     return [[self.__Sigma4thDeriv__(pars, x, ix, m, n, metric,
        #                                     debug=False)
        #             for m in range(x.shape[0])]
        #             for n in range(x.shape[0])
        #             ]

    cdef cppclass KappaKappaExpSquaredKernel[M](ExpSquaredKernel):
        KappaKappaExpSquaredKernel(const unsigned int ndim, M* metric)

    cdef cppclass KappaGamma1ExpSquaredKernel[M](ExpSquaredKernel):
        KappaGamma1ExpSquaredKernel(const unsigned int ndim, M* metric)

    cdef cppclass KappaGamma2ExpSquaredKernel[M](ExpSquaredKernel):
        KappaGamma2ExpSquaredKernel(const unsigned int ndim, M* metric)

    cdef cppclass Gamma1Gamma1ExpSquaredKernel[M](ExpSquaredKernel):
        Gamma1Gamma1ExpSquaredKernel(const unsigned int ndim, M* metric)

    cdef cppclass Gamma1Gamma2ExpSquaredKernel[M](ExpSquaredKernel):
        Gamma1Gamma2ExpSquaredKernel(const unsigned int ndim, M* metric)

    cdef cppclass Gamma2Gamma2ExpSquaredKernel[M](ExpSquaredKernel):
        Gamma2Gamma2ExpSquaredKernel(const unsigned int ndim, M* metric)



cdef inline double eval_python_kernel (const double* pars,
                                       const unsigned int size, void* meta,
                                       const double* x1, const double* x2,
                                       const unsigned int ndim) except *:
    # Build the arguments for calling the function.
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp>ndim
    x1_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x1)
    x2_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x2)

    shape[0] = <np.npy_intp>size
    pars_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>pars)

    # Call the Python function and return the value.
    cdef object self = <object>meta
    return self.f(x1_arr, x2_arr, pars_arr)


cdef inline void eval_python_kernel_grad (const double* pars,
                                          const unsigned int size,
                                          void* meta,
                                          const double* x1, const double* x2,
                                          const unsigned int ndim, double* grad) except *:
    # Build the arguments for calling the function.
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp>ndim
    x1_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x1)
    x2_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x2)

    shape[0] = <np.npy_intp>size
    pars_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>pars)

    # Call the Python function and update the gradient values in place.
    cdef object self = <object>meta
    cdef np.ndarray[DTYPE_t, ndim=1] grad_arr = \
        np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>grad)
    grad_arr[:] = self.g(x1_arr, x2_arr, pars_arr)


cdef inline Kernel* parse_kernel(kernel_spec) except *:
    if not hasattr(kernel_spec, "is_kernel"):
        raise TypeError("Invalid kernel")

    # Deal with operators first.
    cdef Kernel* k1
    cdef Kernel* k2
    cdef unsigned int n1, n2
    if not kernel_spec.is_kernel:
        k1 = parse_kernel(kernel_spec.k1)
        n1 = k1.get_ndim()
        k2 = parse_kernel(kernel_spec.k2)
        n2 = k2.get_ndim()
        if n1 != n2:
            raise ValueError("Dimension mismatch")

        if kernel_spec.operator_type == 0:
            return new Sum(n1, k1, k2)
        elif kernel_spec.operator_type == 1:
            return new Product(n1, k1, k2)
        else:
            raise TypeError("Unknown operator: {0}".format(
                kernel_spec.__class__.__name__))

    # Get the kernel parameters.
    cdef unsigned int ndim = kernel_spec.ndim
    cdef np.ndarray[DTYPE_t, ndim=1] pars = kernel_spec.pars

    cdef Kernel* kernel

    if kernel_spec.kernel_type == -2:
        kernel = new CustomKernel(ndim, kernel_spec.size, <void*>kernel_spec,
                                  &eval_python_kernel, &eval_python_kernel_grad)

    elif kernel_spec.kernel_type == 0:
        kernel = new ConstantKernel(ndim)

    elif kernel_spec.kernel_type == 1:
        kernel = new WhiteKernel(ndim)

    elif kernel_spec.kernel_type == 2:
        kernel = new DotProductKernel(ndim)

    elif kernel_spec.kernel_type == 3:
        if kernel_spec.dim >= 0:
            kernel = new ExpKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new ExpKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new ExpKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 4:
        if kernel_spec.dim >= 0:
            kernel = new ExpSquaredKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new ExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new ExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 5:
        if kernel_spec.dim >= 0:
            kernel = new Matern32Kernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new Matern32Kernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new Matern32Kernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 6:
        if kernel_spec.dim >= 0:
            kernel = new Matern52Kernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new Matern52Kernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new Matern52Kernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 7:
        if kernel_spec.dim >= 0:
            kernel = new RationalQuadraticKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new RationalQuadraticKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new RationalQuadraticKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 8:
        kernel = new CosineKernel(ndim, kernel_spec.dim)

    elif kernel_spec.kernel_type == 9:
        kernel = new ExpSine2Kernel(ndim, kernel_spec.dim)

    elif kernel_spec.kernel_type == 10:
        # the constructor also needs the coordinates
        if kernel_spec.dim >= 0:
            kernel = new KappaKappaExpSquaredKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new KappaKappaExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new KappaKappaExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 11:
        # the constructor also needs the coordinates
        if kernel_spec.dim >= 0:
            kernel = new KappaGamma1ExpSquaredKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new KappaGamma1ExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new KappaGamma1ExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 12:
        # the constructor also needs the coordinates
        if kernel_spec.dim >= 0:
            kernel = new KappaGamma2ExpSquaredKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new KappaGamma2ExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new KappaGamma2ExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 13:
        # the constructor also needs the coordinates
        if kernel_spec.dim >= 0:
            kernel = new Gamma1Gamma1ExpSquaredKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new Gamma1Gamma1ExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new Gamma1Gamma1ExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 14:
        # the constructor also needs the coordinates
        if kernel_spec.dim >= 0:
            kernel = new Gamma1Gamma2ExpSquaredKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new Gamma1Gamma2ExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new Gamma1Gamma2ExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 15:
        # the constructor also needs the coordinates
        if kernel_spec.dim >= 0:
            kernel = new Gamma2Gamma2ExpSquaredKernel[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new Gamma2Gamma2ExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new Gamma2Gamma2ExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")
    else:
        raise TypeError("Unknown kernel: {0}".format(
            kernel_spec.__class__.__name__))

    kernel.set_vector(<double*>pars.data)
    return kernel
