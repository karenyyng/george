#ifndef _GEORGE_METRICS_H_
#define _GEORGE_METRICS_H_

#include <cmath>
#include <vector>
#include <iostream>

using std::vector;

namespace george {
namespace metrics {

class Metric {
public:
    Metric (const unsigned int ndim, const unsigned int size)
        : ndim_(ndim), vector_(size, 1) {};
    // Copy constructor
    Metric ( const Metric& aCopy ) { Copy( aCopy ); };
    virtual ~Metric () {};
    virtual double get_squared_distance (const double* x1, const double* x2) const {
        return 0.0;
    };
    virtual double gradient (const double* x1, const double* x2, double* grad) const {
        return 0.0;
    };

    // Parameter vector spec.
    virtual unsigned int size () const { return vector_.size(); };
    void set_parameter (const unsigned int i, const double value) {
        vector_.at(i) = value;
    };
    double get_parameter (const unsigned int i) const {
        return vector_.at(i);
    };

    void Copy( const Metric& aCopy ) {
      ndim_ = aCopy.ndim_;
      vector_ = aCopy.vector_;
    };

    Metric operator=( const Metric& aCopy ) {
      Copy( aCopy );
      return *this;
    };

protected:
    unsigned int ndim_;
    vector<double> vector_;
};

class OneDMetric : public Metric {
public:

    OneDMetric (const unsigned int ndim, const unsigned int dim)
        : Metric(ndim, 1), dim_(dim) {};

    double get_squared_distance (const double* x1, const double* x2) const {
        double d = x1[dim_] - x2[dim_];
        return d * d / this->vector_[0];
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        double r2 = get_squared_distance(x1, x2);
        grad[0] = -r2 / this->vector_[0];
        return r2;
    };

private:
    unsigned int dim_;
};

class IsotropicMetric : public Metric {
public:

    IsotropicMetric (const unsigned int ndim) : Metric(ndim, 1) {};

    double get_squared_distance (const double* x1, const double* x2) const {
        unsigned int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            r2 += d*d;
        }
        return r2 / this->vector_[0];
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        double r2 = get_squared_distance(x1, x2);
        grad[0] = -r2 / this->vector_[0];
        return r2;
    };
};

class AxisAlignedMetric : public Metric {
public:

    AxisAlignedMetric (const unsigned int ndim) : Metric(ndim, ndim) {};

    double get_squared_distance (const double* x1, const double* x2) const {
        // Multidimensional numpy arrays are 1D C / C++ arrays contiguous in
        // memory, along each row (obs).
        // Both Python and C++ are row-major languages  
        // data matrix: np.array([[1., 2.,], [4., 7.]) corresponds to a 
        // C++ array of {1., 2.} or {4., 7.} 
        unsigned int i;
        double d, r2 = 0.0;
        // std::cout << "get_squared_distance" << std::endl;
        for (i = 0; i < ndim_; ++i) {
            // std::cout << "x1[" << i << "] = " << x1[i] << std::endl;
            // std::cout << "x2[" << i << "] = " << x2[i] << std::endl;
            d = x1[i] - x2[i];
            r2 += d * d / this->vector_[i];
        }
        return r2;
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        unsigned int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            d = d * d / this->vector_[i];
            r2 += d;
            grad[i] = -d / this->vector_[i];
        }
        return r2;
    };
};

}; // namespace metrics
}; // namespace george

#endif
