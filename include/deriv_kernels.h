#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

namespace george {
namespace kernels {



class Kernel {
public:
    Kernel (const unsigned int ndim) : ndim_(ndim) {};
    virtual ~Kernel () {};
    virtual double value (const double* x1, const double* x2) {
        // printf ("calling Kernel.value !!! WRONG INHERITANCE\n");
        return 0.0;
    };
    virtual void gradient (const double* x1, const double* x2, double* grad) {
        unsigned int i;
        for (i = 0; i < this->size(); ++i) grad[i] = 0.0;
    };

    // Input dimension.
    void set_ndim (const unsigned int ndim) { ndim_ = ndim; };
    unsigned int get_ndim () const { return ndim_; };

    // Parameter vector spec.
    virtual unsigned int size () const { return 0; }
    virtual void set_vector (const double* vector) {
        int i, n = this->size();
        for (i = 0; i < n; ++i) {
            this->set_parameter(i, vector[i]);

            // printf ("set_vector is called to set vector[%d] = %.2f\n",
            // i, vector[i]);
        }
    };
    virtual void set_parameter (const unsigned int i, const double value) {};
    virtual double get_parameter (const unsigned int i) const { return 0.0; };

protected:
    unsigned int ndim_;
};


//
// OPERATORS
//

class Operator : public Kernel {
public:
    Operator (const unsigned int ndim, Kernel* k1, Kernel* k2)
        : Kernel(ndim), kernel1_(k1), kernel2_(k2) {};
    ~Operator () {
        delete kernel1_;
        delete kernel2_;
    };
    Kernel* get_kernel1 () const { return kernel1_; };
    Kernel* get_kernel2 () const { return kernel2_; };

    // Parameter vector spec.
    unsigned int size () const { return kernel1_->size() + kernel2_->size(); };
    void set_parameter (const unsigned int i, const double value) {
        unsigned int n = kernel1_->size();
        if (i < n) kernel1_->set_parameter(i, value);
        else kernel2_->set_parameter(i-n, value);
    };
    double get_parameter (const unsigned int i) const {
        unsigned int n = kernel1_->size();
        if (i < n) return kernel1_->get_parameter(i);
        return kernel2_->get_parameter(i-n);
    };

protected:
    Kernel* kernel1_, * kernel2_;
};

class Sum : public Operator {
public:
    Sum (const unsigned int ndim, Kernel* k1, Kernel* k2)
        : Operator(ndim, k1, k2) {};

    double value (const double* x1, const double* x2) {
        return this->kernel1_->value(x1, x2) + this->kernel2_->value(x1, x2);
    };

    void gradient (const double* x1, const double* x2, double* grad) {
        unsigned int n = this->kernel1_->size();
        this->kernel1_->gradient(x1, x2, grad);
        this->kernel2_->gradient(x1, x2, &(grad[n]));
    };
};

class Product : public Operator {
public:
    Product (const unsigned int ndim, Kernel* k1, Kernel* k2)
        : Operator(ndim, k1, k2) {};

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        return this->kernel1_->value(x1, x2) * this->kernel2_->value(x1, x2);
    };

    void gradient (const double* x1, const double* x2, double* grad) {
        unsigned int i, n1 = this->kernel1_->size(), n2 = this->size();

        this->kernel1_->gradient(x1, x2, grad);
        this->kernel2_->gradient(x1, x2, &(grad[n1]));

        double k1 = this->kernel1_->value(x1, x2),
               k2 = this->kernel2_->value(x1, x2);
        for (i = 0; i < n1; ++i) grad[i] *= k2;
        for (i = n1; i < n2; ++i) grad[i] *= k1;
    };
};

//
// BASIC KERNELS
//

class ConstantKernel : public Kernel {
public:
    ConstantKernel (const unsigned int ndim) : Kernel(ndim) {};
    ConstantKernel (const unsigned int ndim, const double value)
        : Kernel(ndim), value_(value) {};

    double value (const double* x1, const double* x2) {
        return value_;
    };
    void gradient (const double* x1, const double* x2, double* grad) {
        grad[0] = 1.0;
    };

    unsigned int size () const { return 1; }
    void set_parameter (const unsigned int i, const double value) {
        value_ = value;
    };
    double get_parameter (const unsigned int i) const { return value_; };

private:
    double value_;
};

class WhiteKernel : public Kernel {
public:
    WhiteKernel (const unsigned int ndim) : Kernel(ndim) {};

    double _switch (const double* x1, const double* x2) const {
        unsigned int i, n = this->get_ndim();
        double d, r2 = 0.0;
        for (i = 0; i < n; ++i) {
            d = x1[i] - x2[i];
            r2 += d * d;
        }
        if (r2 < DBL_EPSILON) return 1.0;
        return 0.0;
    };

    double value (const double* x1, const double* x2) {
        return value_ * _switch(x1, x2);
    };

    void gradient (const double* x1, const double* x2, double* grad) {
        grad[0] = _switch(x1, x2);
    };

    unsigned int size () const { return 1; }
    void set_parameter (const unsigned int i, const double value) {
        value_ = value;
    };
    double get_parameter (const unsigned int i) const { return value_; };

private:
    double value_;
};

class DotProductKernel : public Kernel {
public:
    DotProductKernel (const unsigned int ndim) : Kernel(ndim) {};

    virtual double value (const double* x1, const double* x2) {
        unsigned int i, ndim = this->get_ndim();
        double val = 0.0;
        for (i = 0; i < ndim; ++i) val += x1[i] * x2[i];
        return val;
    };
    void gradient (const double* x1, const double* x2, double* grad) {};
    unsigned int size () const { return 0; }
};


//
// RADIAL KERNELS
//

template <typename M>
class RadialKernel : public Kernel {
public:
    RadialKernel (const long ndim, M* metric)
        : Kernel(ndim), metric_(metric) {};
    ~RadialKernel () {
        delete metric_;
    };

    // Interface to the metric.
    double get_squared_distance (const double* x1, const double* x2) const {
        return metric_->get_squared_distance(x1, x2);
    };

    double get_radial_gradient (double r2) const {
        return 0.0;
    };

    void gradient (const double* x1, const double* x2, double* grad) {
        metric_gradient(x1, x2, grad);
    };

    virtual double metric_gradient (const double* x1, const double* x2, double* grad) {
        int i, n = metric_->size();
        double r2 = metric_->gradient(x1, x2, grad),
               kg = this->get_radial_gradient(r2);
        for (i = 0; i < n; ++i) grad[i] *= kg;
        return r2;
    };


    // Parameter vector spec.
    unsigned int size () const { return metric_->size(); };
    void set_parameter (const unsigned int i, const double value) {
        metric_->set_parameter(i, value);
    };
    double get_parameter (const unsigned int i) const {
        return metric_->get_parameter(i);
    };

protected:
    M* metric_;

};

/* template <typename M>
class ExpKernel : public RadialKernel<M> {
public:
    ExpKernel (const long ndim, M* metric) : RadialKernel<M>(ndim, metric) {};
    virtual double value (const double* x1, const double* x2) {
        return exp(-sqrt(this->get_squared_distance(x1, x2)));
    };
    double get_radial_gradient (double r2) const {
        double r = sqrt(r2);
        if (r < DBL_EPSILON)
            return 0.0;
        return -0.5 * exp(-r) / r;
    };
};
*/
template <typename M>
class ExpSquaredKernel : public RadialKernel<M> {
public:
    ExpSquaredKernel (const long ndim, M* metric)
        : RadialKernel<M>(ndim, metric) {};

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        // printf("Inside ExpSquared original value method\n");
        return exp(-0.5 * this->get_squared_distance(x1, x2));
    };

    double get_radial_gradient (double r2) const {
        return -0.5 * exp(-0.5 * r2);
    };

    virtual double metric_gradient (const double* x1, const double* x2, double* grad) {
        throw "Code should not invoke ExpSquaredKernel.metric_gradient()";
    };


};

/*  template <typename M>
class Matern32Kernel : public RadialKernel<M> {
public:
    Matern32Kernel (const long ndim, M* metric)
        : RadialKernel<M>(ndim, metric) {};
    double value (const double* x1, const double* x2) {
        double r = sqrt(3 * this->get_squared_distance(x1, x2));
        return (1 + r) * exp(-r);
    };
    double get_radial_gradient (double r2) const {
        double r = sqrt(3 * r2);
        return -3 * 0.5 * exp(-r);
    };
};

template <typename M>
class Matern52Kernel : public RadialKernel<M> {
public:
    Matern52Kernel (const long ndim, M* metric)
        : RadialKernel<M>(ndim, metric) {};
    double value (const double* x1, const double* x2) {
        double r2 = 5 * this->get_squared_distance(x1, x2),
               r = sqrt(r2);
        return (1 + r + r2 / 3.0) * exp(-r);
    };
    double get_radial_gradient (double r2) const {
        double r = sqrt(5 * r2);
        return -5 * (1 + r) * exp(-r) / 6.0;
    };
};

template <typename M>
class RationalQuadraticKernel : public RadialKernel<M> {
public:
    RationalQuadraticKernel (const long ndim, M* metric)
        : RadialKernel<M>(ndim, metric) {};

    virtual double value (const double* x1, const double* x2) {
        double r2 = this->get_squared_distance(x1, x2);
        return pow(1 + 0.5 * r2 / alpha_, -alpha_);
    };
    double get_radial_gradient (double r2) const {
        return -0.5 * pow(1 + 0.5 * r2 / alpha_, -alpha_-1);
    };
    void gradient (const double* x1, const double* x2, double* grad) {
        double r2 = this->metric_gradient(x1, x2, &(grad[1])),
               t1 = 1.0 + 0.5 * r2 / alpha_,
               t2 = 2.0 * alpha_ * t1;
        grad[0] = pow(t1, -alpha_) * (r2 / t2 - log(t1));
    };
    unsigned int size () const { return this->metric_->size() + 1; };
    void set_parameter (const unsigned int i, const double value) {
        if (i == 0) alpha_ = value;
        else this->metric_->set_parameter(i - 1, value);
    };
    double get_parameter (const unsigned int i) const {
        if (i == 0) return alpha_;
        return this->metric_->get_parameter(i - 1);
    };

private:
    double alpha_;
};

*/ 

template <typename M>
class DerivativeExpSquaredKernel: public ExpSquaredKernel<M>{
public:
    DerivativeExpSquaredKernel (const long ndim, M* metric);

    using Kernel::value;
    virtual double value (const double* x1, const double* x2);
    virtual double get_radial_gradient (const double* x1, const double* x2);
    virtual double metric_gradient (const double* x1, const double* x2, double* grad);
    void print_1D_vec(const vector <double> vec1D, std::string name) const; 
    void print_1D_vec(const vector <int> vec1D, std::string name) const;
    void print_2D_vec(const vector <vector <int> > vec2D, std::string name) const;

protected:
    vector < vector <int> > pairs_of_B_ixes_;
    vector < vector <int> > pairs_of_C_ixes_;
    vector < vector <int> > comb_B_ixes_;
    vector < vector <int> > comb_C_ixes_;
    
    double X(const double* x1, const double* x2, const int spatial_ix);
    void set_termB_ixes(vector <vector <int> >& v2d);
    void set_termC_ixes(vector <vector <int> >& v2d);
    double termA(const double* x1, const double* x2, const vector<int> ix);
    double termB(const double* x1, const double* x2, const vector<int> ix); 
    double termC(const vector<int> ix);
    void set_combine_B_ixes(const vector<int>& kernel_B_ix);
    void set_combine_C_ixes(const vector<int>& kernel_C_ix);
    double Sigma4thDeriv(const vector<int>& ix, const double* x1, const double* x2);
    double compute_Sigma4deriv_matrix(const double* x1, const double* x2,
                                      const vector< vector<int> >& ix,
                                      const vector<double>& signs);
};

template <typename M>
class KappaKappaExpSquaredKernel: public DerivativeExpSquaredKernel<M>{
public:
    KappaKappaExpSquaredKernel (const long ndim, M* metric);

    using Kernel::value;
    virtual double value (const double* x1, const double* x2);
    double get_radial_gradient (const double* x1, const double* x2);

private:
    vector< vector<int> > ix_list_;
    vector<double> terms_signs_;
    void set_ix_list(vector < vector<int> >& v2d);
    void set_terms_signs(vector<double>& signs);

};

template <typename M>
class KappaGamma1ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    KappaGamma1ExpSquaredKernel (const long ndim, M* metric);

    using Kernel::value;
    virtual double value (const double* x1, const double* x2);
    double get_radial_gradient (const double* x1, const double* x2);

private:
    vector< vector<int> > ix_list_;
    vector<double> terms_signs_;
    void set_ix_list(vector< vector<int> >& v2d);
    void set_terms_signs(vector<double>& signs);
}; 

template <typename M>
class KappaGamma2ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    KappaGamma2ExpSquaredKernel (const long ndim, M* metric);

    using Kernel::value;
    virtual double value (const double* x1, const double* x2);
    double get_radial_gradient (const double* x1, const double* x2);


private:
    vector< vector<int> > ix_list_;
    vector<double> terms_signs_;
    void set_ix_list(vector< vector<int> >& v2d);
    void set_terms_signs(vector<double>& signs);
};

template <typename M>
class Gamma1Gamma1ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    Gamma1Gamma1ExpSquaredKernel (const long ndim, M* metric);

    using Kernel::value;
    virtual double value (const double* x1, const double* x2);
    double get_radial_gradient (const double* x1, const double* x2);

private:
    vector< vector<int> > ix_list_;
    vector<double> terms_signs_;
    void set_ix_list(vector< vector<int> >& v2d);
    void set_terms_signs(vector<double>& signs);
};

template <typename M>
class Gamma1Gamma2ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    Gamma1Gamma2ExpSquaredKernel (const long ndim, M* metric);

    using Kernel::value;
    virtual double value (const double* x1, const double* x2);
    double get_radial_gradient (const double* x1, const double* x2);

private:
    vector< vector<int> > ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(vector< vector<int> >& v2d);
    void set_terms_signs(vector<double>& signs);
};

template <typename M>
class Gamma2Gamma2ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    Gamma2Gamma2ExpSquaredKernel (const long ndim, M* metric);

    using Kernel::value;
    virtual double value (const double* x1, const double* x2);
    double get_radial_gradient (const double* x1, const double* x2);

private:
    vector< vector<int> > ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(vector< vector<int> >& v2d);
    void set_terms_signs(vector<double>& signs);
};


//
// 'Composite' gravitational lensing kernel
//
enum lens_field_t {kappa, gamma1, gamma2};
template <typename M>
class GravLensingExpSquaredKernel: public ExpSquaredKernel<M> {
public:
  GravLensingExpSquaredKernel (M* metric);
  ~GravLensingExpSquaredKernel ();

  using Kernel::value;
  // may not need the virtual keyword 
  virtual double value (const double* x1, const double* x2);
  double get_radial_gradient (double r2) const;  

private:
  double x_max_; // Assume the GP is defined on [0, x_max_]
  int ndim_;

  KappaKappaExpSquaredKernel<M> * kk_;
  KappaGamma1ExpSquaredKernel<M> * kg1_;
  KappaGamma2ExpSquaredKernel<M> * kg2_;
  Gamma1Gamma2ExpSquaredKernel<M> * g1g2_;
  Gamma1Gamma1ExpSquaredKernel<M> * g1g1_;
  Gamma2Gamma2ExpSquaredKernel<M> * g2g2_;
};


class TwoDdynamicArray{
public:
    TwoDdynamicArray(const int& nrow, const int& ncol);
    ~TwoDdynamicArray();
    void create_from_2D_arr(const double pt[][2], const int& nobs);
    double** val;
    const int nrow;
    const int ncol;
};

}; // namespace kernels
}; // namespace george


#endif