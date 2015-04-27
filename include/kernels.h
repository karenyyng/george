#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <iostream>

using std::vector;

namespace george {
namespace kernels {



class Kernel {
public:
    Kernel (const unsigned int ndim) : ndim_(ndim) {};
    virtual ~Kernel () {};
    virtual double value (const double* x1, const double* x2) {
        printf ("calling Kernel.value !!! WRONG INHERITANCE\n");
        return 0.0;
    };
    virtual void gradient (const double* x1, const double* x2, double* grad) const {
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

            printf ("set_vector is called to set vector[%d] = %.2f\n", 
                i, vector[i]);
        }
    };
    virtual void set_parameter (const unsigned int i, const double value) {};
    virtual double get_parameter (const unsigned int i) const { return 0.0; };

protected:
    unsigned int ndim_;
};


class CustomKernel : public Kernel {
public:
    CustomKernel (const unsigned int ndim, const unsigned int size, void* meta,
                  double (*f) (const double* pars, const unsigned int size,
                               void* meta,
                               const double* x1, const double* x2,
                               const unsigned int ndim),
                  void (*g) (const double* pars, const unsigned int size,
                             void* meta,
                             const double* x1, const double* x2,
                             const unsigned int ndim, double* grad))
        : Kernel(ndim), size_(size), meta_(meta), f_(f), g_(g)
    {
        parameters_ = new double[size];
    };
    ~CustomKernel () {
        delete parameters_;
    };

    // Call the external functions.
    virtual double value (const double* x1, const double* x2) {
        return f_(parameters_, size_, meta_, x1, x2, this->get_ndim());
    };
    void gradient (const double* x1, const double* x2, double* grad) const {
        g_(parameters_, size_, meta_, x1, x2, this->get_ndim(), grad);
    };

    // Parameters.
    unsigned int size () const { return size_; }
    void set_parameter (const unsigned int i, const double value) {
        parameters_[i] = value;
    };
    double get_parameter (const unsigned int i) const {
        return parameters_[i];
    };

protected:
    double* parameters_;
    unsigned int ndim_, size_;

    // Metadata needed for this function.
    void* meta_;

    // The function and gradient pointers.
    double (*f_) (const double* pars, const unsigned int size, void* meta,
                  const double* x1, const double* x2,
                  const unsigned int ndim);
    void (*g_) (const double* pars, const unsigned int size, void* meta,
                const double* x1, const double* x2,
                const unsigned int ndim, double* grad);
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

    void gradient (const double* x1, const double* x2, double* grad) const {
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

    void gradient (const double* x1, const double* x2, double* grad) const {
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
    void gradient (const double* x1, const double* x2, double* grad) const {
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

    void gradient (const double* x1, const double* x2, double* grad) const {
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
    void gradient (const double* x1, const double* x2, double* grad) const {};
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
    virtual double get_radial_gradient (double r2) const {
        return 0.0;
    };

    virtual void gradient (const double* x1, const double* x2, double* grad) const {
        metric_gradient(x1, x2, grad);
    };

    double metric_gradient (const double* x1, const double* x2, double* grad) const {
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

template <typename M>
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

template <typename M>
class ExpSquaredKernel : public RadialKernel<M> {
public:
    ExpSquaredKernel (const long ndim, M* metric)
        : RadialKernel<M>(ndim, metric) {};

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        printf("Inside ExpSquared original value method\n");
        return exp(-0.5 * this->get_squared_distance(x1, x2));
    };
    double get_radial_gradient (double r2) const {
        return -0.5 * exp(-0.5 * r2);
    };
};

template <typename M>
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
    void gradient (const double* x1, const double* x2, double* grad) const {
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


// ideally this class should be added in a different .h file 
template <typename M>
class DerivativeExpSquaredKernel: public ExpSquaredKernel<M>{
public: 
    // have to figure out if this constructor is correct 
    // x1 is supposed to be the coordinates 
    DerivativeExpSquaredKernel (const long ndim, M* metric)
      : ExpSquaredKernel<M>(ndim, metric){}; 

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        printf ("inside DerivativeExpSquaredKernel which shouldn't be called");
        return 0.;
    }

                            
protected:
    static const int ix_rows = 6, ix_cols = 4;
    double X(const double* x1, const double* x2, 
            const int spatial_ix){ 
        printf ("Inside X: \n");
        printf ("x1[spatial_ix] = %2f\n", x1[spatial_ix]);
        printf ("x2[spatial_ix] = %2f\n", x2[spatial_ix]);
        printf ("this->metric_->get_parameter(spatial_ix) = %2f\n", 
                this->metric_->get_parameter(spatial_ix));
        printf("spatial_ix = %d\n", spatial_ix);
        return (x1[spatial_ix] - x2[spatial_ix]) * 
            this->metric_->get_parameter(spatial_ix);
    }

    vector< vector<int> > get_termB_ixes(){
        // I am not sure if it is better to create a vector 
        // or a 2D dynamic array in terms of memory usage.
        // Seems like either way, this method will be called N^2 / 2. times.
        unsigned int r, c;
        vector<vector<int> > v2d;
        vector<int> rowvector;
        const int rows = 6, cols = 4;

        int arr[rows][cols] = {{0, 1, 2, 3}, 
                               {0, 2, 1, 3},
                               {0, 3, 1, 2},
                               {2, 3, 0, 1},
                               {1, 3, 0, 2},
                               {1, 2, 0, 3}};

        for (r = 0; r < rows; r++) {
            rowvector.clear();
            for (c = 0; c < cols; c++ ) { rowvector.push_back(arr[r][c]); }
            v2d.push_back(rowvector);
        }

        return v2d;
    }

    vector< vector<int> > get_termC_ixes(){
        unsigned int r, c;
        vector<vector<int> > v2d;
        vector<int> rowvector;
        const int rows = 3, cols = 4;

        int arr[rows][cols] = {{0, 1, 2, 3}, 
                               {0, 2, 1, 3},
                               {0, 3, 1, 2}};

        for (r = 0; r < rows; r++) {
            rowvector.clear();
            for (c = 0; c < cols; c++ ) { rowvector.push_back(arr[r][c]); }
            v2d.push_back(rowvector);
        }

        return v2d;
    }

    double termA(const double* x1, const double* x2, vector<int> ix) {
        double term = 1.;
        for (vector<int>::iterator it=ix.begin(); it != ix.end(); ++it){ 
            term *= this->X(x1, x2, *it);
        }
        printf ("termA = %.2f\n", term);
        return term;
    }

    double termB(const double* x1, const double* x2, 
                 const vector<int> ix) {
        if (ix[2] != ix[3]) { return 0; }

        printf ("termB: ix[1] = %d\n", ix[1]);
        printf ("termB: ix[0] = %d\n", ix[0]);
        printf ("termB = %.2f\n", this->X(x1, x2, ix[0]) * 
                this->X(x1, x2, ix[1] * this->get_parameter(ix[2])));
        return this->X(x1, x2, ix[0]) * this->X(x1, x2, ix[1]) * 
            this->metric_->get_parameter(ix[2]);
    }

    double termC(const vector<int> ix) {
        if (ix[0] != ix[1]) { return 0; } 
        if (ix[2] != ix[3]) { return 0; }
        printf ("termC: ix[2] = %2d\n", ix[2]);
        printf ("termC: this->metric_->get_parameter(ix[2]) = %.2f\n", 
                this->metric_->get_parameter(ix[2]));
        printf ("termC: ix[0] = %2d\n", ix[0]);
        printf ("termC: this->metric_->get_parameter(ix[0]) = %.2f\n", 
                this->metric_->get_parameter(ix[0]));
        return this->metric_->get_parameter(ix[2]) * 
            this->metric_->get_parameter(ix[0]);
    }

    vector< vector<int> > combine_B_ixes(const vector<int> kernel_B_ix){
        // @param B_ix contain the kernel indices, a list of 4 integers 
        vector< vector<int> > ix = this->get_termB_ixes();
        unsigned int rows = ix.size(), cols = ix[0].size();
        unsigned int r, c;
        vector< vector<int> > termB_ixes;
        vector<int> temp_row;

        for (r = 0; r < rows; r++){
            temp_row.clear();
            for (c = 0; c < cols; c++){
                temp_row.push_back(kernel_B_ix[ix[r][c]]);
            }
            termB_ixes.push_back(temp_row);
        }
        return termB_ixes;
    }

    vector< vector<int> > combine_C_ixes(const vector<int> kernel_C_ix){
        // @param C_ix contain the kernel indices 
        vector< vector<int> > ix = this->get_termC_ixes();
        unsigned int rows = ix.size(), cols = ix[0].size();
        unsigned int r, c;
        vector< vector<int> > termC_ixes;
        vector<int> temp_row;

        for (r = 0; r < rows; r++){
            temp_row.clear();
            for (c = 0; c < cols; c++){
                temp_row.push_back(kernel_C_ix[ix[r][c]]);
            }
            termC_ixes.push_back(temp_row);
        }
        return termC_ixes;
    }

    double Sigma4thDeriv(const vector<int> ix, const double* x1, 
            const double* x2){
        // if we do decide to separate beta from the metric 
        double allTermBs = 0.;
        double allTermCs = 0.;

        vector< vector<int> > combine_B_ixes = 
            this->combine_B_ixes(ix);

        vector< vector<int> > combine_C_ixes = 
            this->combine_C_ixes(ix); 

        for (vector< vector<int> >::iterator row_it = combine_B_ixes.begin();
           row_it < combine_B_ixes.end(); ++row_it ){
           allTermBs += termB(x1, x2, *row_it); 
        }
        
        for (vector< vector<int> >::iterator row_it = combine_C_ixes.begin();
           row_it < combine_C_ixes.end(); ++row_it ){
           allTermCs += termC(*row_it); 
        }

        double termA = this->termA(x1, x2, ix);

        printf ("combined terms in Sigma4thDeriv = %.2f \n", 
                (termA - allTermBs + allTermCs) / 4.);
        // beta = metric currently and are multipled within the functions
        // for getting each term  so we don't have to multiply beta again.
        return (termA - allTermBs + allTermCs) / 4.;
    }


    double compute_Sigma4deriv_matrix(const double* x1,const double* x2, 
                                      const vector< vector<int> > ix, 
                                      const vector<double> signs){
        double term = 0;
        unsigned int r;
        int rows = ix.size();  // this should be 4 

        for (r = 0; r < rows; r++){
            term += signs[r] * this->Sigma4thDeriv(ix[r], x1, x2);
        }
        return term;
    } 
};

template <typename M>
class KappaKappaExpSquaredKernel: public DerivativeExpSquaredKernel<M>{
public: 
    KappaKappaExpSquaredKernel (const long ndim, M* metric)
      : DerivativeExpSquaredKernel<M>(ndim, metric){
          printf("constructing KappaKappaExpSquaredKernel\n");
      }; 

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        printf("Inside KappaKappa modified value method\n");
        const vector< vector<int> > ix_list = this->ix_list();

        vector<double> signs = this->terms_signs();
        return exp(-0.5 * this->get_squared_distance(x1, x2)) 
            * this->compute_Sigma4deriv_matrix(x1, x2, ix_list, signs); 
    };

private:
    vector< vector<int> > ix_list(){
        vector< vector<int> > v2d;
        vector<int> rowvec; 
        unsigned int r = 0, c = 0;
        const int rows = 4, cols = 4;
        int arr[rows][cols] = {{0, 0, 0, 0},
                               {0, 0, 1, 1},
                               {1, 1, 0, 0},
                               {1, 1, 1, 1}};

        for (r = 0; r < rows; r++){
            rowvec.clear();
            for (c = 0; c < cols; c++){ rowvec.push_back(arr[r][c]); }
            v2d.push_back(rowvec);
        }
        return v2d;
    }

    vector<double> terms_signs(){
        const double arr[4] = {1., 1., 1., 1.};
        vector<double> signs (arr, arr + sizeof(arr) / sizeof(int));
        return signs;
    }
};

template <typename M>
class KappaGamma1ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public: 
    KappaGamma1ExpSquaredKernel (const long ndim, M* metric)
      : DerivativeExpSquaredKernel<M>(ndim, metric){}; 
}; 

template <typename M>
class KappaGamma2ExpSquaredKernel : public ExpSquaredKernel<M>{
public: 
    KappaGamma2ExpSquaredKernel (const long ndim, M* metric)
      : ExpSquaredKernel<M>(ndim, metric){}; 
}; 

template <typename M>
class Gamma1Gamma1ExpSquaredKernel : public ExpSquaredKernel<M>{
public: 
    // have to figure out if this constructor is correct 
    // x1 is supposed to be the coordinates 
    Gamma1Gamma1ExpSquaredKernel (const long ndim, M* metric)
      : ExpSquaredKernel<M>(ndim, metric){}; 
}; 

template <typename M>
class Gamma1Gamma2ExpSquaredKernel : public ExpSquaredKernel<M>{
public: 
    // x1 is supposed to be the coordinates 
    Gamma1Gamma2ExpSquaredKernel (const long ndim, M* metric)
      : ExpSquaredKernel<M>(ndim, metric){}; 
}; 

template <typename M>
// ideally this class should be added in a different .h file 
class Gamma2Gamma2ExpSquaredKernel : public ExpSquaredKernel<M>{
public: 
    // have to figure out if this constructor is correct 
    // x1 is supposed to be the coordinates 
    Gamma2Gamma2ExpSquaredKernel (const long ndim, M* metric)
      : ExpSquaredKernel<M>(ndim, metric){}; 
}; 

//
// PERIODIC KERNELS
//

class CosineKernel : public Kernel {
public:
    CosineKernel (const unsigned int ndim, const unsigned int dim)
        : Kernel(ndim), dim_(dim) {};

    double value (const double* x1, const double* x2) {
        return cos(2 * M_PI * (x1[dim_] - x2[dim_]) / period_);
    };
    void gradient (const double* x1, const double* x2, double* grad) const {
        double dx = 2 * M_PI * (x1[dim_] - x2[dim_]) / period_;
        grad[0] = dx * sin(dx) / period_;
    };

    unsigned int size () const { return 1; }
    void set_parameter (const unsigned int i, const double value) {
        period_ = value;
    };
    double get_parameter (const unsigned int i) const { return period_; };

private:
    unsigned int dim_;
    double period_;
};

class ExpSine2Kernel : public Kernel {
public:
    ExpSine2Kernel (const unsigned int ndim, const unsigned int dim)
        : Kernel(ndim), dim_(dim) {};

    double value (const double* x1, const double* x2) {
        double s = sin(M_PI * (x1[dim_] - x2[dim_]) / period_);
        return exp(-gamma_ * s * s);
    };
    void gradient (const double* x1, const double* x2, double* grad) const {
        double arg = M_PI * (x1[dim_] - x2[dim_]) / period_,
               s = sin(arg), c = cos(arg), s2 = s * s, A = exp(-gamma_ * s2);

        grad[0] = -s2 * A;
        grad[1] = 2 * gamma_ * arg * c * s * A / period_;
    };

    unsigned int size () const { return 2; }
    void set_parameter (const unsigned int i, const double value) {
        if (i == 0) gamma_ = value;
        else period_ = value;
    };
    double get_parameter (const unsigned int i) const {
        if (i == 0) return gamma_;
        return period_;
    }

private:
    unsigned int dim_;
    double gamma_, period_;
};


}; // namespace kernels
}; // namespace george

#endif
