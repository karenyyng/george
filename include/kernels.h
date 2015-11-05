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

template <typename T>
class TwoDimensionalDynamicArray{
/* This is meant to be a bare metal wrapper to a dynamic array.
* No error handling is intended. 
* Construction of an N-dimensional array is similar.
*/
public:
    T** val;
    int row_no;
    int col_no;

    TwoDimensionalDynamicArray () {};

    // Ideally memory is allocated during construction.
    // But that syntax is forbidden in the header for initializing
    // the array in the DerivativeExpSquaredKernel constructor.
    void allocate_memory(const int& row_no, const int& col_no) { 
        this->row_no = row_no; 
        this->col_no = col_no;
        unsigned int r;

        // std::cout << "row_no = " << this->row_no << ", col_no = " << this->col_no << std::endl;

        // Allocate memory for array of row arrays of pointers.
        // Each pointer here points to the beginning of a row.
        this->val = new T* [row_no];

        // Allocate memory within a row. 
        // There needs to be the same no. of `new` as `delete`.
        for (r=0; r < row_no; r++) this->val[r] = new T[col_no];  
    }

    void print() const{
        unsigned int r, c;
        if (this->row_no > 0 && this->col_no > 0){

            for (r=0; r < this->row_no; r++){
                for (c=0; r < this->col_no; c++){ 
                    std:: cout << this->val[r][c] << " "; }
                cout << "\n";
            } 
            cout << "\n";
        } 
    }

    // Destructor for properly freeing memory.
    ~TwoDimensionalDynamicArray(){
        unsigned int r;
        for (r = 0; r < this->row_no; r++) delete[] val[r];
        delete[] val;
    }

};

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
    void gradient (const double* x1, const double* x2, double* grad) {
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

template <typename M>
class DerivativeExpSquaredKernel: public ExpSquaredKernel<M>{
public:
    DerivativeExpSquaredKernel (const long ndim, M* metric):
        ExpSquaredKernel<M>(ndim, metric){
             this->comb_B_ixes_.allocate_memory(24, 4);
             this->comb_C_ixes_.allocate_memory(12, 4);
             this->pairs_of_B_ixes_.allocate_memory(6, 4);
             this->pairs_of_C_ixes_.allocate_memory(3, 4);

             this->set_termB_ixes(pairs_of_B_ixes_.val);
             this->set_termC_ixes(pairs_of_C_ixes_.val);
        };

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        std::cerr << "Code should not invoke DerivativeExpSquaredKernel.value()";
        exit(1);
        return 0.;
    }

    virtual double get_radial_gradient (const double* x1, const double* x2) {
        std::cerr << "Shouldn't be calling DerivativeExpSquaredKernel.get_radial_gradient()";
        exit(1);
        return 0;
    };

    double metric_gradient (const double* x1, const double* x2, double* grad) {
        int i, n = this->metric_->size();
        double r2 = this->metric_->gradient(x1, x2, grad),
               kg = this->get_radial_gradient(x1, x2);
        for (i = 0; i < n; ++i) grad[i] *= kg;
        return r2;
    };

    void print_1D_vec(const vector <double> vec1D, std::string name) const {
        cout << name << ": " << endl;
        for (unsigned int i = 0; i < vec1D.size(); i++){
                cout << vec1D[i] << ", ";
        }
        cout << endl << endl;
    }

    void print_1D_vec(const vector <int> vec1D, std::string name) const {
        cout << name << ": " << endl;
        for (unsigned int i = 0; i < vec1D.size(); i++){
                cout << vec1D[i] << ", ";
        }
        cout << endl << endl;
    }

    void print_2D_vec(const vector <vector <int> > vec2D, std::string name) const {
        cout << name << ": " << endl;
        for (unsigned int i = 0; i < vec2D.size(); i++){
            for (unsigned int j = 0; j < vec2D[0].size(); j++){
                cout << vec2D[i][j] << ", ";
            }
            cout << endl;
        }
        cout << endl;
    }

protected:
    // M* metric_;
    george::TwoDimensionalDynamicArray<unsigned int> pairs_of_B_ixes_;
    george::TwoDimensionalDynamicArray<unsigned int> pairs_of_C_ixes_;
    george::TwoDimensionalDynamicArray<unsigned int> comb_B_ixes_;
    george::TwoDimensionalDynamicArray<unsigned int> comb_C_ixes_;
    // vector < vector <int> > comb_B_ixes_;
    // vector < vector <int> > comb_C_ixes_;
    
    double X(const double* x1, const double* x2, const int spatial_ix){
        /* l_sq fixed  */

        // printf ("Inside X: \n");
        // printf ("x1[spatial_ix] = %2f\n", x1[spatial_ix]);
        // printf ("x2[spatial_ix] = %2f\n", x2[spatial_ix]);
        // printf ("this->metric_->get_parameter(spatial_ix) = %2f\n",
        //         this->metric_->get_parameter(spatial_ix));
        // printf ("spatial_ix = %d\n", spatial_ix);
        return (x1[spatial_ix] - x2[spatial_ix]) /
            this->metric_->get_parameter(spatial_ix);
    }

    void set_termB_ixes(unsigned int** v2d){
        unsigned int r, c;
        const int rows = 6, cols = 4;

        unsigned int arr[rows][cols] = {{0, 1, 2, 3},
                                        {0, 2, 1, 3},
                                        {0, 3, 1, 2},
                                        {2, 3, 0, 1},
                                        {1, 3, 0, 2},
                                        {1, 2, 0, 3}};

        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++ ) { v2d[r][c] = arr[r][c]; }
        }
    }

    void set_termC_ixes(unsigned int** v2d){
        unsigned int r, c;
        const int rows = 3, cols = 4;


        unsigned int arr[rows][cols] = {{0, 1, 2, 3},
                                        {0, 2, 1, 3},
                                        {0, 3, 1, 2}};

        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) { v2d[r][c] = arr[r][c]; }
        }
    }

    double termA(const double* x1, const double* x2, const unsigned int* ix) {
        double term = 1.;
        for (unsigned int i=0; i < 4; i++){
            term *= this->X(x1, x2, ix[i]);
        }
        return term;
    }

    double termB(const double* x1, const double* x2, const unsigned int* ix) {
    // double termB(const double* x1, const double* x2, const vector<int> ix) {
        /* l_sq fixed  */
        if (ix[2] != ix[3]) { return 0; }

        // printf ("termB: ix[1] = %d\n", ix[1]);
        // printf ("termB: ix[0] = %d\n", ix[0]);
        // printf ("termB = %.2f\n", this->X(x1, x2, ix[0]) *
        //         this->X(x1, x2, ix[1] * this->get_parameter(ix[2])));
        return this->X(x1, x2, ix[0]) * this->X(x1, x2, ix[1]) /
            this->metric_->get_parameter(ix[2]);
    }

    double termC(const unsigned int* ix) {
        /* l_sq fixed  */
        if (ix[0] != ix[1]) { return 0.; }
        if (ix[2] != ix[3]) { return 0.; }
        // printf ("termC: ix[2] = %2d\n", ix[2]);
        // printf ("termC: this->metric_->get_parameter(ix[2]) = %.2f\n",
        //         this->metric_->get_parameter(ix[2]));
        // printf ("termC: ix[0] = %2d\n", ix[0]);
        // printf ("termC: this->metric_->get_parameter(ix[0]) = %.2f\n",
        //         this->metric_->get_parameter(ix[0]));
        return 1. / (this->metric_->get_parameter(ix[2]) *
            this->metric_->get_parameter(ix[0]));
    }

    void set_combine_B_ixes(const unsigned int* kernel_B_ix, const int& term_no){
               
        unsigned int rows = this->pairs_of_B_ixes_.row_no, 
                     cols = this->pairs_of_B_ixes_.col_no;

        int actual_row = 0;

        for (unsigned int r = 0; r < rows; r++){
            actual_row = r + rows * term_no;
            for (unsigned int c = 0; c < cols; c++){
                comb_B_ixes_.val[actual_row][c] = kernel_B_ix[pairs_of_B_ixes_.val[r][c]];
            }
        }
    }

    void set_combine_C_ixes(const unsigned int* kernel_C_ix, const int& term_no){
        unsigned int rows = this->pairs_of_C_ixes_.row_no, 
                     cols = this->pairs_of_C_ixes_.col_no;
        int actual_row = 0;

        for (unsigned int r = 0; r < rows; r++){
            actual_row = r + rows * term_no;
            for (unsigned int c = 0; c < cols; c++){
                 comb_C_ixes_.val[actual_row][c] = kernel_C_ix[pairs_of_C_ixes_.val[r][c]];
            }
        }
    }


    double Sigma4thDeriv(const int term_no, const unsigned int* ix, 
                         const double* x1, const double* x2){
        double allTermBs = 0.;
        double allTermCs = 0.;
        
        int row;  // C++ is row major - bad for performance - oh well.
        // have to consider each variation of the 4 terms on eqn. (2 - 7)
        const int b_row_begin = term_no * 6; 
        const int b_row_end = b_row_begin + 6;

        const int c_row_begin = term_no * 3; 
        const int c_row_end = c_row_begin + 3;
        
        double termA_val = termA(x1, x2, ix);

        for (row = b_row_begin; row < b_row_end; row++){
            allTermBs += termB(x1, x2, comb_B_ixes_.val[row]);
        }

        for (row = c_row_begin; row < c_row_end; row++){
            allTermCs += termC(comb_C_ixes_.val[row]);
        }

        // printf ("combined terms in Sigma4thDeriv = %.2f \n",
        //        (termA - allTermBs + allTermCs) / 4.);
        // beta = metric currently and are multipled within the functions
        // for getting each term  so we don't have to multiply beta again.
        return (termA_val - allTermBs + allTermCs) / 4.;
    }

    double compute_Sigma4deriv_matrix(const double* x1, const double* x2,
                                      unsigned int** ix_list,
                                      const vector<double>& terms_signs){
        double term = 0;

        for (unsigned int r = 0; r < 4; r++){
            term += terms_signs[r] * this->Sigma4thDeriv(r, ix_list[r], x1, x2);
        }
        return term;
    }
};

template <typename M>
class KappaKappaExpSquaredKernel: public DerivativeExpSquaredKernel<M>{
public:
    KappaKappaExpSquaredKernel (const long ndim, M* metric):
        DerivativeExpSquaredKernel<M>(ndim, metric){

           ix_list_.allocate_memory(4, 4);
           this->set_ix_list(this->ix_list_.val);
           this->set_terms_signs(this->terms_signs_);

            unsigned int term_no;
            for (term_no = 0; term_no < this->ix_list_.row_no; term_no++){
                // comb_B_ixes_ and comb_C_ixes_ are (4 terms x 6 perm.) x 4 col in size
                // organised as 24 rows by 4 cols 
                this->set_combine_B_ixes(this->ix_list_.val[term_no], term_no);
                this->set_combine_C_ixes(this->ix_list_.val[term_no], term_no);
            }

        };

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        return exp(-0.5 * this->get_squared_distance(x1, x2))
            * this->compute_Sigma4deriv_matrix(x1, x2, this->ix_list_.val, 
                                               this->terms_signs_);
    };

    double get_radial_gradient (const double* x1, const double* x2) {
        printf("KappaKappaExpSquaredKernel.get_radial_gradient invoked\n");
        return -0.5 * this->value(x1, x2);
    };

private:
    TwoDimensionalDynamicArray<unsigned int> ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(unsigned int** v2d){
        const unsigned int arr[4][4] = {{0, 0, 0, 0},
                                        {0, 0, 1, 1},
                                        {1, 1, 0, 0},
                                        {1, 1, 1, 1}};


        for (unsigned int r = 0; r < 4; r++){
            for (unsigned int c = 0; c < 4; c++){ v2d[r][c] = arr[r][c]; }
        }
    }

    void set_terms_signs(vector<double>& signs){
        // reserve memory for vectors 
        signs.reserve(4);
        const double arr[4] = {1., 1., 1., 1.};
        for (unsigned int c = 0; c < 4; c++){ signs[c] = arr[c]; }
    }

};

template <typename M>
class KappaGamma1ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    KappaGamma1ExpSquaredKernel (const long ndim, M* metric):
       DerivativeExpSquaredKernel<M>(ndim, metric){

           ix_list_.allocate_memory(4, 4);
           this->set_ix_list(this->ix_list_.val);
           this->set_terms_signs(this->terms_signs_);

            unsigned int term_no;
            for (term_no = 0; term_no < this->ix_list_.row_no; term_no++){
                // comb_B_ixes_ and comb_C_ixes_ are (4 terms x 6 perm.) x 4 col in size
                // organised as 24 rows by 4 cols 
                this->set_combine_B_ixes(this->ix_list_.val[term_no], term_no);
                this->set_combine_C_ixes(this->ix_list_.val[term_no], term_no);
            }
       };

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        return exp(-0.5 * this->get_squared_distance(x1, x2))
            * this->compute_Sigma4deriv_matrix(x1, x2, this->ix_list_.val, 
                                               this->terms_signs_);
    };

    double get_radial_gradient (const double* x1, const double* x2) {
        // printf("KappaGamma1ExpSquaredKernel.get_radial_gradient invoked\n");
        return -0.5 * this->value(x1, x2);
    };


private:
    TwoDimensionalDynamicArray<unsigned int> ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(unsigned int** v2d){
        const unsigned int arr[4][4] = {{0, 0, 0, 0},
                                        {0, 0, 1, 1},
                                        {1, 1, 0, 0},
                                        {1, 1, 1, 1}};

        unsigned int r = 0, c = 0;
        for (r = 0; r < 4; r++){
            for (c = 0; c < 4; c++){ v2d[r][c] = arr[r][c]; }
        }
    }

    void set_terms_signs(vector<double>& signs){
        signs.reserve(4);
        const double arr[4] = {1., -1., 1., -1.};
        for (unsigned int c = 0; c < 4; c++){ signs[c] = arr[c]; }
    }
};

template <typename M>
class KappaGamma2ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    KappaGamma2ExpSquaredKernel (const long ndim, M* metric):
       DerivativeExpSquaredKernel<M>(ndim, metric){

           ix_list_.allocate_memory(4, 4);
           this->set_ix_list(this->ix_list_.val);
           this->set_terms_signs(this->terms_signs_);

            unsigned int term_no;
            for (term_no = 0; term_no < this->ix_list_.row_no; term_no++){
                // comb_B_ixes_ and comb_C_ixes_ are (4 terms x 6 perm.) x 4 col in size
                // organised as 24 rows by 4 cols 
                this->set_combine_B_ixes(this->ix_list_.val[term_no], term_no);
                this->set_combine_C_ixes(this->ix_list_.val[term_no], term_no);
            }
       };

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        return exp(-0.5 * this->get_squared_distance(x1, x2))
            * this->compute_Sigma4deriv_matrix(x1, x2, this->ix_list_.val, 
                                               this->terms_signs_);
    };

    double get_radial_gradient (const double* x1, const double* x2) {
        // printf("KappaGamma2ExpSquaredKernel.get_radial_gradient invoked\n");
        return -0.5 * this->value(x1, x2) ;
    };


private:
    TwoDimensionalDynamicArray<unsigned int> ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(unsigned int** v2d){
        unsigned int r = 0, c = 0;

        const unsigned int arr[4][4] = {{0, 0, 0, 1},
                                        {0, 0, 1, 0},
                                        {1, 1, 0, 1},
                                        {1, 1, 1, 0}};

        for (r = 0; r < 4; r++){
            for (c = 0; c < 4; c++){ v2d[r][c] = arr[r][c]; }
        }
    }

    void set_terms_signs(vector<double>& signs){
        signs.reserve(4);
        const double arr[4] = {1., 1., 1., 1.};
        for (unsigned int c = 0; c < 4; c++){ signs[c] = arr[c]; }
    }
};

template <typename M>
class Gamma1Gamma1ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    Gamma1Gamma1ExpSquaredKernel (const long ndim, M* metric):
       DerivativeExpSquaredKernel<M>(ndim, metric){

           ix_list_.allocate_memory(4, 4);

           this->set_ix_list(this->ix_list_.val);
           this->set_terms_signs(this->terms_signs_);

           unsigned int term_no;
           for (term_no = 0; term_no < this->ix_list_.row_no; term_no++){
               // comb_B_ixes_ and comb_C_ixes_ are (4 terms x 6 perm.) x 4 col in size
               // organised as 24 rows by 4 cols 
               this->set_combine_B_ixes(this->ix_list_.val[term_no], term_no);
               this->set_combine_C_ixes(this->ix_list_.val[term_no], term_no);
           }
       };

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        return exp(-0.5 * this->get_squared_distance(x1, x2))
            * this->compute_Sigma4deriv_matrix(x1, x2, this->ix_list_.val, 
                                               this->terms_signs_);
    };

    double get_radial_gradient (const double* x1, const double* x2) {
        printf("Gamma1Gamma1ExpSquaredKernel.get_radial_gradient invoked\n");
        return -0.5 * value(x1, x2);
    };

private:
    TwoDimensionalDynamicArray<unsigned int> ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(unsigned int** v2d){
        unsigned int r = 0, c = 0;

        const unsigned int arr[4][4] = {{0, 0, 0, 0},
                                        {0, 0, 1, 1},
                                        {1, 1, 0, 0},
                                        {1, 1, 1, 1}};

        for (r = 0; r < 4; r++){
            for (c = 0; c < 4; c++){ v2d[r][c] = arr[r][c]; }
        }
    }

    void set_terms_signs(vector<double>& signs){
        signs.reserve(4);
        const double arr[4] = {1., -1., -1., 1.};
        for (unsigned int c = 0; c < 4; c++){ signs[c] = arr[c]; }
    }
};

template <typename M>
class Gamma1Gamma2ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    Gamma1Gamma2ExpSquaredKernel (const long ndim, M* metric):
       DerivativeExpSquaredKernel<M>(ndim, metric){

            ix_list_.allocate_memory(4, 4);

            this->set_ix_list(this->ix_list_.val);
            this->set_terms_signs(this->terms_signs_);

            unsigned int term_no;
            for (term_no = 0; term_no < this->ix_list_.row_no; term_no++){
                // comb_B_ixes_ and comb_C_ixes_ are (4 terms x 6 perm.) x 4 col in size
                // organised as 24 rows by 4 cols 
                this->set_combine_B_ixes(this->ix_list_.val[term_no], term_no);
                this->set_combine_C_ixes(this->ix_list_.val[term_no], term_no);
            }
       };


    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        return exp(-0.5 * this->get_squared_distance(x1, x2))
            * this->compute_Sigma4deriv_matrix(x1, x2, this->ix_list_.val, 
                                               terms_signs_);
    };

    double get_radial_gradient (const double* x1, const double* x2) {
        printf("Gamma1Gamma2ExpSquaredKernel.get_radial_gradient invoked\n");
        return -0.5 * this->value(x1, x2);
    };

private:
    TwoDimensionalDynamicArray<unsigned int> ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(unsigned int** v2d){
        unsigned int r = 0, c = 0;

        const unsigned int arr[4][4] = {{0, 0, 0, 1},
                                        {0, 0, 1, 0},
                                        {1, 1, 0, 1},
                                        {1, 1, 1, 0}};

        for (r = 0; r < 4; r++){
            for (c = 0; c < 4; c++){ v2d[r][c] = arr[r][c]; }
        }
    }

    void set_terms_signs(vector<double>& signs){
        signs.reserve(4);
        const double arr[4] = {1., 1., -1., -1.};
        for (unsigned int c = 0; c < 4; c++){ signs[c] = arr[c]; }
    }
};

template <typename M>
class Gamma2Gamma2ExpSquaredKernel : public DerivativeExpSquaredKernel<M>{
public:
    Gamma2Gamma2ExpSquaredKernel (const long ndim, M* metric):
       DerivativeExpSquaredKernel<M>(ndim, metric){

           ix_list_.allocate_memory(4, 4);

           this->set_ix_list(this->ix_list_.val);
           this->set_terms_signs(this->terms_signs_);

           unsigned int term_no;
           for (term_no = 0; term_no < this->ix_list_.row_no; term_no++){
               // comb_B_ixes_ and comb_C_ixes_ are (4 terms x 6 perm.) x 4 col in size
               // organised as 24 rows by 4 cols 
               this->set_combine_B_ixes(this->ix_list_.val[term_no], term_no);
               this->set_combine_C_ixes(this->ix_list_.val[term_no], term_no);
           }

       };

    using Kernel::value;
    virtual double value (const double* x1, const double* x2) {
        return exp(-0.5 * this->get_squared_distance(x1, x2))
            * this->compute_Sigma4deriv_matrix(x1, x2, this->ix_list_.val, 
                                               terms_signs_);
    };

    double get_radial_gradient (const double* x1, const double* x2) {
        return -0.5 * this->value(x1, x2);
    };

private:
    TwoDimensionalDynamicArray<unsigned int> ix_list_;
    vector<double> terms_signs_;

    void set_ix_list(unsigned int** v2d){
        unsigned int r = 0, c = 0;
        const unsigned int rows = 4, cols = 4;
        const unsigned int arr[rows][cols] = {{0, 1, 0, 1},
                                              {0, 1, 1, 0},
                                              {1, 0, 0, 1},
                                              {1, 0, 1, 0}};

        for (r = 0; r < rows; r++){
            for (c = 0; c < cols; c++){ v2d[r][c] = arr[r][c]; }
        }
    }

    void set_terms_signs(vector<double>& signs){
        signs.reserve(4);
        const double arr[4] = {1., 1., 1., 1.};
        for (unsigned int c = 0; c < 4; c++){ signs[c] = arr[c]; }
    }
};

//
// 'Composite' gravitational lensing kernel
//
enum lens_field_t {kappa, gamma1, gamma2};

template <typename M>
class GravLensingExpSquaredKernel: public ExpSquaredKernel<M> {
public:
  GravLensingExpSquaredKernel (M* metric) :
    ExpSquaredKernel<M>(2, metric) { // This kernel only defined for ndim = 2

      x_max_ = 1.0;

      M* metric_kk = new M(*metric);
      kk_ = new KappaKappaExpSquaredKernel<M>(this->ndim_, metric_kk);

      M* metric_kg1 = new M(*metric);
      kg1_ = new KappaGamma1ExpSquaredKernel<M>(this->ndim_, metric_kg1);

      M* metric_kg2 = new M(*metric);
      kg2_ = new KappaGamma2ExpSquaredKernel<M>(this->ndim_, metric_kg2);

      M* metric_g1g2 = new M(*metric);
      g1g2_ = new Gamma1Gamma2ExpSquaredKernel<M>(this->ndim_, metric_g1g2);

      M* metric_g1g1 = new M(*metric);
      g1g1_ = new Gamma1Gamma1ExpSquaredKernel<M>(this->ndim_, metric_g1g1);

      M* metric_g2g2 = new M(*metric);
      g2g2_ = new Gamma2Gamma2ExpSquaredKernel<M>(this->ndim_, metric_g2g2);
    };

  ~GravLensingExpSquaredKernel () {

    delete kk_;
    delete kg1_;
    delete kg2_;
    delete g1g2_;
    delete g1g1_;
    delete g2g2_;
  }

  // Assume the GP is defined on the interval [0, x_max_]
  //
  // We parse the input x1,x2 by mapping:
  //   - [0, x_max_] -> kappa
  //   - [x_max_, 2*x_max_] -> gamma1
  //   - [2*x_max_, 3*x_max_] -> gamma2
  using Kernel::value;
  virtual double value (const double* x1, const double* x2) {
    // Figure out which kernel to use based on values of x1, x2
    // TODO: We're assuming that x1[0] and x[1] lie on the same interval.
    //  => Need to check for this.
    double x1std[this->ndim_], x2std[this->ndim_];
    lens_field_t lens_field1, lens_field2;
    for (size_t i=0; i<this->ndim_; i++) {
      if (x1[i] < x_max_) {
        lens_field1 = kappa;
        x1std[i] = x1[i];
      } else if (x1[i] < 2*x_max_) {
        lens_field1 = gamma1;
        x1std[i] = x1[i] - x_max_;
      } else if (x1[i] < 3*x_max_) {
        lens_field1 = gamma2;
        x1std[i] = x1[i] - 2*x_max_;
      } else {
        throw "GravLensingExpSquaredKernel::value -- Invalid x range";
      }
      if (x2[i] < x_max_) {
        lens_field2 = kappa;
        x2std[i] = x2[i];
      } else if (x2[i] < 2*x_max_) {
        lens_field2 = gamma1;
        x2std[i] = x2[i] - x_max_;
      } else if (x2[i] < 3*x_max_) {
        lens_field2 = gamma2;
        x2std[i] = x2[i] - 2*x_max_;
      } else {
        throw "GravLensingExpSquaredKernel::value -- Invalid x range";
      }
    }

    // Evaluate the value method of the appropriate kernel
    double result;
    switch(lens_field1)
    {
      case kappa :
        switch(lens_field2)
        {
          case kappa :
            result = kk_->value(x1std, x2std);
            break;
          case gamma1 :
            result = kg1_->value(x1std, x2std);
            break;
          case gamma2 :
            result = kg2_->value(x1std, x2std);
            break;
        }
        break;
      case gamma1 :
        switch(lens_field2)
        {
          case kappa :
            result = kg1_->value(x2std, x1std);
            break;
          case gamma1 :
            result = g1g1_->value(x1std, x2std);
            break;
          case gamma2 :
            result = g1g2_->value(x1std, x2std);
            break;
        }
        break;
      case gamma2 :
        switch(lens_field2)
        {
          case kappa :
            result = kg2_->value(x2std, x1std);
            break;
          case gamma1 :
            result = g1g2_->value(x2std, x1std);
            break;
          case gamma2 :
            result = g2g2_->value(x1std, x2std);
            break;
        }
        break;
    }
    return result;
  };

  double get_radial_gradient (double r2) const {
    printf("GravLensingExpSquaredKernel.get_radial_gradient invoked\n");
    return 0.0;
  };

private:
  double x_max_; // Assume the GP is defined on [0, x_max_]

  KappaKappaExpSquaredKernel<M> * kk_;
  KappaGamma1ExpSquaredKernel<M> * kg1_;
  KappaGamma2ExpSquaredKernel<M> * kg2_;
  Gamma1Gamma2ExpSquaredKernel<M> * g1g2_;
  Gamma1Gamma1ExpSquaredKernel<M> * g1g1_;
  Gamma2Gamma2ExpSquaredKernel<M> * g2g2_;
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
    void gradient (const double* x1, const double* x2, double* grad) {
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

    void gradient (const double* x1, const double* x2, double* grad) {
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
