#include <stdio.h>
#include <iostream>
#include "metrics.h"
#include "deriv_kernels.h"

using george::metrics::AxisAlignedMetric;
using george::metrics::IsotropicMetric; 
  
namespace george {
namespace kernels {

//
// DerivativeExpSquaredKernel
// 
template <typename M> 
DerivativeExpSquaredKernel<M>::DerivativeExpSquaredKernel(const long ndim, M* metric):
ExpSquaredKernel<M>(ndim, metric){
     set_termB_ixes(pairs_of_B_ixes_);
     set_termC_ixes(pairs_of_C_ixes_);
};

template <typename M> 
double DerivativeExpSquaredKernel<M>::value (const double* x1, const double* x2) {
    std::cerr << "Code should not invoke DerivativeExpSquaredKernel.value()";
    exit(1);
    return 0.;
}

template <typename M> 
double DerivativeExpSquaredKernel<M>::get_radial_gradient (const double* x1, const double* x2) {
    std::cerr << "Shouldn't be calling DerivativeExpSquaredKernel.get_radial_gradient()";
    exit(1);
    return 0;
};

template <typename M> 
double DerivativeExpSquaredKernel<M>::metric_gradient (const double* x1, const double* x2, double* grad) {
    int i, n = this->metric_->size();
    double r2 = this->metric_->gradient(x1, x2, grad),
           // gets the alternative get_radial_gradient method
           // otherwise will get error
           kg = this->get_radial_gradient(x1, x2);
    for (i = 0; i < n; ++i) grad[i] *= kg;
    return r2;
};

template <typename M> 
void DerivativeExpSquaredKernel<M>::print_1D_vec(const vector <double> vec1D, std::string name) const {
    cout << name << ": " << endl;
    for (unsigned int i = 0; i < vec1D.size(); i++){
            cout << vec1D[i] << ", ";
    }
    cout << endl << endl;
}

template <typename M> 
void DerivativeExpSquaredKernel<M>::print_1D_vec(const vector <int> vec1D, std::string name) const {
    cout << name << ": " << endl;
    for (unsigned int i = 0; i < vec1D.size(); i++){
            cout << vec1D[i] << ", ";
    }
    cout << endl << endl;
}

template <typename M> 
void DerivativeExpSquaredKernel<M>::print_2D_vec(const vector <vector <int> > vec2D, 
                                                 std::string name) const {
    cout << name << ": " << endl;
    for (unsigned int i = 0; i < vec2D.size(); i++){
        for (unsigned int j = 0; j < vec2D[0].size(); j++){
            cout << vec2D[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl;
}

template <typename M> 
double DerivativeExpSquaredKernel<M>::X(const double* x1, const double* x2, const int spatial_ix){
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

template <typename M>
void DerivativeExpSquaredKernel<M>::set_termB_ixes(vector <vector <int> >& v2d){
    v2d.clear();  // this sets v2d to be an empty vector
    unsigned int r, c;
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
    cout << "termB rows = " << v2d.size() << endl;
    cout << "termB cols = " << v2d[0].size() << endl;
}

template <typename M>
void DerivativeExpSquaredKernel<M>::set_termC_ixes(vector <vector <int> >& v2d){
    v2d.clear();   // this clears all elements in v2d
    unsigned int r, c;
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
    cout << "termC rows = " << v2d.size() << endl;
    cout << "termC cols = " << v2d[0].size() << endl;
}

template <typename M>
double DerivativeExpSquaredKernel<M>::termA(const double* x1, const double* x2, const vector<int> ix) {
    double term = 1.;
    for (unsigned int i=0; i < ix.size(); i++){
        term *= this->X(x1, x2, ix[i]);
    }
    return term;
}

template <typename M>
double DerivativeExpSquaredKernel<M>::termB(const double* x1, const double* x2,
                                            const vector<int> ix) {
    /* l_sq fixed  */
    if (ix[2] != ix[3]) { return 0; }

    // printf ("termB: ix[1] = %d\n", ix[1]);
    // printf ("termB: ix[0] = %d\n", ix[0]);
    // printf ("termB = %.2f\n", this->X(x1, x2, ix[0]) *
    //         this->X(x1, x2, ix[1] * this->get_parameter(ix[2])));
    return this->X(x1, x2, ix[0]) * this->X(x1, x2, ix[1]) /
        this->metric_->get_parameter(ix[2]);
}

template <typename M>
double DerivativeExpSquaredKernel<M>::termC(const vector<int> ix) {
    /* l_sq fixed  */
    if (ix[0] != ix[1]) { return 0; }
    if (ix[2] != ix[3]) { return 0; }
    // printf ("termC: ix[2] = %2d\n", ix[2]);
    // printf ("termC: this->metric_->get_parameter(ix[2]) = %.2f\n",
    //         this->metric_->get_parameter(ix[2]));
    // printf ("termC: ix[0] = %2d\n", ix[0]);
    // printf ("termC: this->metric_->get_parameter(ix[0]) = %.2f\n",
    //         this->metric_->get_parameter(ix[0]));
    return 1. / (this->metric_->get_parameter(ix[2]) *
        this->metric_->get_parameter(ix[0]));
}

template <typename M>
void DerivativeExpSquaredKernel<M>::set_combine_B_ixes(const vector<int>& kernel_B_ix){
    comb_B_ixes_.clear();  // set comb_B_ixes_ to be an empty vector
    unsigned int rows = this->pairs_of_B_ixes_.size(), 
                 cols = this->pairs_of_B_ixes_[0].size();
    vector<int> temp_row;

    cout << "temp row to be added to comb_B_ixes_" << endl;
    for (unsigned int r = 0; r < rows; r++){
        temp_row.clear();
        for (unsigned int c = 0; c < cols; c++){
            temp_row.push_back(kernel_B_ix[pairs_of_B_ixes_[r][c]]);
        }
        print_1D_vec(temp_row, "temp_row");
        comb_B_ixes_.push_back(temp_row);
    }
    // print_1D_vec(kernel_B_ix, "B_ix");
    // print_2D_vec(this->comb_B_ixes_, "comb_B_ixes");
}

template <typename M>
void DerivativeExpSquaredKernel<M>::set_combine_C_ixes(const vector<int>& kernel_C_ix){
    // set comb_C_ixes_ to be an empty vector
    comb_C_ixes_.clear();
    unsigned int rows = this->pairs_of_C_ixes_.size(), 
                 cols = this->pairs_of_C_ixes_[0].size();
    vector<int> temp_row;

    cout << "temp row to be added to comb_C_ixes_" << endl;
    cout << "rows to add = " << rows << endl;
    cout << "cols to add = " << cols << endl;
    for (unsigned int r = 0; r < rows; r++){
        temp_row.clear();
        for (unsigned int c = 0; c < cols; c++){
            temp_row.push_back(kernel_C_ix[pairs_of_C_ixes_[r][c]]);
        }
        print_1D_vec(temp_row, "temp_row");
        comb_C_ixes_.push_back(temp_row);
    }
    print_1D_vec(kernel_C_ix, "C_ix");
    print_2D_vec(comb_C_ixes_, "comb_C_ixes");
}

template <typename M>
double DerivativeExpSquaredKernel<M>::Sigma4thDeriv(const vector<int>& ix, const double* x1, const double* x2){
    double allTermBs = 0.;
    double allTermCs = 0.;
    this->set_combine_B_ixes(ix);
    this->set_combine_C_ixes(ix);

    double termA_val = termA(x1, x2, ix);

    for (vector< vector<int> >::iterator row_it = this->comb_B_ixes_.begin();
       row_it < this->comb_B_ixes_.end(); ++row_it ){
       allTermBs += termB(x1, x2, *row_it);
    }

    for (vector< vector<int> >::iterator row_it = this->comb_C_ixes_.begin();
       row_it < this->comb_C_ixes_.end(); ++row_it ){
       allTermCs += termC(*row_it);
    }

    // printf ("combined terms in Sigma4thDeriv = %.2f \n",
    //        (termA - allTermBs + allTermCs) / 4.);
    // beta = metric currently and are multipled within the functions
    // for getting each term  so we don't have to multiply beta again.
    return (termA_val - allTermBs + allTermCs) / 4.;
}

template <typename M>
double DerivativeExpSquaredKernel<M>::compute_Sigma4deriv_matrix(const double* x1, 
        const double* x2, const vector< vector<int> >& ix, const vector<double>& signs){
    double term = 0;
    int rows = ix.size();  // this should be 4
    // this->print_2D_vec(ix, "ix_list");
    // this->print_1D_vec(signs, "terms_signs");

    for (unsigned int r = 0; r < rows; r++){
        term += signs[r] * this->Sigma4thDeriv(ix[r], x1, x2);
    }
    return term;
}

// 
// KappaKappaExpSquaredKernel
// 
template <typename M>
KappaKappaExpSquaredKernel<M>::KappaKappaExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
void KappaKappaExpSquaredKernel<M>::set_ix_list(vector < vector<int> >& v2d){
    v2d.clear();  // resets v2d 
    vector<int> rowvec;
    const int rows = 4, cols = 4;
    int arr[rows][cols] = {{0, 0, 0, 0},
                           {0, 0, 1, 1},
                           {1, 1, 0, 0},
                           {1, 1, 1, 1}};

    for (unsigned int r = 0; r < rows; r++){
        rowvec.clear();
        for (unsigned int c = 0; c < cols; c++){ rowvec.push_back(arr[r][c]); }
        v2d.push_back(rowvec);
    }
}

template <typename M>
double KappaKappaExpSquaredKernel<M>::value (const double* x1, const double* x2) {
    return exp(-0.5 * this->get_squared_distance(x1, x2))
        * this->compute_Sigma4deriv_matrix(x1, x2, ix_list_, 
                                           terms_signs_);
};

template <typename M>
double KappaKappaExpSquaredKernel<M>::get_radial_gradient (const double* x1, const double* x2) {
    printf("KappaKappaExpSquaredKernel.get_radial_gradient invoked\n");
    return -0.5 * this->value(x1, x2);
};

template <typename M>
void KappaKappaExpSquaredKernel<M>::set_terms_signs(vector<double>& signs){
    const double arr[4] = {1., 1., 1., 1.};
    for (unsigned int c = 0; c < 4; c++){ signs.push_back(arr[c]); }
}

//
// KappaGamma1ExpSquaredKernel
//
template <typename M>
KappaGamma1ExpSquaredKernel<M>::KappaGamma1ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
double KappaGamma1ExpSquaredKernel<M>::value (const double* x1, const double* x2) {
    return exp(-0.5 * this->get_squared_distance(x1, x2))
        * this->compute_Sigma4deriv_matrix(x1, x2, ix_list_, 
                                           terms_signs_);
};

template <typename M>
double KappaGamma1ExpSquaredKernel<M>::get_radial_gradient (const double* x1, const double* x2) {
    printf("KappaGamma1ExpSquaredKernel.get_radial_gradient invoked\n");
    return -0.5 * this->value(x1, x2);
};

template <typename M>
void KappaGamma1ExpSquaredKernel<M>::set_ix_list(vector< vector<int> >& v2d){
    const int rows = 4, cols = 4;
    int arr[rows][cols] = {{0, 0, 0, 0},
                           {0, 0, 1, 1},
                           {1, 1, 0, 0},
                           {1, 1, 1, 1}};

    vector<int> rowvec;
    unsigned int r = 0, c = 0;
    for (r = 0; r < rows; r++){
        rowvec.clear();
        for (c = 0; c < cols; c++){ rowvec.push_back(arr[r][c]); }
        v2d.push_back(rowvec);
    }
}

template <typename M>
void KappaGamma1ExpSquaredKernel<M>::set_terms_signs(vector<double>& signs){
    const double arr[4] = {1., -1., 1., -1.};
    for (unsigned int c = 0; c < 4; c++){ signs.push_back(arr[c]); }
}

//
// KappaGamma2ExpSquaredKernel
//
template <typename M>
KappaGamma2ExpSquaredKernel<M>::KappaGamma2ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
double KappaGamma2ExpSquaredKernel<M>::value(const double* x1, const double* x2) {
    return exp(-0.5 * this->get_squared_distance(x1, x2))
        * this->compute_Sigma4deriv_matrix(x1, x2, ix_list_, 
                                           terms_signs_);
};

template <typename M>
double KappaGamma2ExpSquaredKernel<M>::get_radial_gradient (const double* x1, const double* x2) {
    printf("KappaGamma2ExpSquaredKernel.get_radial_gradient invoked\n");
    return -0.5 * this->value(x1, x2);
};

template <typename M>
void KappaGamma2ExpSquaredKernel<M>::set_ix_list(vector< vector<int> >& v2d){
    vector<int> rowvec;
    unsigned int r = 0, c = 0;
    const int rows = 4, cols = 4;
    int arr[rows][cols] = {{0, 0, 0, 1},
                           {0, 0, 1, 0},
                           {1, 1, 0, 1},
                           {1, 1, 1, 0}};

    for (r = 0; r < rows; r++){
        rowvec.clear();
        for (c = 0; c < cols; c++){ rowvec.push_back(arr[r][c]); }
        v2d.push_back(rowvec);
    }
}

template <typename M>
void KappaGamma2ExpSquaredKernel<M>::set_terms_signs(vector<double>& signs){
    const double arr[4] = {1., 1., 1., 1.};
    for (unsigned int c = 0; c < 4; c++){ signs.push_back(arr[c]); }
}

//
// Gamma1Gamma1ExpSquaredKernel
//
template <typename M>
Gamma1Gamma1ExpSquaredKernel<M>::Gamma1Gamma1ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
double Gamma1Gamma1ExpSquaredKernel<M>::value (const double* x1, const double* x2) {
    return exp(-0.5 * this->get_squared_distance(x1, x2))
        * this->compute_Sigma4deriv_matrix(x1, x2, ix_list_, 
                                           terms_signs_);
};

template <typename M>
double Gamma1Gamma1ExpSquaredKernel<M>::get_radial_gradient (const double* x1, const double* x2) {
    printf("Gamma1Gamma1ExpSquaredKernel.get_radial_gradient invoked\n");
    return -0.5 * this->value(x1, x2);
};

template <typename M>
void Gamma1Gamma1ExpSquaredKernel<M>::set_ix_list(vector< vector<int> >& v2d){
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
}

template <typename M>
void Gamma1Gamma1ExpSquaredKernel<M>::set_terms_signs(vector<double>& signs){
    const double arr[4] = {1., -1., -1., 1.};
    for (unsigned int c = 0; c < 4; c++){ signs.push_back(arr[c]); }
}

//
// Gamma1Gamma2ExpSquaredKernel
//
template <typename M>
Gamma1Gamma2ExpSquaredKernel<M>::Gamma1Gamma2ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
double Gamma1Gamma2ExpSquaredKernel<M>::value (const double* x1, const double* x2) {
    return exp(-0.5 * this->get_squared_distance(x1, x2))
        * this->compute_Sigma4deriv_matrix(x1, x2, ix_list_, 
                                           terms_signs_);
};

template <typename M>
double Gamma1Gamma2ExpSquaredKernel<M>::get_radial_gradient (const double* x1, const double* x2) {
    printf("Gamma1Gamma2ExpSquaredKernel.get_radial_gradient invoked\n");
    return -0.5 * this->value(x1, x2);
};

template <typename M>
void Gamma1Gamma2ExpSquaredKernel<M>::set_ix_list(vector< vector<int> >& v2d){
    vector<int> rowvec;
    unsigned int r = 0, c = 0;
    const int rows = 4, cols = 4;
    int arr[rows][cols] = {{0, 0, 0, 1},
                           {0, 0, 1, 0},
                           {1, 1, 0, 1},
                           {1, 1, 1, 0}};

    for (r = 0; r < rows; r++){
        rowvec.clear();
        for (c = 0; c < cols; c++){ rowvec.push_back(arr[r][c]); }
        v2d.push_back(rowvec);
    }
}

template <typename M>
void Gamma1Gamma2ExpSquaredKernel<M>::set_terms_signs(vector<double>& signs){
    const double arr[4] = {1., 1., -1., -1.};
    for (unsigned int c = 0; c < 4; c++){ signs.push_back(arr[c]); }
}

//
// Gamma2Gamma2ExpSquaredKernel
//

// constructor
template <typename M>
Gamma2Gamma2ExpSquaredKernel<M>::Gamma2Gamma2ExpSquaredKernel (const long ndim, M* metric): DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
double Gamma2Gamma2ExpSquaredKernel<M>::value (const double* x1, const double* x2) {
    return exp(-0.5 * this->get_squared_distance(x1, x2))
        * this->compute_Sigma4deriv_matrix(x1, x2, ix_list_, 
                                           terms_signs_);
};

template <typename M>
double Gamma2Gamma2ExpSquaredKernel<M>::get_radial_gradient (const double* x1, const double* x2) {
    printf("Gamma2Gamma2ExpSquaredKernel.get_radial_gradient invoked\n");
    return -0.5 * this->value(x1, x2);
};

template <typename M>
void Gamma2Gamma2ExpSquaredKernel<M>::set_ix_list(vector< vector<int> >& v2d){
    // vector< vector<int> > v2d;
    vector<int> rowvec;
    unsigned int r = 0, c = 0;
    const int rows = 4, cols = 4;
    int arr[rows][cols] = {{0, 1, 0, 1},
                           {0, 1, 1, 0},
                           {1, 0, 0, 1},
                           {1, 0, 1, 0}};

    for (r = 0; r < rows; r++){
        rowvec.clear();
        for (c = 0; c < cols; c++){ rowvec.push_back(arr[r][c]); }
        v2d.push_back(rowvec);
    }
}

template <typename M>
void Gamma2Gamma2ExpSquaredKernel<M>::set_terms_signs(vector<double>& signs){
    const double arr[4] = {1., 1., 1., 1.};
    for (unsigned int c = 0; c < 4; c++){ signs.push_back(arr[c]); }
}


//
// 'Composite' gravitational lensing kernel
//
template <typename M>
GravLensingExpSquaredKernel<M>::GravLensingExpSquaredKernel (M* metric):
ExpSquaredKernel<M>(2, metric) {  // This kernel only defined for ndim = 2
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

// destructor
template <typename M>
GravLensingExpSquaredKernel<M>::~GravLensingExpSquaredKernel () {
    delete kk_;
    delete kg1_;
    delete kg2_;
    delete g1g2_;
    delete g1g1_;
    delete g2g2_;
}

template <typename M>
double GravLensingExpSquaredKernel<M>::value (const double* x1, const double* x2) {
    // Assume the GP is defined on the interval [0, x_max_]
    //
    // We parse the input x1,x2 by mapping:
    //   - [0, x_max_] -> kappa
    //   - [x_max_, 2*x_max_] -> gamma1
    //   - [2*x_max_, 3*x_max_] -> gamma2
    double x1std[this->ndim_], x2std[this->ndim_];

    // Figure out which kernel to use based on values of x1, x2
    // TODO: We're assuming that x1[0] and x[1] lie on the same interval.
    //  => Need to check for this.
    lens_field_t lens_field1, lens_field2;
    for (size_t i=0; i < this->ndim_; i++) {
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
            std::cerr << "GravLensingExpSquaredKernel::value -- Invalid x range";
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
            std::cerr << "GravLensingExpSquaredKernel::value -- Invalid x range";
            throw "GravLensingExpSquaredKernel::value -- Invalid x range";
        }
    }

    // Evaluate the value method of the appropriate kernel
    double result;
    switch(lens_field1){
        case kappa :
            switch(lens_field2){
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
             switch(lens_field2){
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
            switch(lens_field2){
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
}

// constructor 
TwoDdynamicArray::TwoDdynamicArray(const int& nrow, const int& ncol) : nrow(nrow), ncol(ncol){
    // There is a certain number of `new` in this constructor.
    // There MUST be the same number of `delete` statements in the destructor.
    this->val = new double* [nrow];
    for (unsigned int row=0; row < nrow; row++) this->val[row] = new double[ncol];
}



// destructor 
TwoDdynamicArray::~TwoDdynamicArray(){
    for (unsigned int row=0; row < this->nrow; row++) delete [] this->val[row];
    delete [] this->val;
}

// initialize empty 2D array 
void TwoDdynamicArray::create_from_2D_arr(const double pts[][2], const int& nobs){

    for (unsigned int row=0; row < this->nrow; row++){
        for (unsigned int col=0; col < this->ncol; col++){
            this->val[row][col] = pts[row][col];
            printf("(%d, %d): %f\n", row, col, this->val[row][col]);
        }
    }
}

}; // kernels 
}; // george 

void test_kappakappa_2_coords_fixed_l_sq(){
    const unsigned int ndim = 2;
    double gp_length = 1.0;
    const int nobs = 2;

    double pts[][2] = {{1., 2.}, {4., 7.}};
    george::kernels::TwoDdynamicArray arr(nobs, ndim);
    arr.create_from_2D_arr(pts, nobs);

    AxisAlignedMetric* metric_kk = new AxisAlignedMetric(ndim);
    metric_kk->set_parameter(1, gp_length);

    george::kernels::KappaKappaExpSquaredKernel<AxisAlignedMetric> kk(ndim, metric_kk);
    double val = kk.value(arr.val[0], arr.val[1]);
    cout << "KK value = " << val << endl;
}

int main(){
    test_kappakappa_2_coords_fixed_l_sq();
    return 0;
}
