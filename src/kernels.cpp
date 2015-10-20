#include <iostream>
#include "metrics.h"
#include "kernels.h"


using george::metrics::AxisAlignedMetric;
using george::metrics::IsotropicMetric; 

namespace george {
namespace kernels {

// DerivativeExpSquaredKernel constructor
template <typename M> 
DerivativeExpSquaredKernel<M>::DerivativeExpSquaredKernel (const long ndim, M* metric):
ExpSquaredKernel<M>(ndim, metric){
     set_termB_ixes(pairs_of_B_ixes_);
     set_termC_ixes(pairs_of_C_ixes_);
};

template <typename M>
KappaKappaExpSquaredKernel<M>::KappaKappaExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
KappaGamma1ExpSquaredKernel<M>::KappaGamma1ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
KappaGamma2ExpSquaredKernel<M>::KappaGamma2ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
Gamma1Gamma1ExpSquaredKernel<M>::Gamma1Gamma1ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
Gamma1Gamma2ExpSquaredKernel<M>::Gamma1Gamma2ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

template <typename M>
Gamma2Gamma2ExpSquaredKernel<M>::Gamma2Gamma2ExpSquaredKernel (const long ndim, M* metric): 
DerivativeExpSquaredKernel<M>(ndim, metric){
    set_ix_list(ix_list_);
    set_terms_signs(terms_signs_);
};

}; // kernels 
}; // george 

void test_main(){
    const unsigned int ndim = 2;
    double gp_length = 0.5;
    IsotropicMetric* iso_metric = new IsotropicMetric(ndim);
    iso_metric->set_parameter(0, gp_length);
    
    AxisAlignedMetric* metric_kk = new AxisAlignedMetric(ndim);
    metric_kk->set_parameter(0, gp_length);
    metric_kk->set_parameter(1, gp_length);

    george::kernels::KappaKappaExpSquaredKernel<AxisAlignedMetric> kk(ndim, metric_kk);
}

// int main(){
//     test_main();
//     return 0;
// }
