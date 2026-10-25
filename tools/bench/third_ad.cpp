// Benchmark-only per-model ABI; compile beside a concrete stanc model.hpp.
// All derivatives include exactly one unconstraining Jacobian, matching BridgeStan.
#include <stan/math/mix.hpp>
#include "model.hpp"
#include <stan/io/json/json_data.hpp>
#include <stan/math/mix/functor/grad_hessian.hpp>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <exception>
namespace {
void failure(char* out,size_t capacity,const char* message) noexcept {
  if(out && capacity) {const size_t n=std::min(capacity-1,std::strlen(message));std::memcpy(out,message,n);out[n]='\0';}
}
struct Model {
  const stan_model model;
  explicit Model(const char* filename):model(load(filename)) {}
  static stan_model load(const char* filename) {
    if(!filename) throw std::invalid_argument("null data path");
    std::ifstream stream(filename);if(!stream) throw std::invalid_argument("cannot read model data");
    stan::json::json_data data(stream);return stan_model(data,17001,nullptr);
  }
};
struct Logp {
  const stan_model& model;
  template<typename S> S operator()(const Eigen::Matrix<S,-1,1>& q) const {
    auto values=q.eval();return model.template log_prob<true,true>(values,nullptr);
  }
};
}
#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT unsigned alea_third_abi() noexcept {return 1;}
EXPORT void* alea_third_new(const char* data,char* error,size_t capacity) noexcept {
  try {return new Model(data);}catch(const std::exception& e) {failure(error,capacity,e.what());}
  catch(...) {failure(error,capacity,"unknown model construction failure");}return nullptr;
}
EXPORT void alea_third_drop(void* model) noexcept {delete static_cast<Model*>(model);}
EXPORT size_t alea_third_dimension(const void* model) noexcept {
  return model ? static_cast<const Model*>(model)->model.num_params_r() : 0;
}
EXPORT int alea_third_eval(const void* handle,const double* q,size_t nq,
    double* potential_hessian,size_t nh,double* potential_third,size_t nt,
    char* error,size_t capacity) noexcept {
  try {
    if(!handle||!q||!potential_hessian||!potential_third) throw std::invalid_argument("null derivative buffer");
    const auto& model=static_cast<const Model*>(handle)->model;
    const size_t d=model.num_params_r();
    if(d==0||d>128||nq!=d||nh!=d*d||nt!=d*d*d) throw std::invalid_argument("derivative buffer shape mismatch");
    Eigen::VectorXd point(d);for(size_t k=0;k<d;++k) {if(!std::isfinite(q[k])) throw std::invalid_argument("nonfinite derivative input");point[k]=q[k];}
    double fx;Eigen::MatrixXd hessian;std::vector<Eigen::MatrixXd> third;
    stan::math::grad_hessian(Logp{model},point,fx,hessian,third);
    if(!std::isfinite(fx)) throw std::domain_error("nonfinite log density in third AD");
    // grad_H[i](j,k) = d/dq[k] d^2 logp/dq[i]dq[j].
    // Validate everything before publishing to the caller's scratch.
    for(size_t i=0;i<d;++i) for(size_t j=0;j<d;++j) {
      if(!std::isfinite(hessian(i,j))) throw std::domain_error("nonfinite AD Hessian");
      for(size_t k=0;k<d;++k) if(!std::isfinite(third[i](j,k))) throw std::domain_error("nonfinite AD third derivative");
    }
    for(size_t i=0;i<d;++i) for(size_t j=0;j<d;++j) {
      potential_hessian[i*d+j]=-hessian(i,j);
      for(size_t k=0;k<d;++k) potential_third[k*d*d+i*d+j]=-third[i](j,k);
    }
    return 0;
  }catch(const std::exception& e) {failure(error,capacity,e.what());}
  catch(...) {failure(error,capacity,"unknown third AD failure");}return 1;
}
