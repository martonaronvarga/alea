// Full posterior value/gradient through Stan Math, no handwritten scores.
// Both vectorized and fused-GLM reference kernels use the same posterior/data.
#include <stan/math/rev.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 5) throw std::runtime_error("family data count vectorized|glm [free coordinates]");
    const std::string family = argv[1], implementation = argv[4];
    const int count = std::stoi(argv[3]);
    if (count < 1 || (implementation != "glm" && implementation != "vectorized"))
      throw std::runtime_error("invalid evaluation protocol");
    auto setup = std::chrono::steady_clock::now();
    std::ifstream input(argv[2]); int n, k; input >> n >> k;
    if (!input || n < 0 || k < 0) throw std::runtime_error("invalid input");
    const bool gaussian = family == "mean" || family == "linear";
    const int p = family == "mean" ? 1 : k;
    const int d = p + (gaussian || family == "funnel");
    if (argc != 5 && argc != 5+d) throw std::runtime_error("probe dimension mismatch");
    Eigen::MatrixXd x(n,p); Eigen::VectorXd y(n); std::vector<int> binary(n);
    for (int i=0;i<n;++i) {
      for (int j=0;j<k;++j) input >> x(i,j);
      if (family == "mean") x(i,0)=1.;
      input >> y[i]; binary[i]=int(y[i]);
    }
    if (!input) throw std::runtime_error("truncated input");
    const double construction = std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count();
    double checksum=0., gradient_checksum=0., value=0.; std::vector<double> gradient(d);
    auto start=std::chrono::steady_clock::now();
    for (int i=0;i<count;++i) {
      stan::math::nested_rev_autodiff nested;
      std::vector<stan::math::var> q(d);
      for (int j=0;j<d;++j) q[j]=argc>5 ? std::stod(argv[5+j]) : (j==0 ? (i%10)*.01 : 0.);
      Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> beta(p);
      for (int j=0;j<p;++j) beta[j]=q[j];
      stan::math::var lp;
      if (family == "funnel") {
        Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> z(k);
        for (int j=0;j<k;++j) z[j]=q[j+1];
        lp=stan::math::normal_lpdf(q[0],0.,3.)+stan::math::normal_lpdf(z,0.,stan::math::exp(q[0]*.5));
      } else {
        lp=stan::math::normal_lpdf(beta,0.,2.);
        if (gaussian) {
          auto sigma=stan::math::exp(q[p]);
          lp+=stan::math::normal_lpdf(sigma,0.,1.)+q[p];
          // Match the actual Stan mean model's shared scalar location rather
          // than burdening it with an unnecessary unit-matrix multiply/GLM.
          if (family=="mean") lp+=stan::math::normal_lpdf(y,q[0],sigma);
          else if (implementation=="glm") lp+=stan::math::normal_id_glm_lpdf(y,x,0.,beta,sigma);
          else lp+=stan::math::normal_lpdf(y,stan::math::multiply(x,beta),sigma);
        } else if (family=="logistic") {
          if (implementation=="glm") lp+=stan::math::bernoulli_logit_glm_lpmf(binary,x,0.,beta);
          else lp+=stan::math::bernoulli_logit_lpmf(binary,stan::math::multiply(x,beta));
        } else throw std::runtime_error("unknown family");
      }
      lp.grad(q,gradient); value=lp.val(); checksum+=value;
      for (int j=0;j<d;++j) gradient_checksum+=(j+1)*gradient[j];
      asm volatile("" : : "g"(value) : "memory");
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (!std::isfinite(checksum) || !std::isfinite(gradient_checksum)) throw std::runtime_error("nonfinite evaluation");
    std::cout << std::setprecision(17) << "{\"evaluations\":"<<count<<",\"evaluation_seconds\":"<<seconds
      <<",\"construction_seconds\":"<<construction<<",\"checksum\":"<<checksum
      <<",\"gradient_checksum\":"<<gradient_checksum<<",\"logp\":"<<value<<",\"gradient\":[";
    for(int j=0;j<d;++j) { if(j)std::cout<<","; std::cout<<gradient[j]; }
    std::cout<<"]}\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
