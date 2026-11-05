// Full posterior value/gradient through Stan Math, no handwritten scores.
// Both vectorized and fused-GLM reference kernels use the same posterior/data.
#include <stan/math/rev.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstdlib>

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
    const bool schools = family == "schools" || family == "schools_nc";
    const int d = schools ? n+2 : family == "covariance" ? 5 : p + (gaussian || family == "funnel");
    if (argc != 5 && argc != 5+d) throw std::runtime_error("probe dimension mismatch");
    Eigen::MatrixXd x(n,p); Eigen::VectorXd y(n); std::vector<int> binary(n);
    for (int i=0;i<n;++i) {
      for (int j=0;j<k;++j) input >> x(i,j);
      if (family == "mean") x(i,0)=1.;
      input >> y[i]; binary[i]=int(y[i]);
    }
    if (!input) throw std::runtime_error("truncated input");
    std::vector<double> states;
    if (const char* path = std::getenv("ALEA_BENCH_STATES")) {
      if (argc != 5) throw std::runtime_error("state bank and explicit probe coordinates conflict");
      std::ifstream bank(path);
      double coordinate;
      while (bank >> coordinate) {
        if (!std::isfinite(coordinate)) throw std::runtime_error("nonfinite state bank");
        states.push_back(coordinate);
      }
      if (!bank.eof() || states.empty() || states.size() % d)
        throw std::runtime_error("state bank requires complete finite coordinate rows");
    }
    std::vector<Eigen::Vector2d> observations;
    if(family=="covariance") for(int j=0;j<n;++j) observations.emplace_back(x.row(j).transpose());
    const double construction = std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count();
    double checksum=0., gradient_checksum=0., value=0.; std::vector<double> gradient(d);
    auto start=std::chrono::steady_clock::now();
    for (int i=0;i<count;++i) {
      stan::math::nested_rev_autodiff nested;
      std::vector<stan::math::var> q(d);
      for (int j=0;j<d;++j) q[j]=!states.empty() ? states[(i % (states.size()/d))*d+j]
        : argc>5 ? std::stod(argv[5+j]) : (j==0 ? (i%10)*.01 : 0.);
      Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> beta(p);
      for (int j=0;j<std::min(p,d);++j) beta[j]=q[j];
      stan::math::var lp;
      if (schools) {
        auto mu=q[n], tau=stan::math::exp(q[n+1]);
        lp=stan::math::normal_lpdf(mu,0.,5.)+stan::math::cauchy_lpdf(tau,0.,5.)+q[n+1];
        Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> effect(n),theta(n);
        for(int j=0;j<n;++j) effect[j]=q[j];
        if(family=="schools_nc") {
          lp+=stan::math::normal_lpdf(effect,0.,1.);
          theta=(effect*tau).array()+mu;
        } else {lp+=stan::math::normal_lpdf(effect,mu,tau); theta=effect;}
        lp+=stan::math::normal_lpdf(y,theta,x.col(0));
      } else if(family=="covariance") {
        Eigen::Matrix<stan::math::var,2,1> mu; mu<<q[0],q[1];
        auto s0=stan::math::exp(q[2]),s1=stan::math::exp(q[3]);
        Eigen::Matrix<stan::math::var,2,2> L; L<<s0,0.,q[4],s1;
        lp=stan::math::normal_lpdf(mu,0.,2.)+stan::math::normal_lpdf(s0,0.,1.)
          +stan::math::normal_lpdf(s1,0.,1.)+stan::math::normal_lpdf(q[4],0.,1.)+q[2]+q[3];
        lp+=stan::math::multi_normal_cholesky_lpdf(observations,mu,L);
      } else if (family == "funnel") {
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
