#include <iomanip>
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <walnutpie/walnuts.hpp>

struct Gaussian {
  void operator()(const Eigen::VectorXd& q, double& logp,
                  Eigen::VectorXd& gradient) const {
    logp = -0.5 * q.squaredNorm();
    gradient = -q;
  }
};

struct Handler {
  Eigen::VectorXd last;
  void on_sample(const Eigen::VectorXd& position, double) { last = position; }
  void on_logp_exception(const Eigen::VectorXd&,
                         const std::exception&) noexcept {}
};

int main() {
  std::cout << std::setprecision(17);
  std::cout << "seed,draw,q0,q1,logp\n";
  Gaussian target;
  for (unsigned seed : {17U, 31U}) {
    std::mt19937_64 rng(seed);
    Handler handler;
    Eigen::VectorXd initial(2);
    initial << 0.7, -0.2;
    Eigen::VectorXd inverse_mass = Eigen::VectorXd::Ones(2);
    walnutpie::WalnutsSampler<Gaussian, std::mt19937_64, Handler> sampler(
        rng, handler, target, initial, inverse_mass, 0.2, 4, 8, 1, 0.5);
    for (int draw = 0; draw < 8; ++draw) {
      double logp = sampler();
      std::cout << seed << ',' << draw << ',' << handler.last[0] << ','
                << handler.last[1] << ',' << logp << '\n';
    }
  }
}
