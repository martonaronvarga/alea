#include <chrono>
#include <iostream>
#include <stan/math.hpp>
#include <vector>

int main() {
  using stan::math::var;
  using stan::math::wiener_lpdf;

  // Constrained parameter values (same as Rust benchmark)
  double y = 0.8;
  double a = 1.5;
  double t0 = 0.3;
  double w = 0.55;
  double v = 0.4;
  double sv = 0.1;
  double sw = 0.05;
  double st0 = 0.15;

  // Generic functor – works with both double and var Eigen vectors
  auto log_prob_fun = [&y](const auto &c) {
    // c can be Eigen::Matrix<double,-1,1> or Eigen::Matrix<var,-1,1>
    auto a_var = c(0);
    auto t0_var = c(1);
    auto w_var = c(2);
    auto v_var = c(3);
    auto sv_var = c(4);
    auto sw_var = c(5);
    auto st0_var = c(6);
    return wiener_lpdf(y, a_var, t0_var, w_var, v_var, sv_var, sw_var, st0_var);
  };

  // Parameter values as double Eigen vector
  Eigen::VectorXd params_double(7);
  params_double << a, t0, w, v, sv, sw, st0;

  // Warm-up (gradient path)
  std::cout << "=== Stan C++ Wiener7 example ===" << std::endl;
  {
    double fx;
    Eigen::VectorXd grad(7);
    stan::math::gradient(log_prob_fun, params_double, fx, grad);
    std::cout << std::setprecision(10);
    std::cout << "log_prob: " << fx << std::endl;
    std::cout << "grad: [";
    for (int i = 0; i < 7; ++i) {
      std::cout << (i > 0 ? ", " : "") << grad(i);
    }
    std::cout << "]" << std::endl;

    // Also print in Rust-compatible format
    std::cout << "\nStan C++ log_prob: " << fx << std::endl;
    std::cout << "Stan C++ grad: [";
    for (int i = 0; i < 7; ++i) {
      std::cout << grad(i);
      if (i < 6)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    double y5 = 0.8;
    double a5 = 1.5, t05 = 0.3, w5 = 0.55, v5 = 0.4, sv5 = 0.1;
    double lp5 = stan::math::wiener_lpdf(y5, a5, t05, w5, v5, sv5);
    std::cout << "Stan Wiener5 log_prob: " << lp5 << std::endl;
  }

  const int n_reps = 500;
  double lp_time, grad_time;

  // Benchmark log_prob only (using double vector)
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_reps; ++i) {
      // We can directly capture the result.
      auto lp = log_prob_fun(params_double);
      // If it returns var, call .val(); for double, it's just double.
      volatile double sink = lp;
      (void)sink;
    }
    auto end = std::chrono::high_resolution_clock::now();
    lp_time =
        std::chrono::duration<double, std::micro>(end - start).count() / n_reps;
  }

  // Benchmark full gradient
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_reps; ++i) {
      double fx;
      Eigen::VectorXd grad(7);
      stan::math::gradient(log_prob_fun, params_double, fx, grad);
      volatile double sink = grad(0);
      (void)sink;
    }
    auto end = std::chrono::high_resolution_clock::now();
    grad_time =
        std::chrono::duration<double, std::micro>(end - start).count() / n_reps;
  }

  std::cout << "\n=== Stan C++ Benchmarks ===" << std::endl;
  std::cout << "Stan C++ log_prob: " << lp_time << " µs/call\n";
  std::cout << "Stan C++ grad    : " << grad_time << " µs/call\n";
  return 0;
}
