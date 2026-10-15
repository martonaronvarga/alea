#include <cmath>
#include <iomanip>
#include <iostream>
#include <stan/math.hpp>

using stan::math::var;

double lp5_double(double y, double a, double t0, double w, double v,
                  double sv) {
  return stan::math::wiener_lpdf(y, a, t0, w, v, sv);
}

double lp7zero_double(double y, double a, double t0, double w, double v,
                      double sv) {
  return stan::math::wiener_lpdf(y, a, t0, w, v, sv, 0.0, 0.0);
}

template <typename F> double fd_central(F f, double x, double h) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

void run_case(double y, double a, double t0, double w, double v, double sv) {
  std::cout << std::setprecision(17);

  {
    var yv = y;
    var av = a;
    var t0v = t0;
    var wv = w;
    var vv = v;
    var svv = sv;

    auto lp = stan::math::wiener_lpdf(yv, av, t0v, wv, vv, svv);
    lp.grad();

    std::cout << "AD_5PARAM"
              << " lp=" << lp.val() << " gy=" << yv.adj() << " ga=" << av.adj()
              << " gt0=" << t0v.adj() << " gw=" << wv.adj()
              << " gv=" << vv.adj() << " gsv=" << svv.adj() << "\n";
  }

  stan::math::recover_memory();

  {
    var yv = y;
    var av = a;
    var t0v = t0;
    var wv = w;
    var vv = v;
    var svv = sv;
    var swv = 0.0;
    var st0v = 0.0;

    auto lp = stan::math::wiener_lpdf(yv, av, t0v, wv, vv, svv, swv, st0v);
    lp.grad();

    std::cout << "AD_7ZERO"
              << " lp=" << lp.val() << " gy=" << yv.adj() << " ga=" << av.adj()
              << " gt0=" << t0v.adj() << " gw=" << wv.adj()
              << " gv=" << vv.adj() << " gsv=" << svv.adj()
              << " gsw=" << swv.adj() << " gst0=" << st0v.adj() << "\n";
  }

  stan::math::recover_memory();

  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    const double fd5 = fd_central(
        [&](double ww) { return lp5_double(y, a, t0, ww, v, sv); }, w, h);

    const double fd7z = fd_central(
        [&](double ww) { return lp7zero_double(y, a, t0, ww, v, sv); }, w, h);

    std::cout << "FD h=" << h << " fd5=" << fd5 << " fd7zero=" << fd7z << "\n";
  }

  std::cout << "\n";
}

int main() {
  std::cout << std::setprecision(17);

  std::cout << "=== i4 sv=0.2 ===\n";
  run_case(6.0,  // y
           10.0, // a
           0.01, // t0
           0.1,  // w
           -3.0, // v
           0.2   // sv
  );

  std::cout << "=== same case but sv=0.0 control ===\n";
  run_case(6.0, 10.0, 0.01, 0.1, -3.0, 0.0);

  std::cout << "=== w grid, 5-param values only ===\n";
  for (double w :
       {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90}) {
    const double lp5 = lp5_double(6.0, 10.0, 0.01, w, -3.0, 0.2);
    const double lp7z = lp7zero_double(6.0, 10.0, 0.01, w, -3.0, 0.2);

    std::cout << "w=" << w << " lp5=" << lp5 << " lp7zero=" << lp7z << "\n";
  }

  return 0;
}
