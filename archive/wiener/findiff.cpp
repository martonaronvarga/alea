#include <cmath>
#include <iomanip>
#include <iostream>
#include <stan/math.hpp>

using stan::math::var;

double lp_full_double(double y, double a, double t0, double w, double v,
                      double sv, double sw, double st0) {
  return stan::math::wiener_lpdf(y, a, t0, w, v, sv, sw, st0);
}

template <typename F> double fd_central(F f, double x, double h) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

int main() {
  std::cout << std::setprecision(17);

  const double y = 6.0;
  const double a = 10.0;
  const double t0 = 0.01;
  const double w = 0.1;
  const double v = -3.0;
  const double sv = 0.2;
  const double sw = 0.1;
  const double st0 = 0.0;

  {
    var yv = y;
    var av = a;
    var t0v = t0;
    var wv = w;
    var vv = v;
    var svv = sv;
    var swv = sw;
    var st0v = st0;

    auto lp = stan::math::wiener_lpdf(yv, av, t0v, wv, vv, svv, swv, st0v);
    lp.grad();

    std::cout << "AD_FULL"
              << " lp=" << lp.val() << " gy=" << yv.adj() << " ga=" << av.adj()
              << " gt0=" << t0v.adj() << " gw=" << wv.adj()
              << " gv=" << vv.adj() << " gsv=" << svv.adj()
              << " gsw=" << swv.adj() << " gst0=" << st0v.adj() << "\n";
  }

  stan::math::recover_memory();

  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    const double fdw = fd_central(
        [&](double ww) { return lp_full_double(y, a, t0, ww, v, sv, sw, st0); },
        w, h);

    std::cout << "FD_FULL h=" << h << " gw_fd=" << fdw << "\n";
  }

  for (double ww : {0.08, 0.09, 0.10, 0.11, 0.12}) {
    std::cout << "GRID_FULL w=" << ww
              << " lp=" << lp_full_double(y, a, t0, ww, v, sv, sw, st0) << "\n";
  }

  return 0;
}
