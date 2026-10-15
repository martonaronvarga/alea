#include <cmath>
#include <iomanip>
#include <iostream>
#include <stan/math.hpp>

using stan::math::var;

template <typename F> double central_diff(F &&f, double x, double h) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

double lp5_double(double y, double a, double t0, double w, double v,
                  double sv) {
  return stan::math::wiener_lpdf(y, a, t0, w, v, sv);
}

double lp_full_double(double y, double a, double t0, double w, double v,
                      double sv, double sw, double st0) {
  return stan::math::wiener_lpdf(y, a, t0, w, v, sv, sw, st0);
}

void ad_5param(double y, double a, double t0, double w, double v, double sv) {
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
            << " gt0=" << t0v.adj() << " gw=" << wv.adj() << " gv=" << vv.adj()
            << " gsv=" << svv.adj() << "\n";

  stan::math::recover_memory();
}

void ad_full(double y, double a, double t0, double w, double v, double sv,
             double sw, double st0) {
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
            << " gt0=" << t0v.adj() << " gw=" << wv.adj() << " gv=" << vv.adj()
            << " gsv=" << svv.adj() << " gsw=" << swv.adj()
            << " gst0=" << st0v.adj() << "\n";

  stan::math::recover_memory();
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

  std::cout << "=== five parameter: sv = 0.2 ===\n";
  ad_5param(y, a, t0, w, v, sv);

  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    const double fd = central_diff(
        [&](double ww) { return lp5_double(y, a, t0, ww, v, sv); }, w, h);
    std::cout << "FD_5PARAM h=" << h << " gw_fd=" << fd << "\n";
  }

  for (double ww : {0.08, 0.09, 0.10, 0.11, 0.12}) {
    std::cout << "GRID_5PARAM w=" << ww
              << " lp=" << lp5_double(y, a, t0, ww, v, sv) << "\n";
  }

  std::cout << "\n=== five parameter: sv = 0 control ===\n";
  ad_5param(y, a, t0, w, v, 0.0);

  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    const double fd = central_diff(
        [&](double ww) { return lp5_double(y, a, t0, ww, v, 0.0); }, w, h);
    std::cout << "FD_5PARAM_SV0 h=" << h << " gw_fd=" << fd << "\n";
  }

  std::cout << "\n=== full parameter: sw = 0.1, st0 = 0 ===\n";
  ad_full(y, a, t0, w, v, sv, sw, st0);

  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    const double fd = central_diff(
        [&](double ww) { return lp_full_double(y, a, t0, ww, v, sv, sw, st0); },
        w, h);
    std::cout << "FD_FULL h=" << h << " gw_fd=" << fd << "\n";
  }

  for (double ww : {0.08, 0.09, 0.10, 0.11, 0.12}) {
    std::cout << "GRID_FULL w=" << ww
              << " lp=" << lp_full_double(y, a, t0, ww, v, sv, sw, st0) << "\n";
  }

  return 0;
}
