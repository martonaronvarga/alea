#include <iomanip>
#include <iostream>
#include <stan/math.hpp>
#include <vector>

using stan::math::var;

struct Row7 {
  double y;
  double a;
  double t0;
  double w;
  double v;
  double sv;
  double sw;
  double st0;
};

static void print_full_case(std::size_t i, const Row7 &r) {
  var y = r.y;
  var a = r.a;
  var t0 = r.t0;
  var w = r.w;
  var v = r.v;
  var sv = r.sv;
  var sw = r.sw;
  var st0 = r.st0;

  auto lp = stan::math::wiener_lpdf(y, a, t0, w, v, sv, sw, st0);
  lp.grad();

  std::cout << "i=" << i << " lp=" << lp.val() << " gy=" << y.adj()
            << " ga=" << a.adj() << " gt0=" << t0.adj() << " gw=" << w.adj()
            << " gv=" << v.adj() << " gsv=" << sv.adj() << " gsw=" << sw.adj()
            << " gst0=" << st0.adj() << "\n";
}

static void print_5p_case(const char *label, double y, double a, double t0,
                          double w, double v, double sv) {
  var yv = y;
  var av = a;
  var t0v = t0;
  var wv = w;
  var vv = v;
  var svv = sv;

  // True 5-parameter comparison: sw = 0, st0 = 0.
  auto lp = stan::math::wiener_lpdf(yv, av, t0v, wv, vv, svv, 0.0, 0.0);
  lp.grad();

  std::cout << label << " lp=" << lp.val() << " gy=" << yv.adj()
            << " ga=" << av.adj() << " gt0=" << t0v.adj() << " gw=" << wv.adj()
            << " gv=" << vv.adj() << " gsv=" << svv.adj() << "\n";
}

int main() {
  std::cout << std::setprecision(17);

  std::vector<Row7> rows = {
      {2.0, 2.0, 1e-9, 0.10, 2.0, 0.0, 0.00, 0.000},
      {3.0, 2.0, 0.01, 0.50, 2.0, 0.2, 0.00, 0.000},
      {4.0, 10.0, 0.01, 0.80, 4.0, 0.0, 0.10, 0.000},
      {5.0, 4.0, 0.01, 0.70, 3.0, 0.0, 0.00, 0.007},
      {6.0, 10.0, 0.01, 0.10, -3.0, 0.2, 0.10, 0.000},
      {7.0, 1.0, 0.01, 0.90, 1.0, 0.2, 0.00, 0.007},
      {8.0, 3.0, 0.01, 0.70, -1.0, 0.0, 0.10, 0.007},
      {8.85, 1.7, 0.01, 0.92, -7.3, 0.7, 0.01, 0.009},
      {8.9, 2.4, 0.01, 0.90, -4.9, 0.0, 0.00, 0.009},
      {9.0, 11.0, 0.01, 0.12, 4.5, 0.7, 0.10, 0.009},
      {1.0, 1.5, 0.10, 0.50, 3.0, 0.5, 0.20, 0.000},
  };

  for (std::size_t i = 0; i < rows.size(); ++i) {
    print_full_case(i, rows[i]);
  }

  // True 5-parameter oracle for the failing row:
  // This is the one to compare against your Rust Wiener5 implementation.
  print_5p_case("five_param_oracle",
                6.0,  // y
                10.0, // a
                0.01, // t0
                0.10, // w
                -3.0, // v
                0.2); // sv

  return 0;
}
