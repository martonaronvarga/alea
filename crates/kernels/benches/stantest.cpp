#include <iomanip>
#include <iostream>
#include <stan/math.hpp>
#include <vector>

int main() {
  std::cout << std::setprecision(17);

  const double y = 0.8;
  const double a = 1.5;
  const double t0 = 0.3;
  const double v = 0.4;
  const double sv = 0.1;
  const double sw = 0.05;
  const double st0 = 0.15;

  for (double w : {0.45, 0.5, 0.55, 0.6, 0.65}) {
    stan::math::var yv = y;
    stan::math::var av = a;
    stan::math::var t0v = t0;
    stan::math::var wv = w;
    stan::math::var vv = v;
    stan::math::var svv = sv;
    stan::math::var swv = sw;
    stan::math::var st0v = st0;

    auto lp = stan::math::wiener_lpdf(yv, av, t0v, wv, vv, svv, swv, st0v);
    lp.grad();

    std::cout << "beta=" << w << " lp=" << lp.val() << " dlp/dbeta=" << wv.adj()
              << "\n";
  }

  std::vector<double> y_vec = {2, 3, 4, 5, 6, 7, 8, 8.85, 8.9, 9, 1};
  std::vector<double> a_vec = {2.0, 2.0, 10.0, 4.0,  10.0, 1.0,
                               3.0, 1.7, 2.4,  11.0, 1.5};
  std::vector<double> v_vec = {2.0,  2.0,  4.0,  3.0, -3.0, 1.0,
                               -1.0, -7.3, -4.9, 4.5, 3.0};
  std::vector<double> w_vec = {.1, 0.5, .8, 0.7, .1, .9, .7, .92, .9, .12, 0.5};
  std::vector<double> t0_vec = {1e-9, 0.01, .01, .01, .01, .01,
                                .01,  .01,  .01, .01, 0.1};
  std::vector<double> sv_vec = {0, 0.2, 0, 0, .2, .2, 0, .7, 0, .7, 0.5};
  std::vector<double> sw_vec = {0, 0, .1, 0, .1, 0, .1, .01, 0, .1, 0.2};
  std::vector<double> st0_vec = {0,    0,    0,    0.007, 0, .007,
                                 .007, .009, .009, .009,  0};

  std::vector<double> true_dens = {
      -4.28564747866615, -7.52379235146909, -26.1551056209248,
      -22.1939134892089, -50.0587553794834, -37.2817263586318,
      -10.5428662079438, -61.5915905674246, -117.238967959795,
      -12.5788594249676, -3.1448097740735};

  std::vector<double> true_grad_y = {
      -3.22509339523307, -2.91155058614589, -8.21331631900955,
      -4.82948967379739, -1.50069056428102, -5.25831601347426,
      -1.04831896413742, -2.67457492096193, -12.8617364931501,
      -1.12047317491985, -5.68799957241344};

  std::vector<double> true_grad_a = {
      3.25018678924105,  3.59980430191399,  0.876602303160642, 1.2215517888504,
      -3.02928674030948, 67.0322498959921,  1.95334514374631,  16.4642201959135,
      5.02038145619773,  0.688439187670968, 2.63200041459657};

  std::vector<double> true_grad_t0 = {
      3.22509339523307, 2.91155058614589, 8.21331631900955, 4.82948967379739,
      1.50069056428102, 5.25831601347426, 1.04831896413742, 2.67457492096193,
      12.8617364931501, 1.12047317491985, 5.68799957241344};

  std::vector<double> true_grad_w = {
      5.67120184517318,  -3.64396221090076, -38.7775057146792,
      -14.1837930137393, -34.5869239580708, -10.4535345681946,
      0.679597983582904, -9.93144540834201, 2.09117200953597,
      -6.0858540417876,  -3.74870310978083};

  std::vector<double> true_grad_v = {
      -2.199999998,     -4.44801714898178, -13.6940602985224, -13.7593709622169,
      21.5540563802381, -5.38233555673517, 8.88475440789056,  12.1280680728793,
      43.7785246930371, -5.68143495684294, -1.57639220567218};

  std::vector<double> true_grad_sv = {0,
                                      3.42285198319565,
                                      0,
                                      0,
                                      91.9551438876654,
                                      4.70180879974639,
                                      0,
                                      101.80250964211,
                                      0,
                                      21.4332628706595,
                                      0.877556017134384};

  std::vector<double> true_grad_sw = {0,
                                      0,
                                      10.1052188867058,
                                      0,
                                      8.72398,
                                      0,
                                      -0.122807217815892,
                                      -0.0506322723373748,
                                      0,
                                      -0.0704990526706635,
                                      0.0827817310725268};

  std::vector<double> true_grad_st0 = {0,
                                       0,
                                       0,
                                       2.42836139121338,
                                       0,
                                       2.64529825657625,
                                       0.524800556172613,
                                       1.34278261179603,
                                       6.55490874737353,
                                       0.561295838843035,
                                       0};

  const double err_tol_dens = 1e-6;
  const double err_tol = 1e-4;

  std::cout << std::setprecision(17);

  for (std::size_t i = 0; i < y_vec.size(); ++i) {
    stan::math::var y = y_vec[i];
    stan::math::var a = a_vec[i];
    stan::math::var t0 = t0_vec[i];
    stan::math::var w = w_vec[i];
    stan::math::var v = v_vec[i];
    stan::math::var sv = sv_vec[i];
    stan::math::var sw = sw_vec[i];
    stan::math::var st0 = st0_vec[i];

    auto dens = stan::math::wiener_lpdf(y, a, t0, w, v, sv, sw, st0);
    dens.grad();

    std::cout << "i=" << i << " lp=" << dens.val() << " gy=" << y.adj()
              << " ga=" << a.adj() << " gt0=" << t0.adj() << " gw=" << w.adj()
              << " gv=" << v.adj() << " gsv=" << sv.adj() << " gsw=" << sw.adj()
              << " gst0=" << st0.adj() << "\n";
  }
}
