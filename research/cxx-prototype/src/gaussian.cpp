#include "gaussian.hpp"

double gaussian_log_prob(rust::Slice<const double> x) {
  const double *ptr = x.data();
  size_t n = x.size();

  double s = 0.0;
  for (size_t i = 0; i < n; i++) {
    s -= 0.5 * ptr[i] * ptr[i];
  }
  return s;
}

void gaussian_grad(rust::Slice<const double> x, rust::Slice<double> grad) {
  const double *xp = x.data();
  double *gp = grad.data();
  size_t n = x.size();

  for (size_t i = 0; i < n; i++) {
    gp[i] = -xp[i];
  }
}
