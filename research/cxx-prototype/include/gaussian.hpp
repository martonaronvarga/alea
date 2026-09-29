#pragma once
#include "rust/cxx.h"
#include <cstdint>

double gaussian_log_prob(rust::Slice<const double> x);
void gaussian_grad(rust::Slice<const double> x, rust::Slice<double> grad);
