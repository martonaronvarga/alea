#pragma once
#include <vector>

double log_prob(const std::vector<double> &x);
void grad_log_prob(const std::vector<double> &x, std::vector<double> &grad);
