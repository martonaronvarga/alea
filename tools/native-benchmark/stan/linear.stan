data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  profile("posterior") {
  beta ~ normal(0, 2);
  sigma ~ normal(0, 1);  // half-normal prior under the positivity constraint
  y ~ normal(X * beta, sigma);
  }
}
