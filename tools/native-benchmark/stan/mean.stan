data {
  int<lower=1> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  profile("posterior") {
  mu ~ normal(0, 2);
  sigma ~ normal(0, 1);  // half-normal prior under the positivity constraint
  y ~ normal(mu, sigma);
  }
}
