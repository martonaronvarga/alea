data { int<lower=1> K; }
parameters {
  real v;
  vector[K] z;
}
model {
  profile("posterior") {
  v ~ normal(0, 3);
  z ~ normal(0, exp(v / 2));
  }
}
