data { int<lower=1> N; int<lower=1> K; matrix[N,K] X; array[N] int<lower=0,upper=1> y; }
parameters { vector[K] beta; }
model {
  profile("posterior") {
    beta ~ normal(0,2);
    target += bernoulli_logit_glm_lpmf(y | X, 0, beta);
  }
}
