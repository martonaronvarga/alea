// Same posterior and ordering as the existing PosteriorDB noncentered port.
data { int<lower=1> N; int<lower=1> K; matrix[N,K] X; vector[N] y; }
parameters { vector[N] effect; real mu; real<lower=0> tau; }
model {
  profile("posterior") {
    mu ~ normal(0,5);
    tau ~ cauchy(0,5);
    effect ~ std_normal();
    y ~ normal(mu+tau*effect,X[,1]);
  }
}
