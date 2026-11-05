data { int<lower=1> N; int<lower=1> K; matrix[N,K] X; vector[N] y; }
transformed data {
  array[N] vector[2] observations;
  for (i in 1:N) observations[i]=X[i]';
}
parameters { real mu0; real mu1; real<lower=0> s0; real<lower=0> s1; real loading; }
model {
  profile("posterior") {
    matrix[2,2] L;
    vector[2] mu = [mu0,mu1]';
    L[1,1]=s0; L[1,2]=0; L[2,1]=loading; L[2,2]=s1;
    mu ~ normal(0,2);
    s0 ~ std_normal(); s1 ~ std_normal(); loading ~ std_normal();
    observations ~ multi_normal_cholesky(mu,L);
  }
}
