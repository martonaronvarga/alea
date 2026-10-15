data {
  int<lower=0,upper=4> which;
  int<lower=2> D;
  int<lower=0> N;
  matrix[N,D] X;
  array[N] int<lower=0,upper=1> y;
}
parameters { vector[D] q; }
model {
  if (which <= 1) {
    real rho = which == 0 ? 0.95 : 0.9;
    target += -0.5*square(q[1]);
    target += -0.5*dot_self(q[2:D]-rho*q[1:(D-1)])/(1-square(rho));
  } else if (which == 2) {
    for (i in 1:(D%/%2)) {
      target += -0.5*(square(q[2*i-1])+square(q[2*i]-0.8*(square(q[2*i-1])-1)));
    }
  } else if (which == 3) {
    target += -0.5*dot_self(q);
    y ~ bernoulli_logit(X*q);
  } else {
    target += -0.5*square(q[1])/9-0.5*(D-1)*q[1]-0.5*dot_self(q[2:D])*exp(-q[1]);
  }
}
