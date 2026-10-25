data {
  int<lower=0,upper=11> kind;
  int<lower=1> D;
  int<lower=0> N;
  int<lower=0> K;
  vector[K] x;
}
parameters { vector[D] q; }
model {
  if (kind == 0) {
    for (i in 1:D) target += -0.5*square(q[i]/x[i]);
  } else if (kind == 1) {
    target += -0.5*square(q[1]);
    for (i in 2:D) target += -0.5*square(q[i]-x[1]*q[i-1])/(1-square(x[1]));
  } else if (kind == 2) {
    for (i in 1:(D%/%2)) target += -0.5*(square(q[2*i-1])+square(q[2*i]-x[1]*(square(q[2*i-1])-1)));
  } else if (kind == 3) {
    target += -0.5*square(q[1]/x[1])-0.5*(D-1)*q[1]-0.5*dot_self(q[2:D])*exp(-q[1]);
  } else if (kind == 4) {
    target += -0.5*dot_self(q);
    for (n in 1:N) {
      real eta=0;
      for (i in 1:D) eta+=q[i]*x[(n-1)*(D+1)+i];
      target += x[n*(D+1)]*eta-log1p_exp(eta);
    }
  } else if (kind == 5 || kind == 9) {
    real tau=exp(q[2]);
    target += -0.5*square(q[1]/0.7)-0.5*square((q[2]+0.7)/0.3)-0.5*dot_self(q[3:4]);
    for (n in 1:N) {
      int group=x[2*n-1]==0 ? 3 : 4;
      real eta=q[1]+tau*q[group];
      if (kind == 5) target += -0.5*square((x[2*n]-eta)/0.7);
      else target += x[2*n]*1.2*eta-log1p_exp(1.2*eta);
    }
  } else if (kind == 6) {
    target += -0.5*square(q[1])*(1-square(x[1]))/square(x[2]);
    for (i in 2:D) target += -0.5*square(q[i]-x[1]*q[i-1])/square(x[2]);
    for (i in 1:D) target += -0.5*(q[i]+square(x[i+2])*exp(-q[i]));
  } else if (kind == 7) {
    for (i in 1:D) target += -0.5*square(q[i])-2*square(x[i]-q[i]);
  } else if (kind == 8) {
    target += -0.5*square(q[1]);
    for (n in 1:N) {
      // Symmetric start: lower events reflect drift, no RT sign convention.
      target += wiener_lpdf(x[3*n-2] | 1.2, 0.15, 0.5, x[3*n-1]*q[1]);
    }
  } else {
    real tau=exp(q[2]);
    target += -0.5*square(q[1]/5)+q[2]-log1p(square(tau/5));
    for (i in 1:8) {
      real theta;
      if (kind == 10) {
        theta=q[i+2];
        target += -q[2]-0.5*square((theta-q[1])/tau);
      } else {
        theta=q[1]+tau*q[i+2];
        target += -0.5*square(q[i+2]);
      }
      target += -0.5*square((x[2*i-1]-theta)/x[2*i]);
    }
  }
}
