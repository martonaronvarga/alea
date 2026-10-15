data { int<lower=0,upper=3> which; }
parameters { vector[2] q; }
model {
  if (which == 0) {
    target += -0.5 * (square(q[1]) + square((q[2]-0.6*q[1])/0.8));
  } else if (which == 1) {
    target += -0.5 * (square(q[1]) + square(q[2]-0.4*(square(q[1])-1)));
  } else if (which == 2) {
    array[4] real x = {-1.5,-0.2,0.7,2.0};
    array[4] int y = {0,1,0,1};
    target += -0.5 * dot_self(q);
    for (i in 1:4) target += bernoulli_logit_lpmf(y[i] | q[1]+x[i]*q[2]);
  } else {
    target += -0.5*square(q[1])-0.5*q[1]-0.5*square(q[2])*exp(-q[1]);
  }
}
