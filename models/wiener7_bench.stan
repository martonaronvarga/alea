data {
  real y;
  real a;
  real t0;
  real w;
  real v;
  real sv;
  real sw;
  real st0;
}
parameters {
  // dummy – nothing to sample
}
model { y ~wiener_full(a, t0, w, v, sv, sw, st0); }
