#include <stdint.h>
#include <openblas_config.h>

/* cblas 0.4 declares dimensions and strides as Rust i32. */
_Static_assert(sizeof(blasint) == sizeof(int32_t),
               "Alea requires LP64 OpenBLAS (32-bit BLAS integers), not ILP64");
