/* Native ownership/failure regression for the exact C source built by Cargo.
 * This harness includes GPL-2.0-or-later cubature; see its upstream notice.
 * Every C allocation is counted, and each allocation can fail deterministically.
 */
#include <assert.h>
#include <stdlib.h>
#include <stddef.h>

static size_t live_allocations, allocation_calls, fail_at;

static int should_fail(void) { return ++allocation_calls == fail_at; }
static void *checked_malloc(size_t n) {
    void *p;
    if (should_fail()) return NULL;
    p = malloc(n);
    if (p) ++live_allocations;
    return p;
}
static void *checked_calloc(size_t n, size_t size) {
    void *p;
    if (should_fail()) return NULL;
    p = calloc(n, size);
    if (p) ++live_allocations;
    return p;
}
static void *checked_realloc(void *old, size_t n) {
    void *p;
    int was_null = old == NULL;
    assert(n > 0);
    if (should_fail()) return NULL;
    p = realloc(old, n);
    if (p && was_null) ++live_allocations;
    return p;
}
static void checked_free(void *p) {
    if (p) { assert(live_allocations > 0); --live_allocations; }
    free(p);
}

#define malloc checked_malloc
#define calloc checked_calloc
#define realloc checked_realloc
#define free checked_free
#include "hcubature_checked.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

static size_t callback_calls, fail_callback;
static int integrand_test(unsigned dim, const double *x, void *data,
                          unsigned fdim, double *out) {
    unsigned j;
    (void)data;
    assert(dim >= 1 && fdim == 2);
    if (++callback_calls == fail_callback) return 1;
    /* Deliberately non-polynomial, requiring refinement and heap growth. */
    for (j = 0; j < fdim; ++j)
        out[j] = exp(x[0]) + (x[dim - 1] > 0.37 ? 1.0 : -1.0);
    return 0;
}

static size_t run(unsigned dim, size_t allocation_failure, size_t callback_failure) {
    double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1}, val[2], err[2];
    int result;
    assert(live_allocations == 0);
    allocation_calls = callback_calls = 0;
    fail_at = allocation_failure;
    fail_callback = callback_failure;
    result = hcubature(2, integrand_test, NULL, dim, lo, hi, 1000,
                       1e-12, 1e-12, ERROR_L2, val, err);
    if ((fail_at && allocation_calls >= fail_at) ||
        (fail_callback && callback_calls >= fail_callback)) assert(result != 0);
    else assert(result == 0);
    assert(live_allocations == 0);
    return allocation_calls;
}

int main(void) {
    unsigned dim;
    for (dim = 1; dim <= 3; ++dim) {
        size_t i, allocations = run(dim, 0, 0);
        for (i = 1; i <= allocations; ++i) run(dim, i, 0);
        /* Initial callback and callbacks after serial region splitting. */
        for (i = 1; i <= 100; ++i) run(dim, 0, i);
    }
    return 0;
}
