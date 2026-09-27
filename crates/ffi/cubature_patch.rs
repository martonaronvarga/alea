// Local safety fixes for cubature 4ba70a7d595a00cfb15b02763013e0088013adb7.
// C fragments derive from hcubature.c, copyright Steven G. Johnson et al.,
// GPL-2.0-or-later; retain the complete upstream notice in the generated file.
// Do not modify the submodule. Exact-match guards require review on upstream changes.

fn replace(source: &mut String, before: &str, after: &str) {
    assert_eq!(
        source.matches(before).count(),
        1,
        "cubature patch needs upstream review: {before}"
    );
    *source = source.replacen(before, after, 1);
}

pub fn checked_source(mut source: String) -> String {
    // Preserve the old heap on allocation failure, so cleanup still owns it.
    replace(
        &mut source,
        "     h->nalloc = nalloc;\n     if (nalloc)\n         h->items = (heap_item *) realloc(h->items, sizeof(heap_item)*nalloc);",
        "     if (nalloc) {\n         heap_item *items;\n         if (nalloc > ((size_t)-1) / sizeof(heap_item)) return;\n         items = (heap_item *) realloc(h->items, sizeof(heap_item)*nalloc);\n         if (!items) return;\n         h->items = items;\n     }",
    );
    replace(
        &mut source,
        "         h->items = NULL;\n     }\n}\n\nstatic heap heap_alloc",
        "         h->items = NULL;\n     }\n     h->nalloc = nalloc;\n}\n\nstatic heap heap_alloc",
    );
    // A failed push must not increment n or transfer ownership. Use size_t heap
    // indices rather than narrowing to int (including the child index in pop).
    replace(&mut source, "     int insert;", "     size_t insert;");
    replace(
        &mut source,
        "     insert = h->n;",
        "     if (h->n >= UINT_MAX) return FAILURE; /* unsigned region iteration */\n     insert = h->n;",
    );
    replace(
        &mut source,
        "     if (++(h->n) > h->nalloc) {\n\t  heap_resize(h, h->n * 2);\n\t  if (!h->items) return FAILURE;\n     }",
        "     if (h->n == h->nalloc) {\n         if (h->n > ((size_t)-1) / 2 - 1) return FAILURE;\n         heap_resize(h, (h->n + 1) * 2);\n         if (h->n == h->nalloc) return FAILURE;\n     }\n     ++h->n;",
    );
    replace(
        &mut source,
        "\t  int parent = (insert - 1) / 2;",
        "\t  size_t parent = (insert - 1) / 2;",
    );
    replace(
        &mut source,
        "     int i, n, child;",
        "     size_t i, n, child;",
    );
    replace(&mut source, "\t  int largest;", "\t  size_t largest;");
    // Region arrays own their contents until a successful heap transfer. Clear
    // transferred slots so every failure path can destroy all slots exactly once.
    replace(
        &mut source,
        "     for (i = 0; i < ni; ++i)\n\t  if (heap_push(h, hi[i])) return FAILURE;",
        "     for (i = 0; i < ni; ++i) {\n         if (heap_push(h, hi[i])) return FAILURE;\n         memset(hi + i, 0, sizeof(*hi));\n     }",
    );
    replace(
        &mut source,
        "     R2->h = make_hypercube(dim, R->h.data, R->h.data + dim);",
        "     R2->ee = NULL; /* not an owner of R->ee after the shallow copy */\n     R2->h = make_hypercube(dim, R->h.data, R->h.data + dim);",
    );
    replace(
        &mut source,
        "     R = (region *) malloc(sizeof(region) * nR_alloc);",
        "     R = (region *) calloc(nR_alloc, sizeof(region));",
    );
    replace(
        &mut source,
        "     numEval += r->num_points;",
        "     memset(R, 0, sizeof(*R)); /* ownership transferred to heap */\n     numEval += r->num_points;",
    );
    // The Rust API uses serial hcubature, but preserve cleanup for the compiled
    // vector entry point too. New slots must be zeroed and failed realloc retained.
    replace(
        &mut source,
        "\t\t\t nR_alloc = (nR + 2) * 2;\n\t\t\t R = (region *) realloc(R, nR_alloc * sizeof(region));\n\t\t\t if (!R) goto bad;",
        "                         size_t next;\n                         region *grown;\n                         if (nR > ((size_t)-1) / (2 * sizeof(region)) - 2) goto bad;\n                         next = (nR + 2) * 2;\n                         grown = (region *) realloc(R, next * sizeof(region));\n                         if (!grown) goto bad;\n                         R = grown;\n                         memset(R + nR_alloc, 0, (next - nR_alloc) * sizeof(region));\n                         nR_alloc = next;",
    );
    replace(
        &mut source,
        "bad:\n     free(ee);\n     heap_free(&regions);\n     free(R);",
        "bad:\n     if (R) {\n         size_t pending;\n         for (pending = 0; pending < nR_alloc; ++pending) destroy_region(R + pending);\n     }\n     {\n         size_t retained;\n         for (retained = 0; retained < regions.n; ++retained) destroy_region(regions.items + retained);\n     }\n     free(ee);\n     heap_free(&regions);\n     free(R);",
    );
    source
}
