//! Build-time safety patches accept Windows checkouts without weakening review.
#[path = "../cubature_patch.rs"]
mod cubature_patch;

#[test]
fn lf_and_crlf_generate_identical_checked_source_without_changing_upstream() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/cubature/hcubature.c");
    let original = std::fs::read_to_string(&path).unwrap();
    let lf = original.replace("\r\n", "\n");
    let crlf = lf.replace('\n', "\r\n");
    let expected = cubature_patch::checked_source(lf);
    assert_eq!(cubature_patch::checked_source(crlf), expected);
    assert!(!expected.contains('\r'));
    assert_eq!(std::fs::read_to_string(path).unwrap(), original);
}

#[test]
fn changed_or_duplicated_patch_sites_still_require_upstream_review() {
    let original = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/cubature/hcubature.c"),
    )
    .unwrap();
    for changed in [
        original.replace("h->nalloc = nalloc", "h->nalloc = other"),
        format!("{original}\n{original}"),
    ] {
        assert!(std::panic::catch_unwind(|| cubature_patch::checked_source(changed)).is_err());
    }
}
