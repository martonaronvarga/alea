fn main() {
    if std::env::args().any(|arg| arg == "--updates") {
        nuts_rs::AleaReference::emit_alea_updates();
    } else {
        nuts_rs::AleaReference::emit_alea_reference();
    }
}
