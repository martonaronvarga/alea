//! Thread-bound context restoration. No subscriber or global state is installed.
pub(crate) struct Context {
    dispatch: tracing::Dispatch,
    parent: tracing::Span,
}
impl Context {
    pub(crate) fn capture() -> Self {
        Self {
            dispatch: tracing::dispatcher::get_default(Clone::clone),
            parent: tracing::Span::current(),
        }
    }
    pub(crate) fn enter(&self) -> Scope<'_> {
        let dispatch = tracing::dispatcher::set_default(&self.dispatch);
        let span = self.parent.enter();
        Scope {
            _span: span,
            _dispatch: dispatch,
        }
    }
}
// Fields drop in declaration order: leave the parent before restoring dispatch.
// Both guards stay on the executing thread, including unwinding.
pub(crate) struct Scope<'a> {
    _span: tracing::span::Entered<'a>,
    _dispatch: tracing::dispatcher::DefaultGuard,
}
