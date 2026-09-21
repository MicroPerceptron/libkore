#![feature(type_alias_impl_trait)]
use sdk::prelude::*;
use system as sdk;

sdk::exit_codes! {
    /// A program-specific document failure, outside the standard outcomes.
    const BAD_DOCUMENT = ExitCode::APPLICATION_0;
}

#[derive(sdk::Bundle)]
#[bundle(name = "empty-standard-app")]
pub struct Empty;

#[sdk::task]
pub async fn worker() {}

#[sdk::spmd(1, 1, 1)]
pub fn parallel(_ctx: &'static sdk::runtime::spmd::SpmdCtx, _worker: u32) {}

#[sdk::main]
pub async fn entry(_rt: Rt) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[test]
fn facade_macros_need_no_direct_runtime_or_abi_dependencies() {
    use sdk::runtime::service::Bundle;
    assert!(Empty::WIRING.is_empty());
    assert_eq!(BAD_DOCUMENT.get(), 256);
    assert!(!BAD_DOCUMENT.is_ok());
    let first = worker().expect("one task slot");
    assert!(
        worker().is_err(),
        "same task storage cannot have two owners"
    );
    drop(first);
    let _ = parallel;
}
