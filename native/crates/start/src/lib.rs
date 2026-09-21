//! Rust compiler entry adapter. Link this only into returning executables,
//! never into services that already own `__kore_rt_main` or a panic handler.
//! The application-facing library composes this adapter with its public facade.
#![no_std]
#![cfg_attr(target_os = "none", feature(lang_items))]
#![cfg_attr(target_os = "none", allow(internal_features))]

pub use core::*;
pub use kore_rt as runtime;
pub use kore_rt::{ExitCode, Termination};

#[cfg(target_os = "none")]
mod native {
    use super::*;

    // A local language trait avoids activating language items in kore-rt:
    // Cargo feature unification must not change existing service entry points.
    #[lang = "termination"]
    pub trait MainReturn {
        fn outcome(self) -> ExitCode;
    }
    impl<T: Termination> MainReturn for T {
        fn outcome(self) -> ExitCode {
            self.report()
        }
    }

    #[lang = "start"]
    fn start<T: MainReturn + 'static>(
        main: fn() -> T,
        _: isize,
        _: *const *const u8,
        _: u8,
    ) -> isize {
        kore_rt::startup::finish(main().outcome())
    }

    #[unsafe(no_mangle)]
    pub fn __kore_rt_main(_: kore_rt::entry::Rt) -> ! {
        unsafe extern "C" {
            fn main(argc: core::ffi::c_int, argv: *const *const u8) -> core::ffi::c_int;
        }
        // rustc's generated C shim calls the language start hook above. Kore
        // arguments remain available through the capability-backed process page.
        unsafe {
            main(0, core::ptr::null());
        }
        kore_rt::startup::finish(ExitCode::INTERNAL)
    }

    #[panic_handler]
    fn panic(info: &core::panic::PanicInfo) -> ! {
        // Abort: no unwinding or user callbacks after a panic. The launcher
        // reaps the process through the same kernel worker barrier.
        kore_rt::crash::panic(info)
    }
}
