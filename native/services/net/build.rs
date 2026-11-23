//! Rebuild this service when its network configuration changes.
//!
//! `KORE_WIFI_SSID` and `KORE_WIFI_PSK` reach `option_env!` at compile
//! time (`src/network.rs`). Without these declarations a change to either
//! — same sources, different network — would reuse the stale object and
//! produce an image that joins the previous network while the build log
//! says otherwise. Runs on the host; the metal target never sees it.

fn main() {
    println!("cargo:rerun-if-env-changed=KORE_WIFI_SSID");
    println!("cargo:rerun-if-env-changed=KORE_WIFI_PSK");
}
