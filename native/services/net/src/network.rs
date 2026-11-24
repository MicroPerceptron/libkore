//! The network this image is built to join.
//!
//! Two values baked at compile time: the SSID to associate with and the
//! credential to authenticate with. Both are overridable from the build
//! environment, and both are validated **at compile time** — a build whose
//! network configuration is malformed does not produce an image that
//! discovers the problem on a laptop with no serial port.
//!
//! ```text
//! make flash-x86 KORE_WIFI_SSID=OtherNet KORE_WIFI_PSK=anotherpassphrase
//! ```
//!
//! # Why compile time rather than a file
//!
//! The association rungs (M5, M6 — `res/NETWORK_C2_TO_PING.md` §3) run
//! inside `kore-svc-net`'s founding, before any filesystem this image could
//! read a profile from is mounted and long before a shell exists to type
//! one into. A build-time constant is the only thing available at the point
//! the supplicant needs it. When a persistent profile store exists, this
//! becomes its default rather than its only source.
//!
//! # Nothing here can be used yet
//!
//! The part has running firmware and nothing else: no PHY, no channel, no
//! data rings. [`SSID`] is not read by anything on the M1 path, and
//! [`PASSPHRASE`] will not be read until M6 mints a PTK from it. These
//! constants exist now so the value is settled, validated and auditable
//! before the code that consumes it is written — not because anything
//! consumes it today.
//!
//! # The passphrase is never printed
//!
//! [`SSID`] is fine to render: it is broadcast in the clear by the access
//! point itself. The passphrase is not, and this module offers no way to
//! display it. What it offers instead is [`credential_fingerprint`], four
//! hex digits that say *which* credential is baked without disclosing it —
//! enough to tell two builds apart on a photograph, and useless to anyone
//! reading the photograph.

/// The SSID this image associates with.
///
/// Octets, not a `str`: IEEE 802.11-2020 §9.4.2.2 permits any 0–32 bytes,
/// and `drivers/net/dot11`'s own management code already takes that
/// position. The build environment supplies UTF-8 because a shell variable
/// is text, but nothing downstream may assume it.
pub const SSID: &[u8] = match option_env!("KORE_WIFI_SSID") {
    Some(ssid) => ssid.as_bytes(),
    // **No default here, deliberately.** The network this lab joins is
    // configuration, not program text: it lives in the build, where it can
    // be changed without a source edit and audited in one place. Empty
    // means *unconfigured*, which is a legitimate build — a QEMU image or
    // a host test has no network to join — and the association rungs
    // refuse with a note rather than joining something unintended.
    None => b"",
};

/// The credential for [`SSID`].
///
/// Either an 8..=63 character pass-phrase, which M6 stretches into a PSK
/// with the SSID as salt (IEEE 802.11-2020 Annex J.4.1), or exactly 64 hex
/// characters naming the 256-bit PSK directly (§12.6.1.2). The two are
/// distinguished where they are consumed, not here — a 64-character
/// pass-phrase and a 64-character key are different secrets and stretching
/// the second would silently join a different network.
pub const PASSPHRASE: &[u8] = match option_env!("KORE_WIFI_PSK") {
    Some(psk) => psk.as_bytes(),
    // See [`SSID`]: no credential is written into this file. Empty means
    // unconfigured.
    None => b"",
};

/// Whether this build carries a network to join at all.
///
/// Both halves, or neither: an SSID with no credential and a credential
/// with no SSID are each a half-configured build, and a half-configured
/// build is the one that fails at M6 with a note nobody can read. The
/// const assertion below refuses both.
pub const fn configured() -> bool {
    !SSID.is_empty() && !PASSPHRASE.is_empty()
}

/// Longest SSID the standard permits (IEEE 802.11-2020 §9.4.2.2).
pub const SSID_MAX_BYTES: usize = 32;
/// Shortest pass-phrase the standard permits (Annex J.4.1).
pub const PASSPHRASE_MIN_CHARS: usize = 8;
/// Longest pass-phrase the standard permits (Annex J.4.1).
pub const PASSPHRASE_MAX_CHARS: usize = 63;
/// Characters in a hex-encoded 256-bit PSK (§12.6.1.2).
pub const PSK_HEX_CHARS: usize = 64;

// **Checked here, not at the point of use.** A malformed SSID or
// pass-phrase is a property of the build, and the build is the last place
// a human is watching. Discovering it at M6 would mean a flash, a
// photograph and a note band spent on a typo in a `make` invocation.
const _: () = assert!(
    SSID.is_empty() == PASSPHRASE.is_empty(),
    "set both KORE_WIFI_SSID and KORE_WIFI_PSK, or neither: half a network joins nothing"
);
const _: () = assert!(
    !configured() || SSID.len() <= SSID_MAX_BYTES,
    "KORE_WIFI_SSID must be at most 32 bytes (IEEE 802.11-2020 section 9.4.2.2)"
);
const _: () = assert!(
    !configured()
        || (PASSPHRASE.len() >= PASSPHRASE_MIN_CHARS && PASSPHRASE.len() <= PASSPHRASE_MAX_CHARS)
        || PASSPHRASE.len() == PSK_HEX_CHARS,
    "KORE_WIFI_PSK must be 8..=63 characters, or exactly 64 hex digits for a raw PSK"
);

/// Whether [`PASSPHRASE`] is a raw 256-bit PSK rather than a pass-phrase.
///
/// Exactly 64 characters **and** all of them hex. A 64-character
/// pass-phrase is legal input to nothing — the standard's range stops at
/// 63 — so the length alone would be enough today; the hex test is here so
/// that a future 64-character pass-phrase is rejected by
/// [`credential_is_well_formed`] rather than silently read as a key.
pub const fn credential_is_raw_psk() -> bool {
    if PASSPHRASE.len() != PSK_HEX_CHARS {
        return false;
    }
    let mut i = 0;
    while i < PASSPHRASE.len() {
        if !PASSPHRASE[i].is_ascii_hexdigit() {
            return false;
        }
        i += 1;
    }
    true
}

/// Whether the credential is one of the two forms the standard defines.
///
/// The const assertion above admits a 64-character non-hex string, because
/// a `const` panic cannot say *which* of the two rules it broke. This says
/// it, and the test below turns it into a message.
pub const fn credential_is_well_formed() -> bool {
    (PASSPHRASE.len() >= PASSPHRASE_MIN_CHARS && PASSPHRASE.len() <= PASSPHRASE_MAX_CHARS)
        || credential_is_raw_psk()
}

const _: () = assert!(
    !configured() || credential_is_well_formed(),
    "KORE_WIFI_PSK is 64 characters but not hex: a pass-phrase may be at most 63"
);

/// Four hex digits naming *which* credential this image carries.
///
/// **Not a security property and not presented as one.** It is a 16-bit
/// FNV-1a fold of the pass-phrase, salted with the SSID so the same
/// pass-phrase on two networks reads differently. Its whole job is to let
/// a photograph of a boot distinguish "the image I meant to build" from
/// "the image I built last week", without the credential appearing on a
/// screen, in a photograph, or in a chat log.
///
/// Sixteen bits is deliberate: it is enough to notice a change and far too
/// little to confirm a guess, which is the right trade for a number that
/// will be photographed.
pub const fn credential_fingerprint() -> u16 {
    // FNV-1a, 32-bit, folded to 16. Chosen because it is four lines of
    // `const fn` with no dependency; a cryptographic digest here would
    // imply a guarantee this number does not make.
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < SSID.len() {
        hash ^= SSID[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    // A separator, so `("ab", "cd")` and `("abc", "d")` differ.
    hash ^= 0xff;
    hash = hash.wrapping_mul(0x0100_0193);
    i = 0;
    while i < PASSPHRASE.len() {
        hash ^= PASSPHRASE[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    ((hash >> 16) as u16) ^ (hash as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A build is configured for a network, or it is configured for
    /// none — never for half of one.**
    ///
    /// The source carries no network, so there is no default to pin here.
    /// What there is to check is the invariant the const assertion
    /// enforces: an SSID with no credential, or a credential with no
    /// SSID, is a build that would reach M6 and refuse with a note nobody
    /// can act on.
    #[test]
    fn a_build_carries_a_whole_network_or_none() {
        assert_eq!(
            SSID.is_empty(),
            PASSPHRASE.is_empty(),
            "half-configured: ssid {} bytes, credential {} characters",
            SSID.len(),
            PASSPHRASE.len(),
        );
        assert_eq!(configured(), !SSID.is_empty());
    }

    /// Whatever the build supplied, it is a shape the standard permits.
    ///
    /// The const assertions already refuse a bad build; this one exists so
    /// the failure arrives as a readable test name on a host run rather
    /// than as a `const` panic in the middle of a service compile.
    #[test]
    fn the_configured_credential_is_one_the_standard_defines() {
        // **Only when there is one.** An unconfigured build is legitimate
        // — it is what a bare `cargo test`, a QEMU image and CI all
        // produce — and asserting a shape on an absent network would fail
        // every host run. The half-configured case is the one that is
        // never legitimate, and the test above owns it.
        if !configured() {
            return;
        }
        assert!(SSID.len() <= SSID_MAX_BYTES, "SSID over 32 bytes");
        assert!(
            credential_is_well_formed(),
            "credential is {} characters: a pass-phrase is 8..=63, a raw PSK is 64 hex digits",
            PASSPHRASE.len(),
        );
    }

    /// **The fingerprint changes when either half changes.**
    ///
    /// The whole point is telling two images apart on a photograph, so a
    /// fingerprint that collided on a changed pass-phrase would be worse
    /// than none — it would read as confirmation.
    #[test]
    fn the_fingerprint_separates_the_inputs() {
        const fn fold(ssid: &[u8], psk: &[u8]) -> u16 {
            let mut hash: u32 = 0x811c_9dc5;
            let mut i = 0;
            while i < ssid.len() {
                hash ^= ssid[i] as u32;
                hash = hash.wrapping_mul(0x0100_0193);
                i += 1;
            }
            hash ^= 0xff;
            hash = hash.wrapping_mul(0x0100_0193);
            i = 0;
            while i < psk.len() {
                hash ^= psk[i] as u32;
                hash = hash.wrapping_mul(0x0100_0193);
                i += 1;
            }
            ((hash >> 16) as u16) ^ (hash as u16)
        }

        // The boundary the separator exists for: without it these two
        // would fold identically.
        assert_ne!(fold(b"ab", b"cdefghij"), fold(b"abc", b"defghij"));
        // A one-character change in either half moves it.
        assert_ne!(fold(b"net", b"password"), fold(b"net", b"passworD"));
        assert_ne!(fold(b"net", b"password"), fold(b"neu", b"password"));
        // And it is stable: the same inputs fold the same way twice.
        assert_eq!(fold(b"net", b"password"), fold(b"net", b"password"));
    }

    /// **What this build will actually join** — `make wifi-config`.
    ///
    /// Ignored, so it runs only when asked. It reads the compiled
    /// constants rather than the environment, which is the point: a value
    /// that never reached `option_env!` — a misspelled variable, a stale
    /// object, an export that did not survive a sub-make — prints as the
    /// default here. That is the mistake worth catching before a flash.
    ///
    /// The credential is reported by length and fingerprint only. It is
    /// never printed, because the output of this target ends up in build
    /// logs and on screens that get photographed.
    #[test]
    #[ignore = "a report, not an assertion: run it with `make wifi-config`"]
    fn the_configured_network_prints_itself() {
        if !configured() {
            // **No fingerprint for an unconfigured build.** Printing one
            // would read as confirmation that something is set, which is
            // the opposite of what this report is for.
            std::println!("  ssid         (none configured)");
            std::println!("  credential   (none configured)");
            std::println!("  fingerprint  (none configured)");
            return;
        }
        let ssid = core::str::from_utf8(SSID).unwrap_or("<not UTF-8>");
        std::println!("  ssid         {ssid}");
        std::println!(
            "  credential   {} characters ({})",
            PASSPHRASE.len(),
            if credential_is_raw_psk() {
                "raw 256-bit PSK"
            } else {
                "pass-phrase"
            },
        );
        std::println!("  fingerprint  {:04x}", credential_fingerprint());
    }

    /// A raw PSK is recognised by being 64 hex digits, not by being 64
    /// characters.
    #[test]
    fn a_raw_psk_is_hex_and_a_long_passphrase_is_refused() {
        // `credential_is_raw_psk` reads the compiled-in constant, so the
        // discrimination itself is tested through the same shape here.
        const fn raw(psk: &[u8]) -> bool {
            if psk.len() != PSK_HEX_CHARS {
                return false;
            }
            let mut i = 0;
            while i < psk.len() {
                if !psk[i].is_ascii_hexdigit() {
                    return false;
                }
                i += 1;
            }
            true
        }
        assert!(raw(&[b'a'; 64]));
        assert!(raw(&[b'0'; 64]));
        // 64 characters, not hex: a pass-phrase this long is out of range,
        // and reading it as a key would join a different network.
        assert!(!raw(&[b'z'; 64]));
        assert!(!raw(&[b'a'; 63]));
        assert!(!raw(&[b'a'; 65]));
    }
}
