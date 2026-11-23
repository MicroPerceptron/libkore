# kore-rtw89 firmware

The RTL8922AE firmware is stored at
`vendor/firmware/realtek/rtw89/rtw8922a_fw.bin` in the shared
[firmware repository](https://github.com/MicroPerceptron/firmware), with its
original Realtek license notice. This directory retains the driver's acceptance
manifest, which is parsed at compile time into its size and BLAKE3 pin.

`make fetch-firmware` initializes all vendors; `make fetch-realtek-firmware`
remains an alias. Production x86 packaging prepares and verifies the shared
catalog automatically, stages the `kore-rtw89-firmware` boot module, and carries
the license notices into the image. Runtime verification remains unchanged.

The pinned bytes are upstream's `rtw8922a_fw-4.bin`, stored under the existing
boot-module filename. This format-4 container has no cut-A image; the commented
fallback in the manifest is a separate re-pin, never an automatic substitution.

See `manifest.toml` for the exact source commit, digest, and recorded loader
validation. Firmware availability proves neither a radio link nor frame traffic.
