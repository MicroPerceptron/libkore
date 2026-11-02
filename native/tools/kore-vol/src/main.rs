//! `kore-vol` — a kore volume, from the host: the command line over the
//! `kore_vol` library, which carries the whole procedure (see its docs).
//!
//! ```text
//! kore-vol [--face N] [--block-size B] <image or disk> ls [path]
//! kore-vol [--face N] [--block-size B] <image or disk> cat <path>
//! kore-vol [--face N] [--block-size B] <image or disk> put [--replace] <source> <path>
//! kore-vol [--face N] [--block-size B] <image or disk> rm <path>
//! kore-vol [--block-size B] <image> format
//! ```
//!
//! - **Paths are a terminal's.** `path` is what terminal `N` (default 0)
//!   types, relative to its home (`work/face-N` on the user volume): after
//!   `put dist js/hello`, terminal 0 reads the build as `js/hello`. `ls` and
//!   `cat` also read from the volume's root when the path starts with `/`;
//!   nothing writes there.
//! - **The image** is a raw disk image, offline: the file a QEMU run boots
//!   from after `qemu-img convert -O raw` (`make vol-put-x86` wraps both
//!   conversions), or a copy of a machine's disk; on Linux, as root, a
//!   machine's disk itself, with the machine shut down.
//! - **All or nothing.** Every check runs before anything is written, and
//!   the install becomes visible in one root publication, read back through
//!   a fresh recovery before the tool reports it.
//!
//! Exit status: 0 done, 1 refused (with the reason on stderr), 2 a usage
//! error.

use std::{path::PathBuf, process::ExitCode};

use kore_vol::{Kind, Listing, Mount, VolError};

const USAGE: &str = "\
usage: kore-vol [--face N] [--block-size B] <image or disk> ls [path]
       kore-vol [--face N] [--block-size B] <image or disk> cat <path>
       kore-vol [--face N] [--block-size B] <image or disk> put [--replace] <source> <path>
       kore-vol [--face N] [--block-size B] <image or disk> rm <path>
       kore-vol [--block-size B] <image> format";

enum Command {
    Ls {
        path: Option<String>,
    },
    Cat {
        path: String,
    },
    Put {
        source: PathBuf,
        path: String,
        replace: bool,
    },
    Rm {
        path: String,
    },
    Format,
}

struct Invocation {
    image: PathBuf,
    face: u8,
    block_size: Option<u32>,
    command: Command,
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Invocation, String> {
    let mut face = 0;
    let mut block_size = None;
    let mut replace = false;
    let mut positional = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--face" => {
                let value = args.next().ok_or("--face needs a terminal number")?;
                face = value
                    .parse::<u8>()
                    .ok()
                    .filter(|face| kore_vol::is_face(*face))
                    .ok_or_else(|| format!("--face {value}: not a terminal's face"))?;
            }
            "--block-size" => {
                let value = args.next().ok_or("--block-size needs a size in bytes")?;
                block_size = Some(
                    value
                        .parse::<u32>()
                        .ok()
                        .filter(|size| matches!(size, 512 | 4096))
                        .ok_or_else(|| format!("--block-size {value}: 512 or 4096"))?,
                );
            }
            "--replace" => replace = true,
            "-h" | "--help" => return Err(String::new()),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => positional.push(arg),
        }
    }
    let mut positional = positional.into_iter();
    let image = PathBuf::from(positional.next().ok_or("no image named")?);
    let command = match positional.next().as_deref() {
        Some("ls") => Command::Ls {
            path: positional.next(),
        },
        Some("cat") => Command::Cat {
            path: positional.next().ok_or("cat: no path")?,
        },
        Some("put") => Command::Put {
            source: PathBuf::from(positional.next().ok_or("put: no source")?),
            path: positional.next().ok_or("put: no destination path")?,
            replace,
        },
        Some("rm") => Command::Rm {
            path: positional.next().ok_or("rm: no path")?,
        },
        Some("format") => Command::Format,
        Some(other) => return Err(format!("unknown command {other}")),
        None => return Err("no command".into()),
    };
    if let Some(extra) = positional.next() {
        return Err(format!("unexpected argument {extra}"));
    }
    if replace && !matches!(command, Command::Put { .. }) {
        return Err("--replace applies to put alone".into());
    }
    Ok(Invocation {
        image,
        face,
        block_size,
        command,
    })
}

/// Say so when kore itself would mount the volume read-only: when recovery
/// had to take the fallback root. A medium opened to read mounts read-only
/// whatever its roots hold, so the mount's own mode says nothing here.
fn note_fallback(mount: &Mount) {
    let status = mount.status();
    if status.used_fallback_root {
        eprintln!(
            "kore-vol: reading the fallback root (generation {}): kore mounts this volume read-only",
            status.generation
        );
    }
}

/// A path as `ls` names it back: as given, from the volume's root when it
/// starts with `/`, and `.` for the home itself.
fn shown(path: Option<&str>) -> String {
    match path {
        Some(path) if path.starts_with('/') => {
            let parts = path
                .split('/')
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>();
            format!("/{}", parts.join("/"))
        }
        Some(path) => path.to_string(),
        None => String::from("."),
    }
}

fn run(invocation: Invocation) -> Result<(), VolError> {
    let Invocation {
        image,
        face,
        block_size,
        command,
    } = invocation;
    match command {
        Command::Ls { path } => {
            let listing = kore_vol::read(&image, block_size, |mount| {
                note_fallback(mount);
                mount.list(face, path.as_deref())
            })?;
            match listing {
                Listing::Directory(entries) => {
                    for entry in entries {
                        match entry.kind {
                            Kind::Directory => println!("dir  {:>12}  {}/", "", entry.name),
                            Kind::File => println!("file {:>12}  {}", entry.len, entry.name),
                        }
                    }
                }
                Listing::File { len } => println!("file {:>12}  {}", len, shown(path.as_deref())),
                Listing::EmptyHome => {}
            }
            Ok(())
        }
        Command::Cat { path } => kore_vol::read(&image, block_size, |mount| {
            note_fallback(mount);
            mount.read_to(face, &path, &mut std::io::stdout().lock())
        })
        .map(|_| ()),
        Command::Put {
            source,
            path,
            replace,
        } => {
            let installed = kore_vol::put(&image, block_size, face, &source, &path, replace)?;
            println!(
                "{path}: {} files, {} directories, {} bytes, in terminal {face}'s home \
                 (generation {} -> {}); every file read back identical",
                installed.files,
                installed.directories,
                installed.bytes,
                installed.before,
                installed.after
            );
            Ok(())
        }
        Command::Rm { path } => {
            let removed = kore_vol::rm(&image, block_size, face, &path)?;
            println!(
                "{path}: removed from terminal {face}'s home (generation {} -> {})",
                removed.before, removed.after
            );
            Ok(())
        }
        Command::Format => {
            let formatted = kore_vol::format(&image, block_size)?;
            println!(
                "formatted as kore's first boot would (generation {})",
                formatted.generation
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    let invocation = match parse(std::env::args().skip(1)) {
        Ok(invocation) => invocation,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("kore-vol: {message}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(invocation) {
        Ok(()) => ExitCode::SUCCESS,
        Err(VolError::Output(error)) => {
            eprintln!("kore-vol: standard output: {error}");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("kore-vol: {error}");
            ExitCode::FAILURE
        }
    }
}
