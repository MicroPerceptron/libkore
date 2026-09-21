//! Typed application launch presets.
//!
//! An [`AppBundle`] is an inert expectation about an image's launch shape, not
//! authenticated metadata or authority. Authentication belongs to the image
//! service's executable profile; launchers compare that value with these
//! presets exactly before exec and the window controller transfer affine
//! capabilities through their existing transactions.
//!
//! The exact `Window`, `FilesWindow`, and `PowerWindow` bit combinations are reserved launch
//! presets. A legal superset is a different profile, not an automatically
//! compatible variant: callers must never infer window topology by testing a
//! subset of profile bits.
#![no_std]

use kore_abi::image::{
    PROFILE_ACCELERATOR, PROFILE_CHANNELS, PROFILE_FILESYSTEM, PROFILE_HEAP, PROFILE_SYSTEM_POWER,
    profile_valid,
};
pub use kore_dagfs::FilesystemFace;
pub use kore_window::WindowFace;

/// How service replies reach the process executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeTopology {
    /// Exec provisions the process's only doorbell.
    Exec,
    /// The window controller supplies the process doorbell.
    Window,
    /// The selected service (filesystem or power) and compositor receive
    /// sibling sends rooted in exec's response channel; the process holds
    /// its sole recv. Exec retains the root for peer reconciliation and
    /// atomically returns the compositor sibling with the task handle.
    ///
    /// The process-side adapter must wait as a multiplexer: sample and drain
    /// every ready plane, arm the park intent of each event channel that uses
    /// conditional kicks, recheck, and only then enter one executor wait. A
    /// single-plane `wait_past` may lawfully re-park after a sibling service
    /// supplied the wake, hiding that service from a monolithic event loop.
    SharedWindowService,
}

mod sealed {
    pub trait Sealed {}
}

/// A closed typed launch declaration. The built-in marker types are the only
/// implementations: downstream code cannot pair arbitrary coordinates or wake
/// topology with a profile. Selecting a preset grants no capability and cannot
/// widen an image's authenticated profile.
pub trait AppBundle: sealed::Sealed {
    const PROFILE: u32;
    const WINDOW_FACE: Option<WindowFace>;
    const FILESYSTEM_FACE: Option<FilesystemFace>;
    const WAKE: WakeTopology;
}

/// A process with no automatically attached service face.
pub enum Command {}

impl Command {
    pub const PROFILE: u32 = 0;
    pub const WINDOW_FACE: Option<WindowFace> = None;
    pub const FILESYSTEM_FACE: Option<FilesystemFace> = None;
    pub const WAKE: WakeTopology = WakeTopology::Exec;
}

impl sealed::Sealed for Command {}

impl AppBundle for Command {
    const PROFILE: u32 = Command::PROFILE;
    const WINDOW_FACE: Option<WindowFace> = Command::WINDOW_FACE;
    const FILESYSTEM_FACE: Option<FilesystemFace> = Command::FILESYSTEM_FACE;
    const WAKE: WakeTopology = Command::WAKE;
}

/// A command process with the canonical private filesystem face.
pub enum FsCommand {}

impl FsCommand {
    pub const PROFILE: u32 = PROFILE_FILESYSTEM;
    pub const WINDOW_FACE: Option<WindowFace> = None;
    pub const FILESYSTEM_FACE: Option<FilesystemFace> = Some(FilesystemFace::CANONICAL);
    pub const WAKE: WakeTopology = WakeTopology::Exec;
}

impl sealed::Sealed for FsCommand {}

impl AppBundle for FsCommand {
    const PROFILE: u32 = FsCommand::PROFILE;
    const WINDOW_FACE: Option<WindowFace> = FsCommand::WINDOW_FACE;
    const FILESYSTEM_FACE: Option<FilesystemFace> = FsCommand::FILESYSTEM_FACE;
    const WAKE: WakeTopology = FsCommand::WAKE;
}

/// A standalone window application with heap and bounded channel-space
/// facilities.
pub enum Window {}

impl Window {
    pub const PROFILE: u32 = PROFILE_HEAP | PROFILE_CHANNELS;
    pub const WINDOW_FACE: Option<WindowFace> = Some(WindowFace::STANDALONE);
    pub const FILESYSTEM_FACE: Option<FilesystemFace> = None;
    pub const WAKE: WakeTopology = WakeTopology::Window;
}

impl sealed::Sealed for Window {}

impl AppBundle for Window {
    const PROFILE: u32 = Window::PROFILE;
    const WINDOW_FACE: Option<WindowFace> = Window::WINDOW_FACE;
    const FILESYSTEM_FACE: Option<FilesystemFace> = Window::FILESYSTEM_FACE;
    const WAKE: WakeTopology = Window::WAKE;
}

/// A window application composed with the canonical filesystem face.
pub enum FilesWindow {}

impl FilesWindow {
    pub const PROFILE: u32 = PROFILE_HEAP | PROFILE_CHANNELS | PROFILE_FILESYSTEM;
    pub const WINDOW_FACE: Option<WindowFace> = Some(WindowFace::WITH_FILESYSTEM);
    pub const FILESYSTEM_FACE: Option<FilesystemFace> = Some(FilesystemFace::CANONICAL);
    pub const WAKE: WakeTopology = WakeTopology::SharedWindowService;
}

impl sealed::Sealed for FilesWindow {}

impl AppBundle for FilesWindow {
    const PROFILE: u32 = FilesWindow::PROFILE;
    const WINDOW_FACE: Option<WindowFace> = FilesWindow::WINDOW_FACE;
    const FILESYSTEM_FACE: Option<FilesystemFace> = FilesWindow::FILESYSTEM_FACE;
    const WAKE: WakeTopology = FilesWindow::WAKE;
}

/// A window composed with the process-private PowerService face. Exec supplies
/// its reply channel; the compositor receives a sibling of that same wake root.
pub enum PowerWindow {}

impl PowerWindow {
    pub const PROFILE: u32 = Window::PROFILE | PROFILE_SYSTEM_POWER;
    pub const WINDOW_FACE: Option<WindowFace> = Some(WindowFace::STANDALONE);
    pub const FILESYSTEM_FACE: Option<FilesystemFace> = None;
    pub const WAKE: WakeTopology = WakeTopology::SharedWindowService;
}

impl sealed::Sealed for PowerWindow {}

impl AppBundle for PowerWindow {
    const PROFILE: u32 = PowerWindow::PROFILE;
    const WINDOW_FACE: Option<WindowFace> = PowerWindow::WINDOW_FACE;
    const FILESYSTEM_FACE: Option<FilesystemFace> = PowerWindow::FILESYSTEM_FACE;
    const WAKE: WakeTopology = PowerWindow::WAKE;
}

/// A window application composed with one private recyclable accelerator
/// face. The service and compositor share the process executor's wake root;
/// accelerator completion channels remain explicit per-client grants.
pub enum AcceleratedWindow {}

impl AcceleratedWindow {
    pub const PROFILE: u32 = Window::PROFILE | PROFILE_ACCELERATOR;
    pub const WINDOW_FACE: Option<WindowFace> = Some(WindowFace::STANDALONE);
    pub const FILESYSTEM_FACE: Option<FilesystemFace> = None;
    pub const WAKE: WakeTopology = WakeTopology::SharedWindowService;
}

impl sealed::Sealed for AcceleratedWindow {}

impl AppBundle for AcceleratedWindow {
    const PROFILE: u32 = AcceleratedWindow::PROFILE;
    const WINDOW_FACE: Option<WindowFace> = AcceleratedWindow::WINDOW_FACE;
    const FILESYSTEM_FACE: Option<FilesystemFace> = AcceleratedWindow::FILESYSTEM_FACE;
    const WAKE: WakeTopology = AcceleratedWindow::WAKE;
}

/// The window-capable preset selected by an authenticated image profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowBundle {
    profile: u32,
    face: WindowFace,
    wake: WakeTopology,
}

impl WindowBundle {
    pub const fn from_profile(profile: u32) -> Option<Self> {
        if profile == Window::PROFILE {
            Some(Self {
                profile,
                face: WindowFace::STANDALONE,
                wake: WakeTopology::Window,
            })
        } else if profile == PowerWindow::PROFILE {
            Some(Self {
                profile,
                face: WindowFace::STANDALONE,
                wake: WakeTopology::SharedWindowService,
            })
        } else if profile == FilesWindow::PROFILE {
            Some(Self {
                profile,
                face: WindowFace::WITH_FILESYSTEM,
                wake: WakeTopology::SharedWindowService,
            })
        } else if profile == AcceleratedWindow::PROFILE {
            Some(Self {
                profile,
                face: WindowFace::STANDALONE,
                wake: WakeTopology::SharedWindowService,
            })
        } else {
            None
        }
    }

    pub const fn profile(self) -> u32 {
        self.profile
    }

    pub const fn face(self) -> WindowFace {
        self.face
    }

    pub const fn wake(self) -> WakeTopology {
        self.wake
    }
}

/// Small import surface for application source. It contains only the five
/// closed inert presets; implementation coordinates remain explicitly named.
pub mod prelude {
    pub use crate::{AcceleratedWindow, Command, FilesWindow, FsCommand, PowerWindow, Window};
}

const _: () = {
    assert!(profile_valid(Command::PROFILE));
    assert!(profile_valid(FsCommand::PROFILE));
    assert!(profile_valid(Window::PROFILE));
    assert!(profile_valid(FilesWindow::PROFILE));
    assert!(profile_valid(PowerWindow::PROFILE));
    assert!(profile_valid(AcceleratedWindow::PROFILE));
    assert!(WindowFace::WITH_FILESYSTEM.ring() != FilesystemFace::CANONICAL.control_ring());
    assert!(WindowFace::WITH_FILESYSTEM.ring() != FilesystemFace::CANONICAL.data_window());
    assert!(WindowFace::WITH_FILESYSTEM.request_kick() != FilesystemFace::CANONICAL.request_kick());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_exact_and_faces_do_not_collide() {
        assert_eq!(Command::PROFILE, 0);
        assert_eq!(FsCommand::PROFILE, PROFILE_FILESYSTEM);
        assert_eq!(Window::PROFILE, PROFILE_HEAP | PROFILE_CHANNELS);
        assert_eq!(
            FilesWindow::PROFILE,
            PROFILE_HEAP | PROFILE_CHANNELS | PROFILE_FILESYSTEM
        );
        assert_eq!(Window::WINDOW_FACE, Some(WindowFace::STANDALONE));
        assert_eq!(FsCommand::FILESYSTEM_FACE, Some(FilesystemFace::CANONICAL));
        assert_eq!(FilesWindow::WINDOW_FACE, Some(WindowFace::WITH_FILESYSTEM));
        assert_ne!(WindowFace::WITH_FILESYSTEM.ring(), 0);
        assert_ne!(WindowFace::WITH_FILESYSTEM.ring(), 1);
        assert_ne!(WindowFace::WITH_FILESYSTEM.request_kick(), 0);
        assert_eq!(FilesWindow::FILESYSTEM_FACE, FsCommand::FILESYSTEM_FACE);
        assert_eq!(
            AcceleratedWindow::PROFILE,
            PROFILE_HEAP | PROFILE_CHANNELS | PROFILE_ACCELERATOR
        );
        assert_eq!(AcceleratedWindow::WINDOW_FACE, Some(WindowFace::STANDALONE));
    }

    #[test]
    fn every_legal_profile_classifies_only_the_exact_window_presets() {
        for profile in 0..=kore_abi::image::PROFILE_KNOWN {
            if !profile_valid(profile) {
                continue;
            }
            let classified = WindowBundle::from_profile(profile);
            if profile == Window::PROFILE {
                assert_eq!(classified.unwrap().wake(), WakeTopology::Window);
            } else if profile == FilesWindow::PROFILE
                || profile == PowerWindow::PROFILE
                || profile == AcceleratedWindow::PROFILE
            {
                assert_eq!(
                    classified.unwrap().wake(),
                    WakeTopology::SharedWindowService
                );
            } else {
                assert_eq!(
                    classified, None,
                    "legal profile {profile:#x} aliased a preset"
                );
            }
        }
    }
}
