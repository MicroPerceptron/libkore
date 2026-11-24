//! The Kerr scene's fixed parameters, shared by the build-time bake
//! (`build/kerr.rs`) and the runtime that samples its lens map. Every pixel's
//! geodesic is traced once at build time from a camera at azimuth 0. The
//! metric and the disk are both symmetric about the spin axis, so swinging the
//! camera around that axis by `ψ` is exactly the scene with sky and disk turned
//! by `ψ`: the baked paths stay valid, and the runtime moves the camera by
//! shifting the azimuth it looks up.

/// Black-hole spin `a = J/M²` (M = 1); positive is prograde with the disk.
pub const SPIN: f32 = 0.9;
/// Camera distance in M, its inclination from the spin axis, and the vertical
/// field of view, all in the observer's own (ZAMO) frame.
pub const CAMERA_RADIUS: f32 = 30.0;
pub const CAMERA_INCLINATION_DEG: f32 = 78.0;
pub const FIELD_OF_VIEW_DEG: f32 = 31.0;
/// Yaw and pitch inside the camera frame: frames the hole right of centre,
/// leaving the left third dark for the overlay.
pub const LOOK_YAW_DEG: f32 = -9.0;
pub const LOOK_PITCH_DEG: f32 = 2.5;
/// Outer edge of the thin equatorial disk, in M. The inner edge is the
/// prograde ISCO for `SPIN`, computed by the bake.
pub const DISK_OUTER: f32 = 15.0;
/// Disk emissivity gain before tone mapping.
pub const DISK_GAIN: f32 = 0.12;

/// The lens map's canonical frame: 1920x1200, one texel per pixel on the
/// reference display. A display fits it by height and centres it
/// horizontally.
pub const LENS_WIDTH: usize = 1920;
pub const LENS_HEIGHT: usize = 1200;

/// The turbulence table: columns span one turn in φ (plus one repeated
/// column, so bilinear sampling wraps), rows span ln r over the disk. The
/// kit's noise reaches 576 cycles a turn and about 420 across the disk's
/// ln r; the bake fades each octave the table cannot hold rather than letting
/// it fold into stripes.
pub const TURBULENCE_COLUMNS: usize = 2048;
pub const TURBULENCE_ROWS: usize = 512;

/// How far the camera swings either way around the spin axis, in degrees.
/// The bake pads the sky atlas by this much in azimuth, so the swing never
/// samples past its edges: widening it costs atlas columns, not run time.
pub const ORBIT_SWAY_DEG: f32 = 10.0;

/// The sky: NASA SVS's Deep Star Maps 2020 (galactic plate carrée), turned
/// into the hole's frame and cut into two plates by `tools/sky-prep`
/// (see `assets/sky/README.md`). These place it: the galactic longitude and
/// latitude seen directly behind the hole, and how far the galactic plane is
/// rolled about that line of sight. Changing them means cutting the plates
/// again; the build refuses plates cut for another orientation.
/// From 30 M the Einstein ring spans some 21°, so what sits behind the hole
/// lands on a ring at the frame's edges and most of the view shows lensed
/// images of the far sky. This framing puts the southern sky below the
/// Magellanic Clouds behind the hole: the Milky Way bends into a bright arc
/// hugging the shadow's far side, and the Clouds streak across the left.
pub const SKY_BEHIND_GALACTIC_DEG: [f32; 2] = [300.0, -30.0];
pub const SKY_ROLL_DEG: f32 = 60.0;
/// The sky's brightness against the disk, applied when the bake fills the
/// atlas (the map is linear, its galactic core at 1.0).
pub const SKY_GAIN: f32 = 1.0;

/// The near plate: the map at full resolution (texels per turn), over the
/// azimuths (relative to the line of sight behind the hole) and latitudes the
/// camera can see nearly unlensed, swing included. The fine sky tier reads
/// it; the build fails, naming the bounds it needs, if the trace needs sky
/// outside it. These are the fine patch's bounds (azimuth −27.9°..37.6°,
/// latitude −33.4°..−0.7°, swing and bilinear reach included) with a degree
/// or so to spare: the patch depends on the trace, not on the framing above.
pub const SKY_NEAR_COLUMNS: usize = 8192;
pub const SKY_NEAR_AZIMUTH_DEG: [f32; 2] = [-29.0, 39.0];
pub const SKY_NEAR_LATITUDE_DEG: [f32; 2] = [-35.0, 1.0];
/// The far plate: the whole sphere at the medium tier's step, for the lensed
/// images.
pub const SKY_FAR_COLUMNS: usize = 2048;

/// Sky atlas sphere tiers, finest first: texels per turn of azimuth. A pixel
/// whose rays spread over more sky reads a coarser tier, so the sky arrives
/// prefiltered to its footprint and stars do not sparkle as the camera
/// swings. The finest tier is a patch at the near plate's step, covering only
/// what the camera sees nearly unlensed.
pub const SKY_TIER_COLUMNS: [usize; 3] = [SKY_FAR_COLUMNS, 256, 32];
