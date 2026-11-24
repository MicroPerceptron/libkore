//! Native axui case study: Iced renderer calls become an axui display list,
//! axgfx validates the physical target contract, and axrd lowers the frame to
//! an ordinary `.ax` graph before the selected `virtio-accel` provider
//! executes it.
//!
//! Run with:
//!   cargo +nightly run --release -p axrd --example axui --features "window,axui"
//!
//! Set `AXRD_AXUI_HEADLESS=1` for one non-windowed render, `AXRD_AXUI_PPM=path`
//! to save it, and `AXRD_AXUI_SIZE=WIDTHxHEIGHT` to select the backing size.
//! `AXRD_AXUI_BACKEND` accepts `auto`, `cpu`, `accelerator`, or
//! `heterogeneous`. Add `coreml` on macOS or `openvino` on a host with
//! `libopenvino_c` to validate the same `Accelerator` bridge used by Kore.

use ax::{AdmissionPlanner, DispatchLane, DispatchRequest, Graph};
use axcomp::{
    LinearXrgbWords, PlanarFormat, PlanarOutput, PreparedCache, PreparedScene, SignatureCache,
    pack_planar_xrgb_words, structural_graph_signature,
};
use axgfx::{
    ByteRange, Damage, DeviceEpoch, Extent2d, Features, GraphicsBackend, GraphicsCapabilities,
    GraphicsError, GraphicsLimits, PixelFormat, PixelRect, ReleaseFailure, RenderRequest,
    RetainUntil, SubmissionState, SubmissionTracker, SubmitFailure, SurfaceEpoch, SurfaceInfo,
    SurfaceModifier, SurfaceRetirement, SurfaceUsage, TargetReuse, TargetSupport, VisibilityEpoch,
    VisibilityMode, validate_render_request,
};
use axinput::{
    AbsoluteRange, CursorIcon as AxCursorIcon, CursorRequest, DeviceId, MotionScale, PointerInput,
    PointerInputKind, PointerPosition, PointerSource, Seat, SeatId, Viewport as InputViewport,
};
#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
use axnn_vaccel::Accelerator;
use axnn_vaccel::{
    ACCEL_BACKEND, AxCompileError, AxDeviceModel, AxDispatchVariant, AxExecutable, AxHetero,
    AxTosaAccelerator, BackendError, CPU_BACKEND, Session, admit_variants,
};
use axrd::{
    axui::{AxuiGraphError, compile_frame_graph_with_output_in},
    graph::{
        GraphBackground, GraphOutputFormat, GraphRadialLight, GraphRadialProfile, GraphRadialRim,
    },
};
use axui_core::Frame;
use iced_core::{
    Border, Color, Point, Rectangle, Renderer as _, Shadow, Size,
    renderer::{self, Scale as IcedScale},
};
use softbuffer::{Context as SoftContext, Surface as SoftSurface};
use std::{
    alloc::System,
    num::NonZeroU32,
    rc::Rc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle},
    window::{CursorIcon as WinitCursorIcon, Window as WinitWindow, WindowId},
};

const DEFAULT_WIDTH: usize = 960;
const DEFAULT_HEIGHT: usize = 600;
const MAX_PRIMITIVES: usize = 128;
const FRAME_DEADLINE_US: u64 = 16_667;
const INITIAL_CPU_FRAME_US: u64 = 9_000;
// Sample an admitted host accelerator on the first idle heterogeneous frame;
// measured provider completion immediately feeds the subsequent estimate.
const INITIAL_ACCELERATOR_FRAME_US: u64 = 6_000;
const PREPARED_CACHE_LIMIT: usize = 2;
const STRUCTURAL_CACHE_LIMIT: usize = 16;
const RESIZE_SETTLE: Duration = Duration::from_millis(48);
const AXRD_ERROR_DOMAIN: u32 = u32::from_be_bytes(*b"AXRD");

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
type HostRuntime = AxHetero<AxTosaAccelerator, virtio_accel_openvino::OpenVinoAccelerator>;

#[cfg(all(
    not(all(feature = "openvino", axrd_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
type HostRuntime = AxHetero<AxTosaAccelerator, virtio_accel_coreml::CoreMlAccelerator>;

#[cfg(not(any(
    all(feature = "openvino", axrd_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
type HostRuntime = AxHetero<AxTosaAccelerator, AxTosaAccelerator>;

fn cpu_session() -> Session<AxTosaAccelerator> {
    Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target())
        .expect("failed to create the portable ax CPU provider")
}

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
fn host_runtime() -> (HostRuntime, String) {
    let provider = std::env::var("AXRD_AXUI_OPENVINO_DEVICE").map_or_else(
        |_| virtio_accel_openvino::OpenVinoAccelerator::new(),
        |device| virtio_accel_openvino::OpenVinoAccelerator::with_device(&device),
    );
    let (accelerator, label) = match provider {
        Ok(provider) => {
            let label = provider
                .device_info()
                .map(|info| format!("OpenVINO {:?}", info.identity.class))
                .unwrap_or_else(|_| "OpenVINO".to_owned());
            match Session::new(provider, virtio_accel_openvino::OPENVINO_TOSA_TARGET) {
                Ok(session) => (Some(session), label),
                Err(_) => (None, "CPU (OpenVINO session unavailable)".to_owned()),
            }
        }
        Err(_) => (None, "CPU (OpenVINO unavailable)".to_owned()),
    };
    (AxHetero::new(cpu_session(), accelerator), label)
}

#[cfg(all(
    not(all(feature = "openvino", axrd_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
fn host_runtime() -> (HostRuntime, String) {
    let accelerator = virtio_accel_coreml::CoreMlAccelerator::new_tosa()
        .ok()
        .and_then(|provider| Session::new(provider, virtio_accel_coreml::COREML_TOSA_TARGET).ok());
    let label = if accelerator.is_some() {
        "Core ML / Neural Engine".to_owned()
    } else {
        "CPU (Core ML unavailable)".to_owned()
    };
    (AxHetero::new(cpu_session(), accelerator), label)
}

#[cfg(not(any(
    all(feature = "openvino", axrd_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
fn host_runtime() -> (HostRuntime, String) {
    (AxHetero::new(cpu_session(), None), "CPU".to_owned())
}

#[cfg(all(feature = "openvino", axrd_openvino_runtime))]
fn device_model(runtime: &HostRuntime, force_accelerator: bool) -> AxDeviceModel {
    let mut model = AxDeviceModel::default();
    if force_accelerator {
        model.cpu = false;
        model.accelerator_launch_cost = 0;
        model.accelerator_speedup = 64;
        model.transfer_fixed_cost = 0;
        model.transfer_byte_cost = 0;
    }
    runtime.accel.as_ref().map_or(model.clone(), |session| {
        model.with_accelerator_tosa_capability(session.provider(), session.target())
    })
}

#[cfg(all(
    not(all(feature = "openvino", axrd_openvino_runtime)),
    feature = "coreml",
    target_os = "macos"
))]
fn device_model(runtime: &HostRuntime, force_accelerator: bool) -> AxDeviceModel {
    let mut model = AxDeviceModel::coreml();
    if force_accelerator {
        model.cpu = false;
        model.accelerator_launch_cost = 0;
        model.accelerator_speedup = 64;
        model.transfer_fixed_cost = 0;
        model.transfer_byte_cost = 0;
    }
    runtime.accel.as_ref().map_or(model.clone(), |session| {
        model.with_accelerator_tosa_capability(session.provider(), session.target())
    })
}

#[cfg(not(any(
    all(feature = "openvino", axrd_openvino_runtime),
    all(feature = "coreml", target_os = "macos")
)))]
fn device_model(_runtime: &HostRuntime, _force_accelerator: bool) -> AxDeviceModel {
    AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendRequest {
    Auto,
    Cpu,
    Accelerator,
    Heterogeneous,
}

fn backend_request() -> BackendRequest {
    match std::env::var("AXRD_AXUI_BACKEND").ok().as_deref() {
        None | Some("auto") => BackendRequest::Auto,
        Some("cpu") => BackendRequest::Cpu,
        Some("accelerator") => BackendRequest::Accelerator,
        Some("heterogeneous") | Some("hetero") => BackendRequest::Heterogeneous,
        Some(value) => panic!(
            "AXRD_AXUI_BACKEND must be auto, cpu, accelerator, or heterogeneous; got {value:?}"
        ),
    }
}

#[derive(Clone)]
struct HostTarget {
    pixels: Vec<u32>,
    width: usize,
    height: usize,
}

impl HostTarget {
    fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![0; width * height],
            width,
            height,
        }
    }

    fn resampled(&self, width: usize, height: usize) -> Self {
        if (width, height) == (self.width, self.height) {
            return self.clone();
        }
        let mut target = Self::new(width, height);
        if self.width == 0 || self.height == 0 || width == 0 || height == 0 {
            return target;
        }

        let axis = |dst: usize, src: usize| {
            (0..dst)
                .map(|position| {
                    let coordinate = (((position as u64 * 2 + 1) * src as u64) << 15) / dst as u64;
                    let coordinate = coordinate.saturating_sub(1 << 15);
                    let first = ((coordinate >> 16) as usize).min(src - 1);
                    let second = first.saturating_add(1).min(src - 1);
                    (first, second, (coordinate & 0xffff) as u32)
                })
                .collect::<Vec<_>>()
        };
        let xs = axis(width, self.width);
        let ys = axis(height, self.height);
        let lerp = |left: u32, right: u32, weight: u32| {
            ((left * (65_536 - weight) + right * weight + 32_768) >> 16).min(255)
        };
        for (y, &(y0, y1, wy)) in ys.iter().enumerate() {
            for (x, &(x0, x1, wx)) in xs.iter().enumerate() {
                let p00 = self.pixels[y0 * self.width + x0];
                let p10 = self.pixels[y0 * self.width + x1];
                let p01 = self.pixels[y1 * self.width + x0];
                let p11 = self.pixels[y1 * self.width + x1];
                let channel = |shift: u32| {
                    let top = lerp((p00 >> shift) & 0xff_u32, (p10 >> shift) & 0xff_u32, wx);
                    let bottom = lerp((p01 >> shift) & 0xff_u32, (p11 >> shift) & 0xff_u32, wx);
                    lerp(top, bottom, wy)
                };
                target.pixels[y * width + x] =
                    0xff00_0000 | channel(16) << 16 | channel(8) << 8 | channel(0);
            }
        }
        target
    }
}

#[derive(Debug)]
struct HostSubmission {
    tracker: SubmissionTracker,
    device: DeviceEpoch,
}

struct PreparedExecutables {
    cpu: AxExecutable<System>,
    accelerator: Option<AxExecutable<System>>,
}

#[derive(Clone, Copy, PartialEq)]
struct PreparedKey {
    background: GraphBackground,
    output_format: GraphOutputFormat,
}

/// The first host adapter is intentionally staged: `.ax` providers currently
/// return planar outputs, then this adapter packs them into the exact XRGB
/// target. A Kore adapter can keep the same frontend/lowering and replace only
/// this target binding and event implementation.
struct HostGraphicsBackend {
    runtime: HostRuntime,
    backend_label: String,
    request: BackendRequest,
    scheduler_epoch: Instant,
    lanes: [DispatchLane; 2],
    estimated_frame_us: [u64; 2],
    next_device_epoch: u64,
    prepared: PreparedCache<System, PreparedKey, PreparedExecutables>,
    strict_accelerator_rejections: SignatureCache<System>,
    accelerator_empty_scenes: SignatureCache<System>,
    profile: bool,
}

impl HostGraphicsBackend {
    fn new() -> Self {
        let (runtime, backend_label) = host_runtime();
        Self {
            runtime,
            backend_label,
            request: backend_request(),
            scheduler_epoch: Instant::now(),
            lanes: [
                DispatchLane::new(CPU_BACKEND, 0),
                DispatchLane::new(ACCEL_BACKEND, 0),
            ],
            estimated_frame_us: [INITIAL_CPU_FRAME_US, INITIAL_ACCELERATOR_FRAME_US],
            next_device_epoch: 1,
            prepared: PreparedCache::new_in(PREPARED_CACHE_LIMIT, System)
                .expect("failed to reserve prepared scene cache"),
            strict_accelerator_rejections: SignatureCache::new_in(STRUCTURAL_CACHE_LIMIT, System)
                .expect("failed to reserve accelerator rejection cache"),
            accelerator_empty_scenes: SignatureCache::new_in(STRUCTURAL_CACHE_LIMIT, System)
                .expect("failed to reserve accelerator placement cache"),
            profile: std::env::var_os("AXRD_AXUI_PROFILE").is_some(),
        }
    }

    fn output_format(&self) -> GraphOutputFormat {
        // Final display narrowing happens after F32 composition and
        // antialiasing. It cuts the staged readback to one quarter on the
        // portable CPU path. Keep F32 whenever a live accelerator variant is
        // requested because Core ML does not currently accept the CAST tail.
        if self.request == BackendRequest::Cpu || self.runtime.accel.is_none() {
            GraphOutputFormat::PlanarU8
        } else {
            GraphOutputFormat::PlanarF32
        }
    }

    fn release_prepared(
        &mut self,
        prepared: PreparedScene<System, PreparedKey, PreparedExecutables>,
    ) {
        let prepared = prepared.into_payload();
        if let Some(accelerator) = prepared.accelerator
            && let Err(error) = self.runtime.release_executable(accelerator)
        {
            eprintln!("axui: accelerator executable eviction failed ({error:?})");
        }
        if let Err(error) = self.runtime.release_executable(prepared.cpu) {
            eprintln!("axui: CPU executable eviction failed ({error:?})");
        }
    }

    fn cached_frame(
        &mut self,
        frame: axui_core::FrameView<'_, System>,
        background: GraphBackground,
        output_format: GraphOutputFormat,
    ) -> Option<usize> {
        self.prepared.find(
            frame,
            &PreparedKey {
                background,
                output_format,
            },
        )
    }

    fn prepare_frame(
        &mut self,
        graph: Graph<'_>,
        source: axui_core::FrameView<'_, System>,
        background: GraphBackground,
        output_format: GraphOutputFormat,
        scene_signature: u64,
    ) -> Result<usize, GraphicsError> {
        if self.request == BackendRequest::Accelerator && self.runtime.accel.is_none() {
            eprintln!(
                "axui: strict accelerator mode requested, but {} has no live accelerator session",
                self.backend_label
            );
            return Err(external_error(5));
        }
        let cpu_model = AxDeviceModel {
            accelerator: false,
            ..AxDeviceModel::default()
        };
        let cpu = self
            .runtime
            .compile_in(&graph, &cpu_model, System)
            .map_err(|_| external_error(2))?;
        let force_accelerator = matches!(
            self.request,
            BackendRequest::Accelerator | BackendRequest::Heterogeneous
        );
        let strict_rejected = self.strict_accelerator_rejections.contains(scene_signature);
        if self.request == BackendRequest::Accelerator && strict_rejected {
            let _ = self.runtime.release_executable(cpu);
            return Err(external_error(5));
        }

        let mut structural_strict_failure = strict_rejected;
        let accelerator = if self.request == BackendRequest::Cpu
            || self.runtime.accel.is_none()
            || self.accelerator_empty_scenes.contains(scene_signature)
        {
            None
        } else {
            let compiled = if force_accelerator && strict_rejected {
                self.runtime
                    .compile_in(&graph, &device_model(&self.runtime, false), System)
            } else {
                match self.runtime.compile_in(
                    &graph,
                    &device_model(&self.runtime, force_accelerator),
                    System,
                ) {
                    Ok(executable) => Ok(executable),
                    Err(error) if self.request == BackendRequest::Heterogeneous => {
                        structural_strict_failure = structural_accelerator_rejection(&error);
                        if structural_strict_failure {
                            self.strict_accelerator_rejections.insert(scene_signature);
                        }
                        eprintln!(
                            "axui: strict accelerator admission failed ({error:?}); retrying mixed placement"
                        );
                        self.runtime
                            .compile_in(&graph, &device_model(&self.runtime, false), System)
                    }
                    Err(error) if self.request == BackendRequest::Accelerator => {
                        if structural_accelerator_rejection(&error) {
                            self.strict_accelerator_rejections.insert(scene_signature);
                        }
                        eprintln!("axui: strict accelerator graph admission failed ({error:?})");
                        let _ = self.runtime.release_executable(cpu);
                        return Err(external_error(5));
                    }
                    Err(error) => {
                        eprintln!(
                            "axui: calibrated accelerator placement failed ({error:?}); retaining the CPU variant"
                        );
                        Err(error)
                    }
                }
            };
            match compiled {
                Ok(executable) if executable.single_backend() == Some(CPU_BACKEND) => {
                    if structural_strict_failure {
                        self.accelerator_empty_scenes.insert(scene_signature);
                    }
                    if let Err(error) = self.runtime.release_executable(executable) {
                        eprintln!("axui: duplicate CPU executable eviction failed ({error:?})");
                    }
                    None
                }
                Ok(executable) => Some(executable),
                Err(fallback) => {
                    if self.request == BackendRequest::Heterogeneous {
                        eprintln!(
                            "axui: mixed graph admission also failed ({fallback:?}); retaining the CPU variant"
                        );
                    }
                    None
                }
            }
        };

        let evicted = self.prepared.insert(
            source,
            PreparedKey {
                background,
                output_format,
            },
            PreparedExecutables { cpu, accelerator },
            System,
        );
        if let Some(evicted) = evicted {
            self.release_prepared(evicted);
        }
        let index = self.prepared.len() - 1;
        Ok(index)
    }
}

impl Drop for HostGraphicsBackend {
    fn drop(&mut self) {
        while let Some(prepared) = self.prepared.pop() {
            self.release_prepared(prepared);
        }
    }
}

fn structural_accelerator_rejection(error: &AxCompileError) -> bool {
    matches!(
        error,
        AxCompileError::Backend(BackendError::Unsupported | BackendError::Incompatible)
    )
}

impl GraphicsBackend<System> for HostGraphicsBackend {
    type Target = HostTarget;
    type Submission = HostSubmission;

    fn capabilities(&self) -> GraphicsCapabilities {
        GraphicsCapabilities {
            features: Features::SOLID_QUAD.union(Features::PIXEL_SNAP),
            limits: GraphicsLimits::new(
                Extent2d::new(u16::MAX.into(), u16::MAX.into()).unwrap(),
                MAX_PRIMITIVES as u32,
                MAX_PRIMITIVES as u32,
                1,
                8,
            )
            .unwrap(),
        }
    }

    fn target_support(&self, target: &Self::Target, surface: SurfaceInfo) -> Option<TargetSupport> {
        let extent = surface.extent();
        let exact_pixels = target
            .width
            .checked_mul(target.height)
            .is_some_and(|pixels| pixels == target.pixels.len());
        (surface.format() == PixelFormat::XRGB8888_SRGB
            && surface.modifier() == SurfaceModifier::LINEAR
            && surface.planes().len() == 1
            && surface.planes()[0].row_stride() == u64::from(extent.width()) * 4
            && target.width == extent.width() as usize
            && target.height == extent.height() as usize
            && exact_pixels)
            .then(|| TargetSupport::staged(VisibilityMode::Coherent, RetainUntil::Visible))
    }

    fn submit(
        &mut self,
        mut request: RenderRequest<'_, Self::Target, System>,
    ) -> Result<Self::Submission, SubmitFailure<Self::Submission>> {
        let support = self.target_support(request.target(), request.surface());
        validate_render_request(self.capabilities(), support, &request)
            .map_err(SubmitFailure::Rejected)?;

        let viewport = request.viewport();
        let output_format = self.output_format();
        let background = shader_background(viewport.width(), viewport.height());
        let source = request.frame();
        let cache_started = Instant::now();
        let (prepared_index, cache_hit, lower_us, prepare_us) =
            if let Some(index) = self.cached_frame(source, background, output_format) {
                (index, true, 0, elapsed_micros(cache_started))
            } else {
                let lower_started = Instant::now();
                let graph = compile_frame_graph_with_output_in::<_, MAX_PRIMITIVES>(
                    source,
                    background,
                    output_format,
                    System,
                )
                .map_err(|error| SubmitFailure::Rejected(lowering_error(error)))?;
                let lower_us = elapsed_micros(lower_started);
                let graph_view = graph.graph();
                let signature = structural_graph_signature(graph_view)
                    .map_err(|_| SubmitFailure::Rejected(external_error(1)))?;
                let prepare_started = Instant::now();
                let index = self
                    .prepare_frame(graph_view, source, background, output_format, signature)
                    .map_err(SubmitFailure::Rejected)?;
                (index, false, lower_us, elapsed_micros(prepare_started))
            };
        let prepared = self
            .prepared
            .get(prepared_index)
            .expect("prepared scene index came from the live cache")
            .payload();
        let cpu_executable = &prepared.cpu;
        let accelerator_executable = &prepared.accelerator;

        let ready_at = elapsed_micros(self.scheduler_epoch);
        let dispatch_request =
            DispatchRequest::new(ready_at, ready_at.saturating_add(FRAME_DEADLINE_US), 100);
        let cpu_variant = AxDispatchVariant::new(
            cpu_executable,
            CPU_BACKEND,
            self.estimated_frame_us[CPU_BACKEND as usize],
            100,
        )
        .expect("portable axui graph needs one CPU capacity reservation");
        let accelerator_variant = accelerator_executable.as_ref().and_then(|executable| {
            AxDispatchVariant::new(
                executable,
                ACCEL_BACKEND,
                self.estimated_frame_us[ACCEL_BACKEND as usize],
                100,
            )
        });
        let compound = matches!(
            self.request,
            BackendRequest::Auto | BackendRequest::Heterogeneous
        ) && accelerator_executable
            .as_ref()
            .is_some_and(|executable| executable.single_backend().is_none());
        let execute_started = Instant::now();
        let (outputs, admitted_backend, partition_count, accelerator_nodes) = if compound {
            // AdmissionPlanner intentionally rejects mixed-backend variants:
            // they need more than one capacity reservation. The placed `.ax`
            // schedule already owns that partition boundary, so execute it as
            // one compound AxHetero submission and report it explicitly.
            let executable = accelerator_executable
                .as_ref()
                .expect("compound axui executable was just classified");
            let outputs = self
                .runtime
                .run_in(executable, &[], System)
                .map_err(|_| SubmitFailure::Rejected(external_error(4)))?;
            let accelerator_nodes = executable
                .schedule()
                .placements()
                .iter()
                .filter(|&&backend| backend == ACCEL_BACKEND)
                .count();
            (
                outputs,
                "compound CPU+accelerator".to_owned(),
                executable.schedule().plan().partitions.len(),
                accelerator_nodes,
            )
        } else {
            let mut planner = AdmissionPlanner::new(&mut self.lanes)
                .map_err(|_| SubmitFailure::Rejected(external_error(3)))?;
            let choice = match (self.request, accelerator_variant) {
                (BackendRequest::Accelerator, Some(accelerator)) => {
                    admit_variants(&mut planner, dispatch_request, &[accelerator])
                }
                (BackendRequest::Auto | BackendRequest::Heterogeneous, Some(accelerator)) => {
                    admit_variants(&mut planner, dispatch_request, &[cpu_variant, accelerator])
                }
                _ => admit_variants(&mut planner, dispatch_request, &[cpu_variant]),
            }
            .map_err(|_| SubmitFailure::Rejected(external_error(3)))?;
            let reservation = choice.decision();
            let run_started = Instant::now();
            let outputs = self
                .runtime
                .run_in(choice.executable(), &[], System)
                .map_err(|_| SubmitFailure::Rejected(external_error(4)))?;
            let measured_us = u64::try_from(run_started.elapsed().as_micros())
                .unwrap_or(u64::MAX)
                .max(1);
            planner
                .reconcile(reservation, elapsed_micros(self.scheduler_epoch))
                .map_err(|_| SubmitFailure::Rejected(external_error(3)))?;
            drop(planner);
            let estimate = &mut self.estimated_frame_us[reservation.backend as usize];
            *estimate = estimate.saturating_mul(3).saturating_add(measured_us) / 4;
            let partition_count = choice.executable().schedule().plan().partitions.len();
            let accelerator_nodes = choice
                .executable()
                .schedule()
                .placements()
                .iter()
                .filter(|&&backend| backend == ACCEL_BACKEND)
                .count();
            (
                outputs,
                format!("single lane {}", reservation.backend),
                partition_count,
                accelerator_nodes,
            )
        };
        let execute_us = elapsed_micros(execute_started);
        eprintln!(
            "axui: {:?} admitted {admitted_backend} through {} ({} partition(s), {accelerator_nodes} accelerator node(s))",
            self.request, self.backend_label, partition_count,
        );
        let pack_started = Instant::now();
        if outputs.len() != 3 {
            return Err(SubmitFailure::Rejected(external_error(4)));
        }
        let extent = Extent2d::new(viewport.width(), viewport.height())
            .map_err(|_| SubmitFailure::Rejected(GraphicsError::InvalidArgument))?;
        let planar_format = match output_format {
            GraphOutputFormat::PlanarF32 => PlanarFormat::F32,
            GraphOutputFormat::PlanarU8 => PlanarFormat::U8Biased,
            GraphOutputFormat::PlanarF16 => PlanarFormat::F16,
        };
        let output = PlanarOutput::new(
            [&outputs[0], &outputs[1], &outputs[2]],
            planar_format,
            extent,
        )
        .map_err(|_| SubmitFailure::Rejected(external_error(4)))?;
        let damage = request.damage();
        let target = request.target_mut();
        let target_width = target.width;
        let mut target = LinearXrgbWords::new(&mut target.pixels, extent, target_width)
            .map_err(|_| SubmitFailure::Rejected(external_error(4)))?;
        pack_planar_xrgb_words(output, &mut target, damage)
            .map_err(|_| SubmitFailure::Rejected(external_error(4)))?;
        let pack_us = elapsed_micros(pack_started);
        if self.profile {
            eprintln!(
                "axui profile: lower {:.2} ms, prepare {:.2} ms ({}), execute {:.2} ms, pack {:.2} ms, {:?}; {}/{} prepared, {} CPU program(s) resident",
                lower_us as f64 / 1_000.0,
                prepare_us as f64 / 1_000.0,
                if cache_hit { "cache hit" } else { "cache miss" },
                execute_us as f64 / 1_000.0,
                pack_us as f64 / 1_000.0,
                output_format,
                self.prepared.len(),
                PREPARED_CACHE_LIMIT,
                self.runtime.cpu.provider().live_programs(),
            );
        }

        let device = DeviceEpoch::new(self.next_device_epoch).unwrap();
        self.next_device_epoch = self.next_device_epoch.saturating_add(1).max(1);
        Ok(HostSubmission {
            tracker: SubmissionTracker::new(request.surface_epoch()),
            device,
        })
    }

    fn poll_submission(
        &mut self,
        submission: &mut Self::Submission,
    ) -> Result<SubmissionState, GraphicsError> {
        match submission.tracker.state() {
            SubmissionState::Pending => submission
                .tracker
                .mark_device_complete(submission.device)
                .map_err(|_| GraphicsError::InvalidArgument),
            SubmissionState::DeviceComplete(device) => submission
                .tracker
                .mark_visible(
                    device,
                    VisibilityEpoch::new(submission.tracker.surface_epoch().get()).unwrap(),
                )
                .map_err(|_| GraphicsError::InvalidArgument),
            terminal => Ok(terminal),
        }
    }

    fn cancel_submission(
        &mut self,
        submission: &mut Self::Submission,
    ) -> Result<SubmissionState, GraphicsError> {
        if submission.tracker.state().is_terminal() {
            return Ok(submission.tracker.state());
        }
        submission
            .tracker
            .mark_cancelled(TargetReuse::Safe)
            .map_err(|_| GraphicsError::InvalidArgument)
    }

    fn release_submission(
        &mut self,
        submission: Self::Submission,
    ) -> Result<(), ReleaseFailure<Self::Submission>> {
        if submission.tracker.state().is_terminal() {
            Ok(())
        } else {
            Err(ReleaseFailure::Rejected {
                error: GraphicsError::Busy,
                submission,
            })
        }
    }
}

fn lowering_error(error: AxuiGraphError) -> GraphicsError {
    match error {
        AxuiGraphError::InvalidGeometry => GraphicsError::InvalidArgument,
        AxuiGraphError::CoordinateOverflow | AxuiGraphError::SceneCapacity => {
            GraphicsError::ResourceLimit
        }
        AxuiGraphError::UnsupportedCommand { .. }
        | AxuiGraphError::UnsupportedPaint { .. }
        | AxuiGraphError::UnsupportedBorder { .. }
        | AxuiGraphError::UnsupportedShadow { .. }
        | AxuiGraphError::NonUniformCornerRadii { .. }
        | AxuiGraphError::UnsnappedQuad { .. } => GraphicsError::Unsupported,
        AxuiGraphError::Graph(_) => external_error(1),
    }
}

fn external_error(code: i64) -> GraphicsError {
    GraphicsError::External {
        domain: AXRD_ERROR_DOMAIN,
        code,
    }
}

fn shader_background(width: u32, height: u32) -> GraphBackground {
    let width = width as f32;
    let height = height as f32;
    let field_height = height.max(32.0);
    GraphBackground::Radial {
        base: 0x000c_1210,
        lights: [
            GraphRadialLight {
                x: width * 0.18,
                y: height * 0.12,
                radius: field_height * 0.78,
                color: 0x0032_5b48,
                strength: 0.72,
                profile: GraphRadialProfile::Rational { sharpness: 1.5 },
                rim: None,
            },
            GraphRadialLight {
                x: width * 0.72,
                y: height * 0.48,
                radius: field_height * 0.29,
                color: 0x00ee_8663,
                strength: 0.90,
                profile: GraphRadialProfile::Disc { softness: 1.35 },
                rim: Some(GraphRadialRim {
                    strength: 0.14,
                    sharpness: 38.0,
                }),
            },
            GraphRadialLight {
                x: width * 0.62,
                y: height * 0.36,
                radius: field_height * 0.31,
                color: 0x00ff_d8b4,
                strength: 0.48,
                profile: GraphRadialProfile::Rational { sharpness: 2.0 },
                rim: None,
            },
        ],
    }
}

fn fill(renderer: &mut axui_iced::Renderer<System>, bounds: [f32; 4], radius: f32, color: Color) {
    renderer.fill_quad(
        renderer::Quad {
            bounds: Rectangle::new(
                Point::new(bounds[0], bounds[1]),
                Size::new(bounds[2], bounds[3]),
            ),
            border: Border::default().rounded(radius),
            shadow: Shadow::default(),
            snap: true,
        },
        color,
    );
}

fn record_ui(
    renderer: &mut axui_iced::Renderer<System>,
    physical_width: usize,
    physical_height: usize,
    alternate: bool,
) -> Frame<System> {
    let scale = configured_scale();
    let width = physical_width as f32 / scale;
    let height = physical_height as f32 / scale;
    renderer.hint(IcedScale {
        window: scale,
        application: 1.0,
    });
    renderer.reset(Rectangle::new(Point::ORIGIN, Size::new(width, height)));

    let margin = (width * 0.055).clamp(24.0, 62.0);
    let content_width = width - margin * 2.0;
    let ink = if alternate {
        Color::from_rgba8(129, 178, 111, 1.0)
    } else {
        Color::from_rgba8(240, 139, 98, 1.0)
    };
    let paper = Color::from_rgba8(255, 235, 211, 1.0);
    let glass = Color::from_rgba8(18, 27, 23, 0.76);
    let glass_soft = Color::from_rgba8(28, 39, 33, 0.60);
    let muted = Color::from_rgba8(133, 151, 134, 0.54);

    if width < 440.0 || height < 300.0 {
        let compact_margin = width.min(height).mul_add(0.08, 0.0).max(8.0);
        let panel_width = (width - compact_margin * 2.0).max(1.0);
        let panel_height = (height - compact_margin * 2.0).max(1.0);
        fill(
            renderer,
            [compact_margin, compact_margin, panel_width, panel_height],
            panel_height.min(panel_width) * 0.08,
            glass,
        );
        fill(
            renderer,
            [
                compact_margin + panel_width * 0.08,
                compact_margin + panel_height * 0.12,
                panel_width * 0.52,
                (panel_height * 0.08).max(1.0),
            ],
            8.0,
            paper,
        );
        fill(
            renderer,
            [
                compact_margin + panel_width * 0.08,
                compact_margin + panel_height * 0.28,
                panel_width * 0.32,
                panel_height * 0.46,
            ],
            14.0,
            glass_soft,
        );
        fill(
            renderer,
            [
                compact_margin + panel_width * 0.46,
                compact_margin + panel_height * 0.22,
                panel_width * 0.43,
                panel_height * 0.55,
            ],
            18.0,
            Color::from_rgba8(18, 25, 21, 0.78),
        );
        fill(
            renderer,
            [
                compact_margin + panel_width * 0.58,
                compact_margin + panel_height * 0.67,
                panel_width * 0.20,
                (panel_height * 0.08).max(1.0),
            ],
            8.0,
            ink,
        );
        return renderer.take_frame().expect("Iced emitted balanced groups");
    }

    // A deliberately abstract widget surface: every shape below is emitted
    // through Iced's renderer contract, not through an axrd scene shortcut.
    fill(renderer, [margin, margin, content_width, 48.0], 16.0, glass);
    fill(
        renderer,
        [margin + 16.0, margin + 14.0, 20.0, 20.0],
        10.0,
        ink,
    );
    fill(
        renderer,
        [margin + 48.0, margin + 18.0, 108.0, 12.0],
        6.0,
        paper,
    );
    for index in 0..3 {
        fill(
            renderer,
            [
                width - margin - 158.0 + index as f32 * 44.0,
                margin + 16.0,
                31.0,
                16.0,
            ],
            8.0,
            if index == usize::from(alternate) {
                ink
            } else {
                muted
            },
        );
    }

    let body_top = margin + 78.0;
    let body_height = height - body_top - margin;
    let left_width = content_width * 0.38;
    fill(
        renderer,
        [margin, body_top, left_width, body_height],
        24.0,
        glass,
    );
    fill(
        renderer,
        [margin + 28.0, body_top + 32.0, left_width * 0.62, 18.0],
        9.0,
        paper,
    );
    fill(
        renderer,
        [margin + 28.0, body_top + 62.0, left_width * 0.82, 9.0],
        4.5,
        muted,
    );
    fill(
        renderer,
        [margin + 28.0, body_top + 79.0, left_width * 0.67, 9.0],
        4.5,
        muted,
    );
    let chip_y = body_top + 116.0;
    fill(renderer, [margin + 28.0, chip_y, 82.0, 28.0], 14.0, ink);
    fill(
        renderer,
        [margin + 120.0, chip_y, 96.0, 28.0],
        14.0,
        glass_soft,
    );
    let row_y = body_top + body_height - 126.0;
    for index in 0..3 {
        let y = row_y + index as f32 * 33.0;
        fill(renderer, [margin + 28.0, y, 18.0, 18.0], 9.0, muted);
        fill(
            renderer,
            [
                margin + 58.0,
                y + 3.0,
                left_width * (0.48 + index as f32 * 0.08),
                7.0,
            ],
            3.5,
            if index == 1 { ink } else { muted },
        );
    }

    let stage_x = margin + left_width + 22.0;
    let stage_width = width - margin - stage_x;
    // Deep stage, mid plane, and shallow floating controls overlap the `.ax`
    // radial field to make the layer ordering immediately visible.
    fill(
        renderer,
        [stage_x, body_top, stage_width, body_height],
        28.0,
        Color::from_rgba8(18, 25, 21, 0.28),
    );
    fill(
        renderer,
        [
            stage_x + stage_width * 0.08,
            body_top + body_height * 0.14,
            stage_width * 0.48,
            body_height * 0.34,
        ],
        22.0,
        Color::from_rgba8(37, 52, 43, 0.73),
    );
    fill(
        renderer,
        [
            stage_x + stage_width * 0.46,
            body_top + body_height * 0.08,
            stage_width * 0.43,
            body_height * 0.23,
        ],
        20.0,
        Color::from_rgba8(16, 22, 19, 0.88),
    );
    fill(
        renderer,
        [
            stage_x + stage_width * 0.57,
            body_top + body_height * 0.38,
            stage_width * 0.32,
            body_height * 0.42,
        ],
        24.0,
        Color::from_rgba8(20, 29, 24, 0.86),
    );
    let chart_x = stage_x + stage_width * 0.62;
    let chart_bottom = body_top + body_height * 0.72;
    for (index, ratio) in [0.28_f32, 0.52, 0.41, 0.74, 0.60].into_iter().enumerate() {
        let bar_height = body_height * 0.28 * ratio;
        fill(
            renderer,
            [
                chart_x + index as f32 * stage_width * 0.047,
                chart_bottom - bar_height,
                stage_width * 0.026,
                bar_height,
            ],
            5.0,
            if index == 3 { ink } else { muted },
        );
    }
    fill(
        renderer,
        [
            stage_x + stage_width * 0.12,
            body_top + body_height * 0.67,
            stage_width * 0.37,
            54.0,
        ],
        18.0,
        glass,
    );
    fill(
        renderer,
        [
            stage_x + stage_width * 0.15,
            body_top + body_height * 0.71,
            26.0,
            26.0,
        ],
        13.0,
        ink,
    );
    fill(
        renderer,
        [
            stage_x + stage_width * 0.15 + 38.0,
            body_top + body_height * 0.72,
            stage_width * 0.19,
            8.0,
        ],
        4.0,
        paper,
    );

    renderer.take_frame().expect("Iced emitted balanced groups")
}

fn surface_for(target: &HostTarget) -> (Extent2d, SurfaceInfo) {
    let extent = Extent2d::new(target.width as u32, target.height as u32).unwrap();
    let row_bytes = u64::from(extent.width()) * 4;
    let bytes = row_bytes * u64::from(extent.height());
    let surface = SurfaceInfo::linear_xrgb8888(
        extent,
        ByteRange::new(0, bytes).unwrap(),
        row_bytes,
        SurfaceUsage::RENDER_TARGET.union(SurfaceUsage::PRESENT),
    )
    .unwrap();
    (extent, surface)
}

fn render(
    renderer: &mut axui_iced::Renderer<System>,
    backend: &mut HostGraphicsBackend,
    target: &mut HostTarget,
    epoch_value: u64,
    alternate: bool,
) -> SurfaceRetirement {
    let frame = record_ui(renderer, target.width, target.height, alternate);
    let (extent, surface) = surface_for(target);
    let epoch = SurfaceEpoch::new(epoch_value).expect("render epochs start at one");
    let support = backend
        .target_support(target, surface)
        .expect("host XRGB target must be supported");
    let request = RenderRequest::new(
        frame.view(),
        target,
        surface,
        epoch,
        PixelRect::full(extent),
        Damage::Full,
    );
    let started = Instant::now();
    let mut submission = backend
        .submit(request)
        .unwrap_or_else(|failure| panic!("axui submission failed: {failure:?}"));
    let mut retirement = SurfaceRetirement::new(epoch, support.retain_until);
    loop {
        let state = backend
            .poll_submission(&mut submission)
            .expect("axui provider polling failed");
        retirement.observe(state).unwrap();
        if state.is_terminal() {
            assert!(matches!(state, SubmissionState::Visible { .. }));
            break;
        }
    }
    retirement.mark_committed().unwrap();
    backend
        .release_submission(submission)
        .expect("visible host submission must release");
    retirement.mark_submission_released().unwrap();
    eprintln!(
        "axui: {} Iced command(s) → .ax → {}x{} staged XRGB in {:.2} ms",
        frame.commands().len(),
        target.width,
        target.height,
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    retirement
}

fn write_ppm(path: &std::path::Path, target: &HostTarget) {
    let header = format!("P6\n{} {}\n255\n", target.width, target.height);
    let mut bytes = Vec::with_capacity(header.len() + target.pixels.len() * 3);
    bytes.extend_from_slice(header.as_bytes());
    for &pixel in &target.pixels {
        bytes.extend_from_slice(&[(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8]);
    }
    std::fs::write(path, bytes).expect("failed to write axui PPM preview");
}

fn configured_size() -> (usize, usize) {
    let Some(value) = std::env::var_os("AXRD_AXUI_SIZE") else {
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT);
    };
    let value = value.to_string_lossy();
    let (width, height) = value
        .split_once('x')
        .or_else(|| value.split_once('X'))
        .expect("AXRD_AXUI_SIZE must be WIDTHxHEIGHT");
    let size = (
        width.parse::<usize>().expect("invalid axui width"),
        height.parse::<usize>().expect("invalid axui height"),
    );
    assert!(
        size.0 > 0 && size.1 > 0 && size.0 <= u16::MAX as usize && size.1 <= u16::MAX as usize,
        "AXRD_AXUI_SIZE exceeds the current axrd scene ABI"
    );
    size
}

fn configured_scale() -> f32 {
    let scale = std::env::var("AXRD_AXUI_SCALE")
        .ok()
        .map(|value| value.parse::<f32>().expect("invalid AXRD_AXUI_SCALE"))
        .unwrap_or(1.0);
    assert!(
        scale.is_finite() && scale > 0.0,
        "AXRD_AXUI_SCALE must be finite and positive"
    );
    scale
}

fn elapsed_micros(epoch: Instant) -> u64 {
    u64::try_from(epoch.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn host_cursor_icon(icon: AxCursorIcon) -> WinitCursorIcon {
    match icon {
        AxCursorIcon::Default | AxCursorIcon::Custom(_) => WinitCursorIcon::Default,
        AxCursorIcon::ContextMenu => WinitCursorIcon::ContextMenu,
        AxCursorIcon::Help => WinitCursorIcon::Help,
        AxCursorIcon::Pointer => WinitCursorIcon::Pointer,
        AxCursorIcon::Progress => WinitCursorIcon::Progress,
        AxCursorIcon::Wait => WinitCursorIcon::Wait,
        AxCursorIcon::Cell => WinitCursorIcon::Cell,
        AxCursorIcon::Crosshair => WinitCursorIcon::Crosshair,
        AxCursorIcon::Text => WinitCursorIcon::Text,
        AxCursorIcon::Alias => WinitCursorIcon::Alias,
        AxCursorIcon::Copy => WinitCursorIcon::Copy,
        AxCursorIcon::Move => WinitCursorIcon::Move,
        AxCursorIcon::NoDrop => WinitCursorIcon::NoDrop,
        AxCursorIcon::NotAllowed => WinitCursorIcon::NotAllowed,
        AxCursorIcon::Grab => WinitCursorIcon::Grab,
        AxCursorIcon::Grabbing => WinitCursorIcon::Grabbing,
        AxCursorIcon::ResizeHorizontal => WinitCursorIcon::EwResize,
        AxCursorIcon::ResizeVertical => WinitCursorIcon::NsResize,
        AxCursorIcon::ResizeDiagonalUp => WinitCursorIcon::NeswResize,
        AxCursorIcon::ResizeDiagonalDown => WinitCursorIcon::NwseResize,
        AxCursorIcon::ResizeColumn => WinitCursorIcon::ColResize,
        AxCursorIcon::ResizeRow => WinitCursorIcon::RowResize,
        AxCursorIcon::AllScroll => WinitCursorIcon::AllScroll,
        AxCursorIcon::ZoomIn => WinitCursorIcon::ZoomIn,
        AxCursorIcon::ZoomOut => WinitCursorIcon::ZoomOut,
    }
}

fn demo_cursor_interaction(
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> iced_core::mouse::Interaction {
    let width = width as f64;
    let height = height as f64;
    if x >= width * 0.72 && y <= height * 0.20 {
        iced_core::mouse::Interaction::Pointer
    } else if x >= width * 0.43 && y >= height * 0.24 {
        iced_core::mouse::Interaction::Grab
    } else {
        iced_core::mouse::Interaction::Idle
    }
}

struct InteractiveApp {
    context: SoftContext<OwnedDisplayHandle>,
    surface: Option<SoftSurface<OwnedDisplayHandle, Rc<WinitWindow>>>,
    renderer: axui_iced::Renderer<System>,
    backend: HostGraphicsBackend,
    /// Last exact graph-rendered frame; resize previews always resample this
    /// stable source rather than repeatedly filtering an earlier preview.
    target: HostTarget,
    presentation: HostTarget,
    retirement: Option<SurfaceRetirement>,
    epoch: u64,
    alternate: bool,
    pending_size: Option<(usize, usize)>,
    has_exact_frame: bool,
    needs_exact_render: bool,
    needs_present: bool,
    exact_render_due: Option<Instant>,
    ppm_path: Option<std::path::PathBuf>,
    cursor: CursorRequest,
    seat: Seat,
}

impl InteractiveApp {
    fn apply_cursor(&mut self, window: &WinitWindow, request: CursorRequest) {
        if request == self.cursor {
            return;
        }
        window.set_cursor(host_cursor_icon(request.icon));
        window.set_cursor_visible(request.visible);
        self.cursor = request;
    }

    fn request_size(&mut self, width: u32, height: u32) {
        let size = (width as usize, height as usize);
        if width > 0
            && height > 0
            && width <= u16::MAX.into()
            && height <= u16::MAX.into()
            && (size != (self.presentation.width, self.presentation.height)
                || self.pending_size.is_some())
        {
            // Present a filtered view of the last exact frame immediately,
            // then prepare only the newest settled extent. This keeps native
            // resize responsive without compiling one provider program per
            // intermediate window callback.
            self.pending_size = Some(size);
            self.seat
                .set_viewport(InputViewport::new(width, height).unwrap());
            self.needs_exact_render = true;
            self.needs_present = true;
            self.exact_render_due = Some(Instant::now() + RESIZE_SETTLE);
        }
    }

    fn redraw(&mut self) {
        if let Some((width, height)) = self.pending_size.take() {
            let preview_started = Instant::now();
            self.presentation = self.target.resampled(width, height);
            self.needs_present = true;
            if self.backend.profile {
                eprintln!(
                    "axui profile: resize preview {}x{} in {:.2} ms",
                    width,
                    height,
                    preview_started.elapsed().as_secs_f64() * 1_000.0,
                );
            }
        }
        let exact_due = self.needs_exact_render
            && (!self.has_exact_frame
                || self
                    .exact_render_due
                    .is_none_or(|deadline| Instant::now() >= deadline));
        if exact_due {
            self.target = HostTarget::new(self.presentation.width, self.presentation.height);
            self.epoch = self.epoch.saturating_add(1);
            self.retirement = Some(render(
                &mut self.renderer,
                &mut self.backend,
                &mut self.target,
                self.epoch,
                self.alternate,
            ));
            self.presentation = self.target.clone();
            self.has_exact_frame = true;
            self.needs_exact_render = false;
            self.needs_present = true;
            self.exact_render_due = None;
            if let Some(path) = self.ppm_path.take() {
                write_ppm(&path, &self.target);
            }
        }

        if !self.needs_present {
            return;
        }

        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        let width = NonZeroU32::new(self.presentation.width as u32).unwrap();
        let height = NonZeroU32::new(self.presentation.height as u32).unwrap();
        surface
            .resize(width, height)
            .expect("failed to resize the axui host surface");
        surface.window().pre_present_notify();
        let mut buffer = surface
            .buffer_mut()
            .expect("failed to acquire the axui host surface");
        buffer.copy_from_slice(&self.presentation.pixels);
        buffer.present().expect("failed to present the axui frame");
        self.needs_present = false;

        if let Some(mut retirement) = self.retirement.take() {
            retirement
                .mark_presented(SurfaceEpoch::new(self.epoch).unwrap())
                .unwrap();
            assert!(retirement.is_reusable());
        }
    }
}

impl ApplicationHandler for InteractiveApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.surface.is_some() {
            return;
        }
        let attributes = WinitWindow::default_attributes()
            .with_title("axui · Iced → axgfx → axrd → .ax")
            .with_inner_size(PhysicalSize::new(
                self.target.width as u32,
                self.target.height as u32,
            ));
        let window = Rc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create the axui example window"),
        );
        let surface = SoftSurface::new(&self.context, Rc::clone(&window))
            .expect("failed to create the axui host surface");
        self.surface = Some(surface);
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self
            .surface
            .as_ref()
            .map(SoftSurface::window)
            .map(Rc::clone)
        else {
            return;
        };
        if window.id() != window_id {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) => {
                        event_loop.exit();
                    }
                    winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) => {
                        self.alternate = !self.alternate;
                        self.needs_exact_render = true;
                        self.exact_render_due = None;
                        window.request_redraw();
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => {
                self.request_size(size.width, size.height);
                window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                self.request_size(size.width, size.height);
                window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let size = window.inner_size();
                let range = AbsoluteRange::new(
                    size.width.saturating_sub(1).max(1),
                    size.height.saturating_sub(1).max(1),
                )
                .unwrap();
                let pointer = self
                    .seat
                    .feed(PointerInput {
                        seat: SeatId(0),
                        device: DeviceId(0),
                        source: PointerSource::Mouse,
                        timestamp: None,
                        kind: PointerInputKind::AbsoluteMotion {
                            x: position.x.max(0.0) as u32,
                            y: position.y.max(0.0) as u32,
                            range,
                        },
                    })
                    .expect("Winit pointer belongs to the host seat")
                    .position;
                let interaction = demo_cursor_interaction(
                    size.width as usize,
                    size.height as usize,
                    f64::from(pointer.x),
                    f64::from(pointer.y),
                );
                self.apply_cursor(&window, axui_iced::cursor_request(interaction));
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(surface) = self.surface.as_ref() else {
            return;
        };
        if self.needs_present {
            surface.window().request_redraw();
        }
        if self.needs_exact_render {
            match self.exact_render_due {
                Some(deadline) if Instant::now() < deadline => {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                }
                _ => surface.window().request_redraw(),
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

fn main() {
    let size = configured_size();
    let mut renderer = axui_iced::Renderer::new_in(renderer::Settings::default(), System);
    let mut backend = HostGraphicsBackend::new();
    let mut target = HostTarget::new(size.0, size.1);
    if std::env::var_os("AXRD_AXUI_HEADLESS").is_some() {
        let frames = std::env::var("AXRD_AXUI_HEADLESS_FRAMES")
            .ok()
            .map(|value| {
                value
                    .parse::<u64>()
                    .expect("invalid AXRD_AXUI_HEADLESS_FRAMES")
            })
            .unwrap_or(1);
        assert!(frames > 0, "AXRD_AXUI_HEADLESS_FRAMES must be positive");
        for epoch in 1..=frames {
            let retirement = render(&mut renderer, &mut backend, &mut target, epoch, false);
            retirement.mark_presented_for_headless(SurfaceEpoch::new(epoch).unwrap());
        }
        if let Some(path) = std::env::var_os("AXRD_AXUI_PPM") {
            write_ppm(std::path::Path::new(&path), &target);
        }
        return;
    }

    let event_loop = EventLoop::new().expect("failed to create the axui host event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    let context = SoftContext::new(event_loop.owned_display_handle())
        .expect("failed to create the axui software presentation context");
    let presentation = target.clone();
    let mut app = InteractiveApp {
        context,
        surface: None,
        renderer,
        backend,
        target,
        presentation,
        retirement: None,
        epoch: 0,
        alternate: false,
        pending_size: None,
        has_exact_frame: false,
        needs_exact_render: true,
        needs_present: true,
        exact_render_due: None,
        ppm_path: std::env::var_os("AXRD_AXUI_PPM").map(Into::into),
        cursor: CursorRequest::DEFAULT,
        seat: Seat::new(
            SeatId(0),
            InputViewport::new(size.0 as u32, size.1 as u32).unwrap(),
            PointerPosition::default(),
            MotionScale::IDENTITY,
        ),
    };
    event_loop
        .run_app(&mut app)
        .expect("axui host event loop failed");
}

trait HeadlessRetirement {
    fn mark_presented_for_headless(self, epoch: SurfaceEpoch);
}

impl HeadlessRetirement for SurfaceRetirement {
    fn mark_presented_for_headless(mut self, epoch: SurfaceEpoch) {
        self.mark_presented(epoch).unwrap();
        assert!(self.is_reusable());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_preview_resamples_from_the_exact_surface() {
        let source = HostTarget {
            pixels: vec![0xffff_0000, 0xff00_ff00, 0xff00_00ff, 0xffff_ffff],
            width: 2,
            height: 2,
        };

        let preview = source.resampled(4, 4);

        assert_eq!((preview.width, preview.height), (4, 4));
        assert_eq!(preview.pixels[0], source.pixels[0]);
        assert_eq!(preview.pixels[3], source.pixels[1]);
        assert_eq!(preview.pixels[12], source.pixels[2]);
        assert_eq!(preview.pixels[15], source.pixels[3]);
    }

    #[test]
    fn host_cursor_reference_uses_resolved_iced_interaction() {
        assert_eq!(
            demo_cursor_interaction(1000, 600, 900.0, 30.0),
            iced_core::mouse::Interaction::Pointer
        );
        assert_eq!(
            demo_cursor_interaction(1000, 600, 600.0, 300.0),
            iced_core::mouse::Interaction::Grab
        );
        assert_eq!(
            demo_cursor_interaction(1000, 600, 30.0, 300.0),
            iced_core::mouse::Interaction::Idle
        );
    }

    #[test]
    fn prepared_viewport_cache_evicts_provider_resources() {
        let mut renderer = axui_iced::Renderer::new_in(renderer::Settings::default(), System);
        let mut backend = HostGraphicsBackend::new();
        backend.request = BackendRequest::Cpu;

        for (epoch, (width, height)) in [(160, 100), (176, 112), (192, 120)].into_iter().enumerate()
        {
            let mut target = HostTarget::new(width, height);
            let epoch = epoch as u64 + 1;
            let retirement = render(&mut renderer, &mut backend, &mut target, epoch, false);
            retirement.mark_presented_for_headless(SurfaceEpoch::new(epoch).unwrap());
        }

        assert_eq!(backend.prepared.len(), PREPARED_CACHE_LIMIT);
        assert_eq!(backend.runtime.cpu.provider().live_programs(), 2);
        assert_eq!(backend.runtime.cpu.provider().live_buffers(), 6);

        let allocations = backend.runtime.cpu.provider().buffer_allocations();
        let mut target = HostTarget::new(192, 120);
        let retirement = render(&mut renderer, &mut backend, &mut target, 4, false);
        retirement.mark_presented_for_headless(SurfaceEpoch::new(4).unwrap());
        assert_eq!(
            backend.runtime.cpu.provider().buffer_allocations(),
            allocations,
            "an exact display-list hit allocated new provider I/O"
        );
    }

    #[test]
    fn structural_scene_signature_tracks_operator_and_tensor_abi() {
        let mut renderer = axui_iced::Renderer::new_in(renderer::Settings::default(), System);
        let full_a = record_ui(&mut renderer, 960, 600, false);
        let full_a = compile_frame_graph_with_output_in::<_, MAX_PRIMITIVES>(
            full_a.view(),
            shader_background(960, 600),
            GraphOutputFormat::PlanarF32,
            System,
        )
        .unwrap();
        let full_b = record_ui(&mut renderer, 1200, 760, true);
        let full_b = compile_frame_graph_with_output_in::<_, MAX_PRIMITIVES>(
            full_b.view(),
            shader_background(1200, 760),
            GraphOutputFormat::PlanarF32,
            System,
        )
        .unwrap();
        let compact = record_ui(&mut renderer, 320, 240, false);
        let compact = compile_frame_graph_with_output_in::<_, MAX_PRIMITIVES>(
            compact.view(),
            shader_background(320, 240),
            GraphOutputFormat::PlanarF32,
            System,
        )
        .unwrap();
        let byte_output = record_ui(&mut renderer, 960, 600, false);
        let byte_output = compile_frame_graph_with_output_in::<_, MAX_PRIMITIVES>(
            byte_output.view(),
            shader_background(960, 600),
            GraphOutputFormat::PlanarU8,
            System,
        )
        .unwrap();

        assert_eq!(
            structural_graph_signature(full_a.graph()).unwrap(),
            structural_graph_signature(full_b.graph()).unwrap()
        );
        assert_eq!(
            structural_graph_signature(full_a.graph()).unwrap(),
            structural_graph_signature(compact.graph()).unwrap()
        );
        assert_ne!(
            structural_graph_signature(full_a.graph()).unwrap(),
            structural_graph_signature(byte_output.graph()).unwrap()
        );
    }
}
