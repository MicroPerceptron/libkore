//! **The MT7925 composition** — the class binding over `kore-mt7925`, and
//! what a part at **D0** offers the row's generic device host.
//!
//! This is the sibling of [`wifi`](super::wifi), and the difference between
//! them is the whole point: `Rtw89NetBinding` binds a radio at **D1**, with
//! firmware loaded and answering. This one binds a part at **D0** — it
//! answered with its own identity through the L1 remap window and handed the
//! driver ownership, and nothing more. No firmware has moved.
//!
//! **That lesser standing is in the types, not in a comment.** A bound
//! MT7925 is not a radio for witness purposes: [`MetalHost::is_wifi`] stays
//! false for it, so `net-wifi-fw-alive` cannot be earned by a part that has
//! never seen firmware. Its own witness is `net-mt7925-d0`, and it reports
//! the revision rather than a bit, because the number is what a photograph
//! of the panel has to show on a machine with no serial port.
//!
//! [`MetalHost::is_wifi`]: super::MetalHost::is_wifi
//!
//! Registers are in **BAR0** — `res/MT7925_EVIDENCE.md` §12, from
//! `pcim_iomap_regions(pdev, BIT(0), …)` at `mt7925/pci.c:629`. BAR2 is
//! never mapped by the vendor driver and is not mapped here.

use kore_driver::host::{
    CapBuf, ClassQuiesce, DeviceBinding, DeviceResources, DmaActive, DmaPrepared, InterruptBlock,
    InterruptPolicy, PciBindingContract,
};
use kore_driver::seam::{Dma, DmaPool, DmaShared};
use kore_mt7925::dma::Queue;
use kore_mt7925::{Error as Mt7925Error, Mt7925, Product, Waits};

use super::{NetDeviceHost, NicBound, NoFrames};

/// The BAR the MT7925's registers live in.
///
/// Named with its part rather than as a bare `REGS_BAR` because this module
/// and [`wifi`](super::wifi) are both re-exported from `metal`, and two
/// parts' BAR numbers are exactly the kind of thing that must not silently
/// resolve to whichever was imported last.
///
/// BAR0, 2 MiB — and the size is not incidental: `__mt7925_reg_addr` returns
/// a chip-space address unchanged below `0x200000`, so the direct-addressing
/// limit and the BAR's span are the same number. BAR2 (32 KiB) is never
/// mapped by the vendor driver.
pub const MT7925_REGS_BAR: u8 = 0;

/// The wait tier's hooks: the boot clock and the relaxation every other
/// metal core probes with. Identical to the RTL8922AE's, deliberately.
const WAITS: Waits = Waits {
    clock: kore_driver::host::clock_now_ns,
    pause: kore_driver::host::relax,
};

/// The part identified and owned, before the resources are handed over.
pub struct Mt7925NetPrepared {
    /// The ROM patch's bytes, when the boot carried them and they are the
    /// pinned ones. Carried from `prepare` because `start` is where the
    /// rings exist and the config is gone by then.
    patch: Option<&'static [u8]>,
    /// The N9 RAM code's bytes, on the same terms as [`Self::patch`] —
    /// carried only when the verdict is `Pinned`, which is a verdict over
    /// **both** blobs, so neither can reach the part alone.
    ram: Option<&'static [u8]>,
    /// What the boot carried, decided before DMA was activated.
    firmware: FirmwareVerdict,
    part: Mt7925,
}

/// How far this row drives the part at bind time.
///
/// **Why a switch exists at all.** The net row founds at manifest position
/// ten of eighteen, *before* the compositor, and the only machine that
/// carries this part has no serial port. A bind that hangs or faults here
/// therefore produces no output of any kind: Limine boots instantly and
/// quietly (`timeout: 0`, `quiet: yes`), kore never reaches the screen, and
/// a fault resets the machine into whatever the firmware's boot order names
/// next. "The laptop stopped booting" is the entire diagnostic.
///
/// So each rung above the last one proven on metal is enabled deliberately,
/// one per flash, and the default is the last rung that **is** proven.
/// D1a — the ring latched — returned `net-mt7925-d1-ring PASS` on
/// 2026-11-09; D1b and D1c were written and flashed together, which is the
/// mistake this constant exists to stop repeating.
///
/// Raise it by one, flash, read `status services`, and only then raise it
/// again.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Reach {
    /// D1a only: program the download ring and read it back. **Proven.**
    Ring,
    /// D1b: also fill the event ring and start the WFDMA engine.
    Engine,
    /// D1c: also send one MCU command and read its answer.
    Mcu,
    /// **C2b** — the ROM patch crosses, or the part says it already has it.
    /// **Proven on metal 2026-11-12** (`net-mt7925-patch-sent PASS`).
    Patch,
    /// **C2c, first half** — the N9 RAM code crosses: four regions,
    /// 1,087,744 bytes, 267 chunks. **Proven on metal 2026-11-12**
    /// (`net-mt7925-ram-sent PASS`), at the time behind an ownership round
    /// trip the vendor does not make over PCIe; that step is gone (see
    /// `Phy`'s fourth photograph).
    ///
    /// Separated from [`Self::Start`] because 267 chunks lap the 128-slot
    /// download ring **twice** where the patch's 46 never lapped it once.
    /// `send_payload`'s lapping wait and the staging slab's slot reuse
    /// have never run on hardware, and a boot that changed the ring's
    /// wrap *and* started firmware could not say which one spoke.
    Ram,
    /// **C2c, second half** — wait for `MT_CONN_ON_MISC` to report the N9
    /// core powered and up. **Proven on metal 2026-11-12**
    /// (`net-wifi-fw-alive PASS`).
    ///
    /// `FW_START_REQ` is answered inside [`Self::Ram`]; this is the part's
    /// own word that the code it was handed actually executes, which is a
    /// strictly larger claim and the only one `net-wifi-fw-alive` makes.
    Start,
    /// **M1a** — the same ladder, with the exchange's reply triage armed.
    ///
    /// **This rung adds nothing, and that is the point.** It sends no new
    /// command and reaches no new register; it only makes an exchange
    /// step over a reply that is not its own instead of taking the next
    /// done descriptor. So the boot that carries it must show the *same*
    /// seven witnesses as the boot before — a rung that can only regress
    /// is the cheapest thing on this ladder to attribute, and the
    /// triage's first appearance on hardware is worth that cheapness.
    ///
    /// **Proven on metal 2026-11-14**: the same seven witnesses, no note,
    /// which is exactly what it had to show.
    Triage,
    /// **M1b** — the part describes itself under the unified command
    /// class, and a block of its own efuse comes back.
    ///
    /// **Refused on metal 2026-11-18**: note 86, `[raw 0x71656]` — the
    /// capability query, `NoAnswer`, with the same seven witnesses beneath
    /// it. The command was checked against upstream field by field
    /// (descriptor, `option`, sequence, queue, body, reply triage) and
    /// matches; the budget is 20 s against upstream's 3 s. That note could
    /// not say *where* the silence was, so the transport evidence word now
    /// carries the cursors that do — see `transport_evidence`.
    ///
    /// **Second photograph, 2026-11-18**: `[raw 0xd0d71656]` — `tx_idx`
    /// 13 = `tx_next` 13, nothing past the event cursor, nothing skipped.
    /// The command left the host and the part wrote nothing back. The
    /// detail byte now carries the firmware's own state at that moment
    /// (`kore_mt7925::d2::FirmwareState`): dead, alarmed, or running and
    /// silent.
    ///
    /// **Third photograph, 2026-11-19**: `[raw 0xd0d78156]` — detail
    /// `0x81`, the N9 still ready and no error flag raised. Running and
    /// silent. The one structural difference left against upstream was
    /// receive ring 2, which upstream programs before the download and
    /// this driver did not; it is now programmed (32 slots, not 1,536 —
    /// `kore_mt7925::dma::EVENT_DATA_DEPTH`) and the exchange watches it.
    /// The evidence word's top nibble says what ring 2 held if the part
    /// is still silent (`transport_evidence`).
    ///
    /// **Fourth photograph, 2026-11-19**: `[raw 0x800000d78156]` — ring 2
    /// programmed and empty (top nibble 8). The part wrote to neither
    /// ring. A capture of `mt7925e` loading on this machine the same day
    /// shows the vendor's first post-`N9_RDY` commands answered on ring 0
    /// within ~2.5 ms, with nothing but interrupt bookkeeping in between —
    /// and no ownership handshake between the patch and the RAM code,
    /// which this ladder did. That handshake is removed.
    ///
    /// **Fifth photograph, 2026-11-19**: `[raw 0x800000d78156]` again, every
    /// witness intact — the handshake was neither needed nor the cause.
    /// The capture's write set, diffed against this driver, leaves only
    /// interrupt enables; the ring registers it cannot see were checked
    /// against source instead, and ring 0 was being posted `cpu_idx = 512`
    /// on a 512-slot ring — an index naming no slot, where upstream posts
    /// 511 (`kore_mt7925::dma::rx_fill_index`).
    ///
    /// **Sixth photograph, 2026-11-19**: `[raw 0x800000d78156]` again. The
    /// index fix stays (it is upstream's and the guard is right) but was
    /// not the cause. What remained of the capture's difference was
    /// interrupts, and the architect chose to wire them properly rather
    /// than mimic them: the engine arms the vendor's `0x2e07fff5`, the PCIe
    /// MAC is enabled after the subsystem reset, and every wait from the
    /// engine's start parks on the held route and services the status the
    /// way `mt792x_irq_tasklet` does (`park_on_route`). The evidence word
    /// now says what the route saw (`transport_evidence`).
    Phy,
}

/// What the boot carried for this part, and whether it is what we pinned.
///
/// **A verdict, not a `Result`**, because "no module" and "the wrong
/// module" are different facts about the machine and only one of them is a
/// fault. A boot with no firmware is the ordinary QEMU posture and must not
/// read as tampering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirmwareVerdict {
    /// The boot carried neither module.
    Absent,
    /// The boot carried one and not the other.
    Partial,
    /// Both modules are present and neither is the pinned blob.
    Unpinned,
    /// Both modules are present and both match the pin.
    Pinned,
}

impl FirmwareVerdict {
    /// Both blobs present and both the pinned bytes — the only verdict the
    /// download may run on.
    pub fn is_pinned(self) -> bool {
        matches!(self, Self::Pinned)
    }

    /// Check both modules against their compiled pins.
    pub fn of(patch: Option<&[u8]>, ram: Option<&[u8]>) -> Self {
        use kore_mt7925::fw;
        let (Some(patch), Some(ram)) = (patch, ram) else {
            return match (patch, ram) {
                (None, None) => Self::Absent,
                _ => Self::Partial,
            };
        };
        let ok = fw::pinned(patch, fw::PATCH_PINNED_SIZE, &fw::PATCH_PINNED_DIGEST).is_ok()
            && fw::pinned(ram, fw::RAM_PINNED_SIZE, &fw::RAM_PINNED_DIGEST).is_ok();
        if ok { Self::Pinned } else { Self::Unpinned }
    }
}

/// The rung this image drives to. See [`Reach`].
///
/// **`Triage` for the flash of 2026-11-13**, and this one is deliberately
/// the least interesting boot on the ladder.
///
/// C2 is complete: `net-wifi-fw-alive PASS` on 2026-11-12, the firmware
/// running. What this raises is not a rung the part can refuse — it is
/// the exchange's new reply triage, which steps over an unsolicited event
/// or another sequence's answer instead of taking the next done
/// descriptor, and which M1's unified commands will depend on absolutely.
///
/// **So the boot that carries it must look identical to the one before.**
/// Seven witnesses, no note. There is no eighth to earn here; a rung that
/// can only regress is the cheapest thing on this ladder to attribute,
/// and the triage's first run on hardware is worth spending a boot to
/// isolate. If a witness goes missing, the filter took something it
/// should not have, and the one that vanishes says which.
///
/// `Phy` is the flash after, and it waits on the witness carve having a
/// bit to spare — bit 63 is INIT's, and a const assertion in
/// `kore_rt::svc` refuses the overlap. The ladder's seven ordered flags
/// become a four-bit ordinal first.
const REACH: Reach = Reach::Phy;

/// One page: the command staging buffer.
///
/// A command is a 64-byte descriptor plus a body under sixteen bytes, so a
/// page is far more than this rung needs — but `DmaPool::alloc` is
/// page-granular anyway and the download's `FW_SCATTER` chunk is 4,096,
/// which is what this buffer becomes at C2.
const PAGE_BYTES: usize = 4096;

/// The firmware download's staging slab — one chunk buffer per `Fwdl` slot.
///
/// Sized by the ring, not by the blob, so the slab and the ring wrap
/// together: slot *n* always owns the same bytes, and the only moment a
/// buffer can be reused is the moment its descriptor can be, which is the
/// one `send_payload` waits for.
const FWDL_STAGING_BYTES: usize =
    Queue::Fwdl.depth() as usize * kore_mt7925::mcu::FW_SCATTER_MAX;

/// The event ring's backings, held for as long as the engine names them.
///
/// Both are kept even when bring-up refuses part way: a table whose base
/// the engine has already latched must not be freed before `quiesce` has
/// stopped the engine, and a refusal is exactly when that is easiest to
/// forget.
#[derive(Default)]
struct EventRing {
    table: Option<CapBuf>,
    buffers: Option<CapBuf>,
    /// The command ring's descriptor table.
    mcu_table: Option<CapBuf>,
    /// Where a command's bytes are staged before the engine fetches them.
    mcu_payload: Option<CapBuf>,
    /// The firmware download's staging slab: one `FW_SCATTER_MAX` buffer
    /// per `Fwdl` slot.
    ///
    /// Held here, with the other backings the engine was given addresses
    /// for, because a chunk is published and never acknowledged — the part
    /// may still be fetching the last descriptors when `download_patch`
    /// returns, and freeing these pages then hands them to the next
    /// allocation with hardware pointed at them.
    fwdl_payload: Option<CapBuf>,
    /// **Receive ring 2**'s descriptor table and buffers — see
    /// `kore_mt7925::dma::Queue::EventData`.
    ///
    /// Held here for the reason every backing is: the engine is given
    /// their addresses before it is enabled, and `quiesce` must stop it
    /// before they are freed.
    data_table: Option<CapBuf>,
    /// Ring 2's receive buffers, one `MT_RX_BUF_SIZE` slab per slot.
    data_buffers: Option<CapBuf>,
    /// **The conversation with the MCU**, owned here because it outlives
    /// every caller that holds one.
    ///
    /// `ask_the_part` and `download_patch` each used to build their own
    /// `Rings` with the counters and the sequence at zero. The second one
    /// therefore drove the command ring's producer index *backwards* and
    /// read D1c's two answers back as replies to its own first two
    /// commands — whose sequences matched, because the sequence had
    /// restarted too. C2b reported a semaphore it never took and a
    /// download request the part never saw.
    ///
    /// It starts at zero exactly once, beside `enable_engine`, which is
    /// what zeroes the hardware indices it mirrors.
    conv: kore_mt7925::d1::Conversation,
}

/// **What the interrupt route did while the ladder waited.**
///
/// Accumulated across every park of one rung: how many waits the route
/// actually ended (rather than the slice's deadline), and the OR of every
/// status bit and MCU word [`kore_mt7925::d1::Core::service_interrupt`]
/// acknowledged. The first is whether delivery works on this machine at all;
/// the second is what the part tried to say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IrqLog {
    /// Waits the route ended with `Fired`.
    pub fired: u32,
    /// `MT_WFDMA0_HOST_INT_STA`, every acknowledged bit.
    pub sta: u32,
    /// `MT_MCU_CMD`, every acknowledged word.
    pub mcu_cmd: u32,
}

/// The longest one park lasts before the rings are polled regardless.
///
/// **The route is the preferred edge, the slice only bounds it** — the
/// xHCI readiness proof's lesson (`services/input`): a waiter that trusts
/// delivery alone hangs on a machine where delivery is dead, and one that
/// only spins burns a core. One millisecond is well inside the ~2.5 ms the
/// capture shows between a command and its answer, and costs five MMIO
/// accesses per wake.
const PARK_SLICE_NS: u64 = 1_000_000;

/// **The ladder's waiter, once the engine is armed**: park on the held
/// route for at most [`PARK_SLICE_NS`], then service the interrupt the way
/// `mt792x_irq_tasklet` does — on every wake, fired or not, because a
/// status bit left set cannot raise the next edge.
///
/// `Copy`, because the C2 and M1 entry points hand one waiter to every
/// exchange they make: it captures two shared borrows and records through a
/// `Cell`, so there is nothing for a copy to diverge on.
///
/// A machine with no clock, no route, or a refused park degrades to the
/// probe pause it replaced, still servicing the status.
fn park_on_route<'a>(
    resources: &'a DeviceResources<DmaActive>,
    log: &'a core::cell::Cell<IrqLog>,
) -> impl FnMut() + Copy + 'a {
    move || {
        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            kore_driver::host::relax();
            return;
        };
        let fired = match (resources.route(), kore_driver::host::clock_now_ns()) {
            (Some(route), Some(now)) => {
                match kore_rt::cap::verbs::wait_interrupt_until(
                    route,
                    now.saturating_add(PARK_SLICE_NS),
                ) {
                    Ok(kore_rt::cap::verbs::InterruptWait::Fired) => true,
                    Ok(_) => false,
                    // Nothing parked: pause instead, or the budget's probes
                    // run back to back.
                    Err(_) => {
                        kore_driver::host::relax();
                        false
                    }
                }
            }
            _ => {
                kore_driver::host::relax();
                false
            }
        };
        let seen = kore_mt7925::d0::Core::new(regs).service_interrupt();
        let mut l = log.get();
        l.fired = l.fired.saturating_add(u32::from(fired));
        l.sta |= seen.sta;
        l.mcu_cmd |= seen.mcu_cmd;
        log.set(l);
    }
}

/// Ring 2 as the exchange takes it, or `None` when it was never programmed.
///
/// A free function over the two fields rather than a method on `EventRing`:
/// every caller also lends the exchange `&mut ring.conv`, and a method
/// would borrow the whole ring to do it.
fn data_ring<'b>(
    table: &'b Option<CapBuf>,
    buffers: &'b Option<CapBuf>,
) -> Option<kore_mt7925::d1::DataRing<'b, CapBuf>> {
    match (table.as_ref(), buffers.as_ref()) {
        (Some(table), Some(buffers)) => Some(kore_mt7925::d1::DataRing { table, buffers }),
        _ => None,
    }
}

/// Write one armed receive descriptor into each of `depth` slots of `table`,
/// each naming its own `buf_bytes` slab of `buffers`.
///
/// Returns how many slots were armed; fewer than `depth` means the table or
/// the slab was too small, and the caller refuses. Through the shared seam,
/// word by word: the bounds are checked rather than asserted in a comment,
/// and each word is one volatile store, which is what a table the engine
/// reads on its own schedule requires.
fn arm_receive_ring(table: &CapBuf, buffers: &CapBuf, depth: u16, buf_bytes: usize) -> u16 {
    let slots = DmaShared::new(table);
    let mut wrote = 0u16;
    for slot in 0..depth {
        let iova = buffers.iova().get() + u64::from(slot) * buf_bytes as u64;
        let Some(desc) = kore_mt7925::dma::Descriptor::receive(iova, buf_bytes as u32) else {
            break;
        };
        let base = kore_mt7925::dma::Descriptor::slot_at(usize::from(slot));
        if desc
            .words()
            .into_iter()
            .enumerate()
            .try_for_each(|(word, value)| slots.write_u32_at(base + word * 4, value))
            .is_err()
        {
            break;
        }
        wrote += 1;
    }
    wrote
}

/// The bound owner: a part at D0 and the capability resource set.
pub struct Mt7925NetBound {
    part: Mt7925,
    resources: DeviceResources<DmaActive>,
    /// The firmware-download ring's descriptor table, held for as long as
    /// the part holds its address.
    ///
    /// Kept in the bound owner rather than dropped at the end of `start`:
    /// the engine was given this buffer's device address, so freeing it
    /// while the part still names it would hand the ring's pages to the
    /// next allocation with hardware still pointed at them.
    ring: Option<CapBuf>,
    /// Whether D1a passed — the ring programmed and read back.
    ring_ready: bool,
    /// The event ring's descriptor table and the receive buffers it points
    /// at, held for as long as the engine names them.
    event: EventRing,
    /// Whether the part answered an MCU command — D1c.
    mcu_answered: bool,
    /// The D1c refusal's rung, or zero when the rung was not attempted or
    /// was answered. Published by `entry.rs`; see [`Self::bring_up_rung`].
    mcu_rung: u8,
    /// What the boot's two firmware modules are: both pinned, one
    /// missing, or present and not the pinned bytes.
    firmware: FirmwareVerdict,
    /// What C2b found when it asked, or `None` when the rung was not
    /// driven this boot.
    patch: Option<kore_mt7925::c2::PatchOutcome>,
    /// What C2c sent, or `None` when the rung was not driven this boot.
    ram: Option<kore_mt7925::c2::RamOutcome>,
    /// What M1 read back, or `None` when that rung was not driven.
    phy: Option<kore_mt7925::d2::PhyReport>,
    /// `MT_CONN_ON_MISC` at the moment the N9 reported itself up, or
    /// `None` when that rung was not driven or did not finish.
    ///
    /// **The value, not a flag.** `firmware_ready` already collapses this
    /// register to a yes; keeping what it read means a boot that succeeds
    /// can still say what the part looked like when it did.
    fw_alive: Option<u32>,
    /// One byte of evidence for the rung, in the note's detail bits.
    ///
    /// Zero is reserved for "not instrumented", exactly as the
    /// compositor's step byte reserves it, so an older build prints
    /// "not named" rather than reading as a real finding.
    mcu_detail: u8,
    /// Four bytes of evidence for the rung, in the note's bits 16..=47.
    ///
    /// Zero is "this rung has no numbers", which is true of most of them.
    /// See the packing beside `patch_evidence` in `bind` for what each
    /// byte is.
    mcu_evidence: u32,
    /// Whether the engine was started and its rings filled.
    ///
    /// **Not a witness.** It records what this row did, so `quiesce` knows
    /// it must stop an engine before releasing the backings. The claim that
    /// the engine *ran* belongs to D1c, where a response arrives and moves
    /// `dma_idx` for a reason.
    engine_started: bool,
}

/// What the claim decided from the probed identity.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mt7925Config {
    /// The part the claimed function's PCI identity selected.
    pub product: Product,
    /// The ROM patch module's bytes, when the boot carried one.
    ///
    /// The whole mapped module, page-rounded: the pin's own length is what
    /// the blob is, and `kore_mt7925::fw::pinned` slices to it. `None` is
    /// the ordinary un-provisioned posture, and every QEMU boot without the
    /// vendored submodule.
    pub patch: Option<&'static [u8]>,
    /// The N9 RAM code module's bytes, on the same terms.
    pub ram: Option<&'static [u8]>,
}

/// The class declaration: BAR0, one interrupt block.
///
/// `Required(SINGLE)` — and `res/MT7925_EVIDENCE.md` §10 is why the policy
/// below is not the same statement it is for the Realtek part. The MT7925
/// has **no MSI-X capability at all**, and its MSI capability reads
/// `Count=1/32`: the part offers **32** messages and Linux enables one. So
/// `SINGLE` here is a deliberate narrowing of a 32-vector part, not a match
/// to its limit, and the block's `PreferMsiX` source must admit an MSI-only
/// function or this part does not bind.
pub struct Mt7925NetBinding;

impl PciBindingContract for Mt7925NetBinding {
    const BARS: kore_rt::service::PciBarMask = kore_rt::service::PciBarMask::BAR0;
    const INTERRUPTS: InterruptPolicy = InterruptPolicy::Required(InterruptBlock::SINGLE);
    /// **32**, and this part is the reason the declaration exists.
    ///
    /// The WFDMA descriptor's buffer pointer is a `__le32` (`dma.h:120`),
    /// the vendor driver asks for `DMA_BIT_MASK(32)` on everything that is
    /// not an MT7928 (`mt7925/pci.c:651`), and `ADDR_EXT_EN` is set only for
    /// the MT7927/MT7928 (`mt792x_dma.c:199`, `:305`). Without this line the
    /// host would place the pool at `DMA_IOVA_BASE` — 4 GiB exactly — and
    /// every address this part fetched would be the low half of one it was
    /// never given. `res/MT7925_EVIDENCE.md` §15.
    const DMA_ADDRESS_BITS: u32 = kore_mt7925::dma::DMA_ADDR_BITS;
}

impl DeviceBinding for Mt7925NetBinding {
    type Pci = Self;
    type Config = Mt7925Config;
    type Prepared = Mt7925NetPrepared;
    type Bound = Mt7925NetBound;
    type Error = Mt7925Error;

    /// D0 runs here, before any DMA is activated: it needs none. The
    /// identity read is itself the proof the window and the remap work, so
    /// a part that fails it never reaches `start`.
    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        config: Self::Config,
    ) -> Result<Self::Prepared, Mt7925Error> {
        let regs = resources.bar(MT7925_REGS_BAR).ok_or(Mt7925Error::NotPresent)?;
        let part = Mt7925::bring_up(regs, config.product, WAITS)?;
        // **Checked here, before any DMA is activated**, because the answer
        // is about the boot image and not about the part: a machine that
        // shipped the wrong bytes should say so whether or not the radio
        // later binds, and the check costs two hashes of memory we already
        // hold.
        let firmware = FirmwareVerdict::of(config.patch, config.ram);
        // Only a pinned blob is carried forward. An unverified image must
        // not reach the part, and `None` here is what makes that
        // structural rather than a check the download has to remember.
        let patch = firmware.is_pinned().then_some(config.patch).flatten();
        let ram = firmware.is_pinned().then_some(config.ram).flatten();
        Ok(Mt7925NetPrepared {
            part,
            firmware,
            patch,
            ram,
        })
    }

    /// D1a runs here, where the DMA pool is live: the download ring is
    /// programmed and read back.
    ///
    /// **A refusal here does not fail the bind.** The part is still at D0 —
    /// it answered its identity and handed over ownership — and a row that
    /// can report that honestly is worth more than one that refuses and
    /// reports nothing. The same posture the welcome and the frame face
    /// now take: a rung that may lawfully not be reached does not get to
    /// cost a row that can serve without it. The verdict rides
    /// [`Mt7925NetBound::ring_ready`] out to the witness instead.
    fn start(
        prepared: Self::Prepared,
        resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, Mt7925Error> {
        // **The engine is reset before any ring is programmed.** Upstream
        // runs `mt792x_dma_disable(dev, true)` first (`mt792x_dma.c:95`)
        // and the logic reset at its end throws away ring state, so a
        // descriptor base written before it would not survive. A refusal
        // here means the engine would not go quiet; the ring is still
        // programmed (the logic reset never ran, so nothing was lost) but
        // the engine is not started over a part in an unknown state.
        // Behind the same gate as the engine itself: `Reach::Ring` must stay
        // byte-for-byte the rung metal confirmed, because it is the image
        // this machine is recovered with when a rung above it will not boot.
        // A gate that leaves new hardware behaviour outside itself is not a
        // recovery path.
        // **The Wi-Fi subsystem first, before the engine and before any
        // ring.** Upstream's order (`mt7925_pci_probe`: wfsys reset at
        // `pci.c:742`, DMA init at `:757`, the MCU after that), and the
        // step this ladder was missing when the part would not answer:
        // without it the MCU keeps whatever state it was left in, and a
        // ROM-level question asked of a part that is past the ROM has no
        // reason to be answered.
        //
        // Behind the `Mcu` gate, not run unconditionally, because it is new
        // hardware behaviour and `Reach::Engine` is the image this machine
        // is recovered with. A gate that leaves a subsystem reset outside
        // itself is not a recovery path.
        // **Every step's refusal is kept, not just D1c's.** The first
        // version of this carried only the MCU rung, and the next boot
        // reported `up` with two witnesses and no note at all — because
        // the step that failed was one of the two whose refusal was still
        // being thrown away. A ladder where some rungs can fail silently
        // is a ladder that costs a flash per silent rung.
        let mut rung = 0u8;
        let mut note = |r: kore_mt7925::d1::D1Refusal| {
            if rung == 0 {
                rung = r.rung();
            }
        };
        let wfsys = REACH < Reach::Mcu || {
            match Self::reset_wfsys(&resources) {
                Ok(()) => true,
                Err(r) => {
                    note(r);
                    false
                }
            }
        };
        let engine_reset = wfsys
            && (REACH < Reach::Engine || {
                match Self::reset_engine(&resources) {
                    Ok(()) => true,
                    Err(r) => {
                        note(r);
                        false
                    }
                }
            });
        let (ring, ring_ready) = Self::program_download_ring(&prepared.part, &resources);
        // D1b only when D1a stood: an engine started over a ring the part
        // never latched is the one case where carrying on would be worse
        // than stopping.
        let (mut event, engine_started) = if engine_reset && ring_ready && REACH >= Reach::Engine {
            Self::start_engine(&resources)
        } else {
            (EventRing::default(), false)
        };
        // **Every wait from here parks on the route.** `start_engine` armed
        // the vendor's interrupt mask; from this point the part may raise
        // what upstream's handler would acknowledge, so each wait services
        // it — see [`park_on_route`].
        let irq = core::cell::Cell::new(IrqLog::default());
        let wait = park_on_route(&resources, &irq);
        // D1c. The refusal is **kept**, not reduced to a bool: on a machine
        // with no serial port the rung it carries is the only thing that
        // can say why, and it rides out through the row's note.
        // **The ownership round trip, immediately before the first MCU
        // command.** `mt7925e_mcu_init` hands the part back to its firmware
        // and takes it again (`mt7925/pci_mcu.c`) on the lines before
        // `mt7925_run_firmware`, which issues that command.
        //
        // It is easy to conclude this is unnecessary, and this driver did:
        // `mt7925_pci_probe` takes ownership near its top and resets the
        // Wi-Fi subsystem after, so reading only the probe suggests
        // ownership survives the reset. The re-take lives in another file.
        // D0 took ownership, then we reset the subsystem out from under it
        // — and a part that owns itself again fetches a command and never
        // answers, which is exactly what metal reported.
        let owned = REACH < Reach::Mcu
            || match Self::retake_ownership(&resources) {
                Ok(()) => true,
                Err(r) => {
                    note(r);
                    false
                }
            };
        let mcu = if owned && engine_started && REACH >= Reach::Mcu {
            Some(Self::ask_the_part(&resources, &mut event, wait))
        } else {
            None
        };
        let mcu_answered = matches!(mcu, Some(Ok(())));

        if let Some(Err(refusal)) = mcu {
            note(refusal);
        }
        // **C2b.** Only over a part that just answered, and only with a
        // blob that verified: the download is the first thing this ladder
        // does that changes the part rather than asking it something.
        let mut patch_err = None;
        let patch = if mcu_answered && REACH >= Reach::Patch {
            match Self::download_patch(&resources, &mut event, ring.as_ref(), prepared.patch, wait) {
                Ok(outcome) => Some(outcome),
                Err(refusal) => {
                    if rung == 0 {
                        rung = refusal.rung();
                    }
                    patch_err = Some(refusal);
                    None
                }
            }
        } else {
            None
        };
        // **C2c.** Only over a part that took the patch on this boot or
        // already held it — a RAM download onto a part with no patch is
        // not a rung, it is a different experiment.
        let mut ram_err = None;
        let ram = if patch.is_some() && REACH >= Reach::Ram {
            // **Straight from the patch to the RAM code, with no ownership
            // round trip between them.** This driver used to hand the part
            // back and retake it here, on a misreading: upstream repeats
            // that handshake inside `mt792x_load_firmware` for **SDIO
            // only** (`mt792x_core.c:1121`, `if (mt76_is_sdio(...))`), and
            // the 2026-11-19 capture of `mt7925e` on this machine has no
            // `MT_CONN_ON_LPCTL` access at all between the patch download
            // and `FW_N9_RDY`. Its two handshakes are the probe's and
            // `mt7925e_mcu_init`'s — both before the download, and both
            // already here (D0 and `retake_ownership`).
            //
            // The RAM code crossed and started either way, so this is not
            // what C2c needed. It is the one step this ladder took that the
            // vendor never does, on the boot whose firmware then would not
            // answer its first unified command (M1b, note 86).
            match Self::download_ram(&resources, &mut event, ring.as_ref(), prepared.ram, wait) {
                Ok(outcome) => Some(outcome),
                Err(refusal) => {
                    if rung == 0 {
                        rung = refusal.rung();
                    }
                    ram_err = Some(refusal);
                    None
                }
            }
        } else {
            None
        };
        // The part's own word that the firmware runs — a strictly larger
        // claim than `FW_START_REQ` being answered, and the only one
        // `net-wifi-fw-alive` makes.
        let fw_alive = if ram.is_some() && REACH >= Reach::Start {
            match Self::await_firmware(&resources, wait) {
                Ok(misc) => Some(misc),
                Err(refusal) => {
                    if rung == 0 {
                        rung = refusal.rung();
                    }
                    ram_err = Some(refusal);
                    None
                }
            }
        } else {
            None
        };
        // **M1.** Only over a part whose firmware is running: a
        // configuration query to a part that never started is a question
        // asked of nobody.
        let mut phy_err = None;
        let phy = if fw_alive.is_some() && REACH >= Reach::Phy {
            // M1's evidence carries only what the route did during M1.
            irq.set(IrqLog::default());
            match Self::describe_the_part(&resources, &mut event, ring.as_ref(), wait) {
                Ok(report) => Some(report),
                Err(refusal) => {
                    if rung == 0 {
                        rung = refusal.rung();
                    }
                    phy_err = Some(refusal);
                    None
                }
            }
        } else {
            None
        };
        // A rung that was never reached because the engine never started
        // still owes an explanation, and `EngineBusy` is not it. Say so.
        if REACH >= Reach::Mcu && !engine_started && rung == 0 {
            rung = kore_mt7925::d1::D1Refusal::RxNotTaken { dma_idx: 0, posted: 0 }.rung();
        }
        // One byte that splits what `NoAnswer` could not: whether the
        // engine ever consumed the command descriptor. The value is
        // carried, not judged — `dma_idx`'s meaning on a transmit queue is
        // a belief about this part, and metal is what settles it.
        // A refused section carries its transport rung as the evidence
        // byte, so note 79 says which of the five ways the command failed.
        // **One refusal drives the note, whichever rung produced it.**
        // `rung` already took the first refusal on the ladder; the detail
        // and evidence must come from that same refusal or they describe
        // a different failure than the rung does.
        let c2_err = patch_err.or(ram_err);
        let (mcu_detail, mcu_evidence) =
            bring_up_note_numbers(phy_err.as_ref(), c2_err.as_ref(), mcu.as_ref(), irq.get());
        let mcu_rung = rung;
        Ok(Mt7925NetBound {
            part: prepared.part,
            firmware: prepared.firmware,
            resources,
            ring,
            ring_ready,
            event,
            engine_started,
            mcu_answered,
            mcu_rung,
            mcu_detail,
            mcu_evidence,
            patch,
            ram,
            fw_alive,
            phy,
        })
    }

    /// No engine was started and no interrupt source enabled — but D1a may
    /// have given the part a ring base, so that is taken back.
    ///
    /// Zeroing the descriptor base before the backing is released is the
    /// whole of it: DMA was never enabled (`GLO_CFG` is untouched) and
    /// `cpu_idx` was left at zero, so the engine has nothing in flight to
    /// drain. What it must not keep is the address of pages that are about
    /// to belong to somebody else.
    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, Mt7925Error> {
        let Some(regs) = bound.resources.bar(MT7925_REGS_BAR) else {
            return Ok(ClassQuiesce::Stopped);
        };
        let core = kore_mt7925::d0::Core::new(regs);
        // **No delivery into a route that is being released.** Both
        // sources go before the engine stops, so its last completions
        // raise nothing.
        core.disarm_interrupts();
        // **The engine first, and its quiet observed, before anything else.**
        // D1b handed the part the address of a ring and a slab of receive
        // buffers. Those backings are released when this owner drops, and an
        // engine still walking them writes into whatever is mapped there
        // next — a corruption with no author. The stop is `GLO_CFG_TEARDOWN`
        // and then both busy bits read clear, on the vendor's own 1 ms
        // budget.
        if bound.engine_started || bound.event.table.is_some() {
            if core
                .disable_engine(kore_driver::host::relax, kore_driver::host::clock_now_ns)
                .is_err()
            {
                // The engine did not go quiet. Refusing here is the whole
                // point: the caller must retain this owner rather than free
                // backings the part is still reading.
                return Err(Mt7925Error::NotPresent);
            }
            bound.engine_started = false;
        }
        // Only now are the ring bases taken back.
        for queue in [Queue::Fwdl, Queue::McuWm, Queue::EventWm, Queue::EventData] {
            let bank = queue.regs();
            core.write(bank.desc_base(), 0);
            core.write(bank.ring_size(), 0);
            core.write(bank.cpu_idx(), 0);
        }
        bound.ring_ready = false;
        Ok(ClassQuiesce::Stopped)
    }

    fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive> {
        bound.resources
    }

    fn refusal_code(error: &Self::Error) -> u8 {
        error.rung()
    }
}

/// **Four bytes of evidence for a C2b refusal, one number each.**
///
/// The detail byte says *which way* a rung failed; this says *what the part
/// was doing when it did*. It exists because note 79 came back from metal on
/// 2026-11-11 carrying its transport rung and nothing else — and every
/// question that rung raised (which section, had the engine fetched
/// anything, had the command even left the host) was already answered inside
/// the refusal and thrown away at this boundary.
///
/// Byte-aligned so the word can be read off a photograph of a terminal:
/// `0xDDCCBBAA` is section `AA`, `Fwdl` index `BB`, transmit index `CC`,
/// event-ring index `DD`. Each saturates at 255, which still reads as "past
/// what a byte can say" and never as a small number it is not.
///
/// **A free function, not a local in `bind`.** The first version packed this
/// inline, which put it on a path no test can reach without hardware — the
/// shape `AGENTS.md` names first, and one this driver has already paid for.
fn patch_evidence_of(err: Option<&kore_mt7925::c2::C2Refusal>) -> u32 {
    let byte = |n: u32| n.min(255);
    match err {
        Some(kore_mt7925::c2::C2Refusal::DownloadRefused {
            section,
            fwdl_idx,
            inner,
            ..
        }) => {
            // `NoAnswer` is the only transport rung with indices of its
            // own; the others refused before anything was posted, and a
            // zero there would mean "not applicable", not "zero".
            let (tx_idx, rx_idx) = match inner {
                kore_mt7925::d1::D1Refusal::NoAnswer {
                    tx_idx, dma_idx, ..
                } => (byte(*tx_idx), byte(*dma_idx)),
                _ => (0, 0),
            };
            u32::from(*section) | (byte(*fwdl_idx) << 8) | (tx_idx << 16) | (rx_idx << 24)
        }
        // The finish already reports its `Fwdl` index as the detail byte;
        // repeating it in the second byte costs nothing and keeps one
        // reading rule for the whole word.
        Some(kore_mt7925::c2::C2Refusal::FinishRefused { fwdl_idx }) => byte(*fwdl_idx) << 8,
        // **C2c's start keeps the index whole, not clamped to a byte.**
        // The patch's forty-six chunks fit in a byte; the RAM code's 267
        // do not, and a clamp here would turn "the download stopped at
        // 260" into the same 255 that "every chunk crossed" reports. The
        // note's evidence word has the room, so it is used.
        Some(kore_mt7925::c2::C2Refusal::StartRefused { fwdl_idx, .. }) => {
            (*fwdl_idx).min(0xFFFF)
        }
        // The register itself, whole: `MT_CONN_ON_MISC`'s low two bits are
        // the diagnosis and the rest is context worth keeping.
        Some(kore_mt7925::c2::C2Refusal::FirmwareNotReady { misc }) => *misc,
        _ => 0,
    }
}

impl NicBound for Mt7925NetBound {
    type Backend<'a> = NoFrames;

    /// No frame path at D0 — there is not even firmware. Every verb refuses
    /// `NotReady`, which is the honest report.
    fn backend(&mut self) -> Option<NoFrames> {
        None
    }

    fn resources(&self) -> &DeviceResources<DmaActive> {
        &self.resources
    }

    /// A bound part passed D0 by construction — every rung below it refuses
    /// the bind — so it earns no refusal note.
    /// **Zero, always — and deliberately.**
    ///
    /// `MetalHost::founding_note` reads `self.frames`, and an MT7925 is
    /// bound into `self.radio`, so nothing ever calls this for this part.
    /// A version of this function that returned the D1 rung was written,
    /// shipped, and reported nothing on metal: dead code with passing
    /// tests, which is the unreachable seam `AGENTS.md` names first.
    ///
    /// The rung rides [`Self::bring_up_rung`] instead, out through
    /// `RadioHost` and `MetalHost` — the same path the witnesses take,
    /// and one `entry.rs` demonstrably calls.
    fn founding_note(&self) -> u64 {
        0
    }

    fn detail(&self) -> u8 {
        0
    }
}

impl Mt7925NetBinding {
    /// Stop and reset the DMA engine before anything is programmed into it.
    ///
    /// `mt792x_dma_disable(dev, force=true)`. Never fails the bind: a part
    /// whose engine will not quiet is still a part that passed D0, and the
    /// verdict rides out by leaving the engine unstarted.
    /// Hand the part back to its firmware and take it again, the way
    /// `mt7925e_mcu_init` does before the first MCU command.
    fn retake_ownership(
        resources: &DeviceResources<DmaActive>,
    ) -> Result<(), kore_mt7925::d1::D1Refusal> {
        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(kore_mt7925::d1::D1Refusal::NotOwned { lpctl: 0 });
        };
        let core = kore_mt7925::d0::Core::new(regs);
        let fault = |_| kore_mt7925::d1::D1Refusal::NotOwned {
            lpctl: core.read(kore_mt7925::regs::mt7925::MT_CONN_ON_LPCTL),
        };
        core.hand_back(kore_driver::host::relax, kore_driver::host::clock_now_ns)
            .map_err(fault)?;
        core.take_ownership(kore_driver::host::relax, kore_driver::host::clock_now_ns)
            .map_err(fault)
    }

    fn reset_wfsys(
        resources: &DeviceResources<DmaActive>,
    ) -> Result<(), kore_mt7925::d1::D1Refusal> {
        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(kore_mt7925::d1::D1Refusal::WfsysNotReady { sw_rst: 0 });
        };
        let core = kore_mt7925::d0::Core::new(regs);
        core.reset_wfsys(kore_driver::host::relax, kore_driver::host::clock_now_ns)?;
        // `mt7925_pci_probe`'s next MMIO write after the reset, and the
        // capture's: let the PCIe MAC raise interrupts at all.
        core.arm_pcie_interrupts();
        Ok(())
    }

    fn reset_engine(
        resources: &DeviceResources<DmaActive>,
    ) -> Result<(), kore_mt7925::d1::D1Refusal> {
        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(kore_mt7925::d1::D1Refusal::EngineBusy { glo_cfg: 0 });
        };
        kore_mt7925::d0::Core::new(regs)
            .reset_engine(kore_driver::host::relax, kore_driver::host::clock_now_ns)
    }

    /// Allocate the firmware-download ring and program it, returning the
    /// buffer to hold and whether the engine took it.
    ///
    /// Never fails the bind. Every way this can refuse leaves the part at
    /// D0, which is a true thing the row can still report.
    fn program_download_ring(
        part: &Mt7925,
        resources: &DeviceResources<DmaActive>,
    ) -> (Option<CapBuf>, bool) {
        let Some(pool) = resources.pool() else {
            return (None, false);
        };
        let queue = Queue::Fwdl;
        // `DmaPool::alloc` is page-aligned and zeroed by contract, which is
        // what a descriptor table needs: the engine must not read a stale
        // `done` bit out of a slot nobody wrote.
        let Some(table) = pool.alloc(queue.table_bytes()) else {
            return (None, false);
        };
        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return (Some(table), false);
        };
        let core = kore_mt7925::d0::Core::new(regs);
        let ok = core
            .program_queue(queue, table.iova().get(), table.len() as u64)
            .is_ok();
        // The buffer is returned either way. A refusal after the base was
        // written would leave the engine naming these pages, so they are
        // held until quiesce clears it.
        (Some(table), ok)
    }
}

impl Mt7925NetBinding {
    /// Start the engine and prove it takes what it is handed — **D1b**.
    ///
    /// Programs the event queue, fills it with receive buffers, enables
    /// WFDMA, rings the doorbell, and reads the engine's own `dma_idx`
    /// back. Returns the backings to hold and whether the engine moved.
    ///
    /// Never fails the bind, for the same reason D1a does not: a part that
    /// reached D1a and no further is still a part that answered, and a row
    /// reporting that honestly is worth more than one that refuses.
    fn start_engine(resources: &DeviceResources<DmaActive>) -> (EventRing, bool) {
        let mut ring = EventRing::default();
        let (Some(pool), Some(regs)) = (resources.pool(), resources.bar(MT7925_REGS_BAR)) else {
            return (ring, false);
        };
        let queue = Queue::EventWm;
        let depth = queue.depth();
        let Some(table) = pool.alloc(queue.table_bytes()) else {
            return (ring, false);
        };
        ring.table = Some(table);
        // One contiguous slab of `MT_RX_BUF_SIZE` buffers, one per slot.
        let buf_bytes = kore_mt7925::regs::mt7925::MT_RX_BUF_SIZE as usize;
        let Some(buffers) = pool.alloc(buf_bytes * depth as usize) else {
            return (ring, false);
        };
        ring.buffers = Some(buffers);
        let (Some(table), Some(buffers)) = (ring.table.as_ref(), ring.buffers.as_ref()) else {
            return (ring, false);
        };
        let core = kore_mt7925::d0::Core::new(regs);
        if core
            .program_queue(queue, table.iova().get(), table.len() as u64)
            .is_err()
        {
            return (ring, false);
        }
        // **The descriptors are written before the engine is enabled.** An
        // engine started over an unfilled ring consumes slots whose buffers
        // are not yet mapped.
        let wrote = arm_receive_ring(table, buffers, depth, buf_bytes);
        // The filled ring is published before the engine is enabled below.
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        if wrote != depth {
            return (ring, false);
        }
        // **Every ring is programmed before the engine is enabled.** This
        // used to allocate and program the command ring *after*
        // `enable_engine`, which is not upstream's order
        // (`mt792x_dma_init` programs every queue and then calls
        // `mt792x_dma_enable`, `mt792x_dma.c:95-130`) and is not safe:
        // `CSR_DISP_BASE_PTR_CHAIN_EN` means a running engine has each
        // ring's descriptor base cached in its prefetch window, so a base
        // written to a live engine need never be read. A receive-only
        // engine never exposed it, because the only ring it was handed
        // late was the one nothing ever rang.
        let mcu = Queue::McuWm;
        let (Some(mcu_table), Some(mcu_payload)) =
            (pool.alloc(mcu.table_bytes()), pool.alloc(PAGE_BYTES))
        else {
            return (ring, false);
        };
        ring.mcu_table = Some(mcu_table);
        ring.mcu_payload = Some(mcu_payload);
        // One staging buffer per download slot, so the forty-six chunks
        // that are published without ever being acknowledged each name
        // bytes nothing else will overwrite. `the_rings_fit_the_pool_the_
        // row_is_endowed` holds the carve to the endowment.
        let Some(fwdl_payload) = pool.alloc(FWDL_STAGING_BYTES) else {
            return (ring, false);
        };
        ring.fwdl_payload = Some(fwdl_payload);
        let Some(mcu_table) = ring.mcu_table.as_ref() else {
            return (ring, false);
        };
        if core
            .program_queue(mcu, mcu_table.iova().get(), mcu_table.len() as u64)
            .is_err()
        {
            return (ring, false);
        }
        // **Receive ring 2**, programmed and armed like ring 0 and before
        // the engine runs, for the same reason. Upstream programs it in
        // the same pass (`mt792x_dma_alloc_queues`, `mt792x_dma.c:146`);
        // this driver left it out until the RAM firmware went silent on
        // ring 0 — see `kore_mt7925::dma::Queue::EventData`.
        let data = Queue::EventData;
        let data_depth = data.depth();
        let (Some(data_table), Some(data_buffers)) = (
            pool.alloc(data.table_bytes()),
            pool.alloc(buf_bytes * data_depth as usize),
        ) else {
            return (ring, false);
        };
        ring.data_table = Some(data_table);
        ring.data_buffers = Some(data_buffers);
        let (Some(data_table), Some(data_buffers)) =
            (ring.data_table.as_ref(), ring.data_buffers.as_ref())
        else {
            return (ring, false);
        };
        if arm_receive_ring(data_table, data_buffers, data_depth, buf_bytes) != data_depth {
            return (ring, false);
        }
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        if core
            .program_queue(data, data_table.iova().get(), data_table.len() as u64)
            .is_err()
        {
            return (ring, false);
        }
        // Only now, with every ring latched and both receive rings
        // filled, is the engine allowed to run.
        core.enable_engine();
        // **Both rings one slot short**, which is upstream's fill
        // (`mt76_dma_rx_fill` stops at `ndesc - 1`) and what ring 2's
        // recycling keeps: the producer index sits one behind the host's
        // cursor, so a full ring and an empty one never read the same.
        //
        // Ring 0 used to be posted `wrote` — 512, on a 512-slot ring: an
        // index naming no slot. The ROM answered over it; the RAM firmware
        // has answered nothing over it on four flashes (M1b).
        core.post_receive(queue, kore_mt7925::dma::rx_fill_index(depth));
        core.post_receive(data, kore_mt7925::dma::rx_fill_index(data_depth));
        (ring, true)
    }

    /// Ask the part one question and see whether it answers — **D1c**.
    ///
    /// The ROM patch semaphore, which is upstream's own first download step
    /// (`mt76_connac2_load_patch`, `mt76_connac_mcu.c:3330`) and therefore
    /// a question this part is known to answer before any firmware is
    /// loaded. Taken and then released: a probe that left the semaphore
    /// held would make the download that follows it refuse.
    /// Put the ROM patch into the part — **C2b**.
    ///
    /// Never fails the bind, like every rung below it: a part that reached
    /// D1c and no further is still a part that answered, and the verdict
    /// rides the row's note.
    /// **C2c** — the N9 RAM code, over the rings D1a programmed.
    ///
    /// The same shape as [`Self::download_patch`], over the other blob and
    /// the other pin. The one structural difference is that the producer
    /// it inherits is no longer zero: `Conversation` carries both the MCU
    /// sequence and the `Fwdl` doorbell across this boundary, which is
    /// exactly what the patch download's own counters failed to do on
    /// 2026-11-11.
    fn download_ram(
        resources: &DeviceResources<DmaActive>,
        ring: &mut EventRing,
        fwdl: Option<&CapBuf>,
        image: Option<&'static [u8]>,
        wait: impl FnMut() + Copy,
    ) -> Result<kore_mt7925::c2::RamOutcome, kore_mt7925::c2::C2Refusal> {
        use kore_mt7925::c2::C2Refusal;
        use kore_mt7925::d1::D1Refusal;

        let Some(image) = image else {
            return Err(C2Refusal::Transport(D1Refusal::RingTooSmall {
                need: 1,
                have: 0,
            }));
        };
        let blob = kore_mt7925::fw::pinned(
            image,
            kore_mt7925::fw::RAM_PINNED_SIZE,
            &kore_mt7925::fw::RAM_PINNED_DIGEST,
        )
        .map_err(|_| C2Refusal::Transport(D1Refusal::RingTooSmall { need: 1, have: 0 }))?;
        let parsed = kore_mt7925::fw::RamCode::parse(blob)
            .map_err(|_| C2Refusal::SectionOutOfRange { offs: 0, len: 0 })?;

        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(C2Refusal::Transport(D1Refusal::Unreachable { iova: 0, len: 0 }));
        };
        let (
            Some(mcu_table),
            Some(payload),
            Some(rx_table),
            Some(rx_buffers),
            Some(fwdl),
            Some(fwdl_payload),
        ) = (
            ring.mcu_table.as_ref(),
            ring.mcu_payload.as_ref(),
            ring.table.as_ref(),
            ring.buffers.as_ref(),
            fwdl,
            ring.fwdl_payload.as_ref(),
        ) else {
            return Err(C2Refusal::Transport(D1Refusal::RingTooSmall { need: 1, have: 0 }));
        };
        let mut rings = kore_mt7925::d1::Rings {
            tx_table: mcu_table,
            fwdl_table: fwdl,
            fwdl_payload,
            fwdl_chunk_bytes: kore_mt7925::mcu::FW_SCATTER_MAX,
            tx_payload: payload,
            rx_table: rx_table,
            rx_buffers: rx_buffers,
            rx_buf_bytes: kore_mt7925::regs::mt7925::MT_RX_BUF_SIZE as usize,
            data: data_ring(&ring.data_table, &ring.data_buffers),
            conv: &mut ring.conv,
        };
        kore_mt7925::d0::Core::new(regs).download_ram(
            &mut rings,
            &parsed,
            blob,
            wait,
            kore_driver::host::clock_now_ns,
        )
    }

    /// **M1** — ask the part to describe itself, over the rings D1a
    /// programmed.
    ///
    /// The same shape as [`Self::download_ram`], under the other command
    /// class. The scratch buffer is a page: a capability reply on this
    /// part is a few hundred bytes, but its size is the firmware's choice
    /// and a buffer too small refuses with both numbers rather than
    /// handing the walk a short body.
    fn describe_the_part(
        resources: &DeviceResources<DmaActive>,
        ring: &mut EventRing,
        fwdl: Option<&CapBuf>,
        wait: impl FnMut() + Copy,
    ) -> Result<kore_mt7925::d2::PhyReport, kore_mt7925::d2::PhyRefusal> {
        use kore_mt7925::d1::D1Refusal;
        use kore_mt7925::d2::{PhyRefusal, PhyStep};

        let unreachable = |inner| PhyRefusal::Transport {
            step: PhyStep::Capability,
            inner,
            // Refused before the rings existed: nothing was asked, so
            // there is no firmware answer to read.
            after: None,
        };
        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(unreachable(D1Refusal::Unreachable { iova: 0, len: 0 }));
        };
        let (
            Some(mcu_table),
            Some(payload),
            Some(rx_table),
            Some(rx_buffers),
            Some(fwdl),
            Some(fwdl_payload),
        ) = (
            ring.mcu_table.as_ref(),
            ring.mcu_payload.as_ref(),
            ring.table.as_ref(),
            ring.buffers.as_ref(),
            fwdl,
            ring.fwdl_payload.as_ref(),
        ) else {
            return Err(unreachable(D1Refusal::RingTooSmall { need: 1, have: 0 }));
        };
        let mut rings = kore_mt7925::d1::Rings {
            tx_table: mcu_table,
            fwdl_table: fwdl,
            fwdl_payload,
            fwdl_chunk_bytes: kore_mt7925::mcu::FW_SCATTER_MAX,
            tx_payload: payload,
            rx_table,
            rx_buffers,
            rx_buf_bytes: kore_mt7925::regs::mt7925::MT_RX_BUF_SIZE as usize,
            data: data_ring(&ring.data_table, &ring.data_buffers),
            conv: &mut ring.conv,
        };
        // **On the stack, and bounded.** One page, which is more than a
        // capability reply has ever needed and less than the receive slot
        // it is copied out of.
        let mut scratch = [0u8; PAGE_BYTES];
        kore_mt7925::d0::Core::new(regs).bring_up_phy(
            &mut rings,
            &mut scratch,
            wait,
            kore_driver::host::clock_now_ns,
        )
    }

    /// **The last rung of C2** — wait for the part's own word that the
    /// firmware runs.
    fn await_firmware(
        resources: &DeviceResources<DmaActive>,
        wait: impl FnMut() + Copy,
    ) -> Result<u32, kore_mt7925::c2::C2Refusal> {
        use kore_mt7925::c2::C2Refusal;
        use kore_mt7925::d1::D1Refusal;

        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(C2Refusal::Transport(D1Refusal::Unreachable { iova: 0, len: 0 }));
        };
        kore_mt7925::d0::Core::new(regs)
            .wait_firmware_ready(wait, kore_driver::host::clock_now_ns)
    }

    fn download_patch(
        resources: &DeviceResources<DmaActive>,
        ring: &mut EventRing,
        // The download ring D1a programmed into the engine. Chunks ride
        // `Fwdl`, and the engine reads the base it was given.
        fwdl: Option<&CapBuf>,
        image: Option<&'static [u8]>,
        wait: impl FnMut() + Copy,
    ) -> Result<kore_mt7925::c2::PatchOutcome, kore_mt7925::c2::C2Refusal> {
        use kore_mt7925::c2::C2Refusal;
        use kore_mt7925::d1::D1Refusal;

        let Some(image) = image else {
            // No blob, or one that did not verify. `FirmwareVerdict`
            // already said which on the row's own line; this rung refuses
            // rather than sending bytes nobody vouched for.
            return Err(C2Refusal::Transport(D1Refusal::RingTooSmall {
                need: 1,
                have: 0,
            }));
        };
        let blob = kore_mt7925::fw::pinned(
            image,
            kore_mt7925::fw::PATCH_PINNED_SIZE,
            &kore_mt7925::fw::PATCH_PINNED_DIGEST,
        )
        .map_err(|_| C2Refusal::Transport(D1Refusal::RingTooSmall { need: 1, have: 0 }))?;
        let parsed = kore_mt7925::fw::Patch::parse(blob)
            .map_err(|_| C2Refusal::SectionOutOfRange { offs: 0, len: 0 })?;

        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(C2Refusal::Transport(D1Refusal::Unreachable { iova: 0, len: 0 }));
        };
        let (
            Some(mcu_table),
            Some(payload),
            Some(rx_table),
            Some(rx_buffers),
            Some(fwdl),
            Some(fwdl_payload),
        ) = (
            ring.mcu_table.as_ref(),
            ring.mcu_payload.as_ref(),
            ring.table.as_ref(),
            ring.buffers.as_ref(),
            fwdl,
            ring.fwdl_payload.as_ref(),
        ) else {
            return Err(C2Refusal::Transport(D1Refusal::RingTooSmall { need: 1, have: 0 }));
        };
        let mut rings = kore_mt7925::d1::Rings {
            tx_table: mcu_table,
            // **The ring D1a programmed into the engine.** Chunks ride
            // `Fwdl` and the engine reads the base it was given, not
            // the command ring's.
            fwdl_table: fwdl,
            fwdl_payload,
            fwdl_chunk_bytes: kore_mt7925::mcu::FW_SCATTER_MAX,
            tx_payload: payload,
            rx_table: rx_table,
            rx_buffers: rx_buffers,
            rx_buf_bytes: kore_mt7925::regs::mt7925::MT_RX_BUF_SIZE as usize,
            data: data_ring(&ring.data_table, &ring.data_buffers),
            conv: &mut ring.conv,
        };
        kore_mt7925::d0::Core::new(regs).download_patch(
            &mut rings,
            &parsed,
            blob,
            wait,
            kore_driver::host::clock_now_ns,
        )
    }

    fn ask_the_part(
        resources: &DeviceResources<DmaActive>,
        ring: &mut EventRing,
        wait: impl FnMut() + Copy,
    ) -> Result<(), kore_mt7925::d1::D1Refusal> {
        let Some(regs) = resources.bar(MT7925_REGS_BAR) else {
            return Err(kore_mt7925::d1::D1Refusal::Unreachable { iova: 0, len: 0 });
        };
        let (Some(mcu_table), Some(payload), Some(rx_table), Some(rx_buffers)) = (
            ring.mcu_table.as_ref(),
            ring.mcu_payload.as_ref(),
            ring.table.as_ref(),
            ring.buffers.as_ref(),
        ) else {
            return Err(kore_mt7925::d1::D1Refusal::RingTooSmall { need: 1, have: 0 });
        };
        let core = kore_mt7925::d0::Core::new(regs);
        // **No pointer arithmetic and no `unsafe` here any more.** These
        // used to be four `from_raw_parts` calls whose bounds were argued
        // in a comment; `Rings` now carries the buffers themselves, and the
        // driver reaches into them through the seam that bounds-checks
        // every access and makes the descriptor words volatile. The payload
        // address is read off the payload buffer, so it cannot disagree
        // with the buffer it names.
        let mut rings = kore_mt7925::d1::Rings {
            tx_table: mcu_table,
            // D1c sends no firmware chunks, so neither of these is ever
            // read: `exchange` routes everything but `FW_SCATTER` to
            // `McuWm`, and `Rings::bank`/`Rings::staging` are what decide
            // that.
            fwdl_table: mcu_table,
            fwdl_payload: payload,
            fwdl_chunk_bytes: payload.len(),
            tx_payload: payload,
            rx_table: rx_table,
            rx_buffers: rx_buffers,
            rx_buf_bytes: kore_mt7925::regs::mt7925::MT_RX_BUF_SIZE as usize,
            data: data_ring(&ring.data_table, &ring.data_buffers),
            conv: &mut ring.conv,
        };
        // **Upstream's first command, before anything is asked** —
        // `mt792x_load_firmware` opens with `NIC_POWER_CTRL` and a
        // power-on poll that only warns (`mt792x_core.c:1101-1106`). The
        // 2026-11-20 capture has it first; this ladder never sent it. Its
        // reply and the power-on state are upstream's to ignore and ours
        // too; only a command the engine never fetched refuses, because
        // the next one would overwrite it in the shared staging buffer.
        core.restart(&mut rings, wait, kore_driver::host::clock_now_ns)?;
        let asked = core
            .exchange(
                &mut rings,
                kore_mt7925::mcu::Command::PatchSem,
                &kore_mt7925::mcu::patch_sem_body(true),
                wait,
                kore_driver::host::clock_now_ns,
            )
            .map(|_| ());
        if asked.is_ok() {
            // Give it back. A held semaphore is a download that refuses,
            // and this rung is a question, not a claim on the part.
            let _ = core.exchange(
                &mut rings,
                kore_mt7925::mcu::Command::PatchSem,
                &kore_mt7925::mcu::patch_sem_body(false),
                wait,
                kore_driver::host::clock_now_ns,
            );
        }
        asked
    }
}

impl Mt7925NetBound {
    /// Whether D1a passed: the download ring programmed and read back.
    pub fn ring_ready(&self) -> bool {
        self.ring_ready
    }

    /// Whether this row started the engine — teardown state, not testimony.
    pub fn engine_started(&self) -> bool {
        self.engine_started
    }

    /// Whether the part answered an MCU command — the `net-mt7925-d1-mcu`
    /// witness.
    /// What C2b found, or `None` when the rung was not driven.
    pub fn patch(&self) -> Option<kore_mt7925::c2::PatchOutcome> {
        self.patch
    }

    /// What C2c sent, or `None` when the rung was not driven.
    pub fn ram(&self) -> Option<kore_mt7925::c2::RamOutcome> {
        self.ram
    }

    /// What M1 read back, or `None` when that rung was not driven.
    pub fn phy(&self) -> Option<kore_mt7925::d2::PhyReport> {
        self.phy
    }

    /// `MT_CONN_ON_MISC` when the N9 reported itself up, or `None`.
    ///
    /// **This, and not [`Self::firmware_ready`], is what
    /// `net-wifi-fw-alive` may be earned from.** `firmware_ready` reads
    /// the register now and would answer true on a part some *other*
    /// driver started — after a warm boot from Linux, say. This is `Some`
    /// only when this boot drove the download and then watched the core
    /// come up, which is the same distinction `PATCH_SENT` draws against
    /// `PATCH_RESIDENT`.
    pub fn fw_alive(&self) -> Option<u32> {
        self.fw_alive
    }

    /// What the boot carried for this part, and whether it is pinned.
    pub fn firmware(&self) -> FirmwareVerdict {
        self.firmware
    }

    /// The first rung the D1 ladder refused at, or zero if it refused
    /// nowhere. Published as the row's note by `entry.rs`.
    pub fn bring_up_rung(&self) -> u8 {
        self.mcu_rung
    }

    /// One byte of evidence for [`Self::bring_up_rung`]; zero when there
    /// is none.
    pub fn bring_up_detail(&self) -> u8 {
        self.mcu_detail
    }

    /// Four bytes of evidence for [`Self::bring_up_rung`]; zero when the
    /// rung has no numbers of its own.
    pub fn bring_up_evidence(&self) -> u32 {
        self.mcu_evidence
    }

    /// Whether the MCU answered the one question D1c asks.
    pub fn mcu_answered(&self) -> bool {
        self.mcu_answered
    }

    /// The composed revision the part answered with: `0x79250000` on the
    /// lab's Framework 13 (`res/MT7925_EVIDENCE.md` §11). This is what the
    /// `net-mt7925-d0` witness prints.
    pub fn revision(&self) -> u32 {
        self.part.revision()
    }

    /// Whether the part's N9 core already reported itself up. Expected
    /// `false` before D1; recorded so D1 knows what it inherited.
    pub fn firmware_ready(&self) -> bool {
        self.part.firmware_ready()
    }
}

/// The frame engine's host over the MT7925: the generic device host,
/// instantiated.
pub type Mt7925Host = NetDeviceHost<Mt7925NetBinding>;


/// **The two numbers the bring-up note publishes, chosen by which rung
/// refused.**
///
/// The note's rung already names the first refusal on the ladder; this
/// picks the detail and the evidence from *that same* refusal. A note
/// whose rung names the capability query and whose evidence is a stale
/// download index describes two failures at once —
/// `res/NETWORK_C2_TO_PING.md` §2 records that defect twice, and both
/// times a flash was spent finding it.
///
/// Pure and out of `bind`, which needs a part. The composition is the
/// thing worth testing: a correct chooser that nothing calls is the
/// unreachable seam `AGENTS.md` names, and this crate has already
/// shipped one.
fn bring_up_note_numbers(
    phy_err: Option<&kore_mt7925::d2::PhyRefusal>,
    c2_err: Option<&kore_mt7925::c2::C2Refusal>,
    mcu: Option<&Result<(), kore_mt7925::d1::D1Refusal>>,
    irq: IrqLog,
) -> (u8, u32) {
    let patch_detail = match c2_err {
        Some(kore_mt7925::c2::C2Refusal::DownloadRefused { inner, .. }) => inner.rung(),
        // Which chunk stalled. Chunk zero is a transport that never
        // worked; chunk forty is a part that stopped accepting, and
        // they are different repairs. Clamped, because the byte is a
        // byte and 255 still reads as "late".
        Some(kore_mt7925::c2::C2Refusal::ChunkNotTaken { chunk, .. }) => {
            (*chunk).min(255) as u8
        }
        // **How far the download actually got.** Nothing answers a
        // chunk, so the finish is the first thing that can report on
        // them, and the engine's own `Fwdl` index is the report: zero
        // says the ring never moved, forty-six says every chunk was
        // fetched and only the finish failed.
        Some(kore_mt7925::c2::C2Refusal::FinishRefused { fwdl_idx }) => {
            (*fwdl_idx).min(255) as u8
        }
        // The part's own word for why.
        Some(kore_mt7925::c2::C2Refusal::FinishStatus { status }) => *status,
        Some(kore_mt7925::c2::C2Refusal::StartRefused { inner, .. }) => inner.rung(),
        Some(kore_mt7925::c2::C2Refusal::StartStatus { status }) => *status,
        // Nothing: the register is the whole report and it rides the
        // evidence word, where it fits.
        Some(kore_mt7925::c2::C2Refusal::FirmwareNotReady { .. }) => 0,
        _ => 0,
    };
    let patch_evidence = patch_evidence_of(c2_err);
    let mcu_detail = match mcu {
        Some(Err(kore_mt7925::d1::D1Refusal::NoAnswer { tx_idx: 0, .. })) => 1,
        Some(Err(kore_mt7925::d1::D1Refusal::NoAnswer { .. })) => 2,
        _ => patch_detail,
    };
    // The evidence belongs to the patch rung alone: a D1 refusal below
    // C2 reports through `mcu_detail`, and attaching a stale patch word
    // to it would be a number that means nothing.
    let mcu_evidence = if c2_err.is_some() { patch_evidence } else { 0 };
    // **M1's refusal owns its own numbers.**
    //
    // The rung above already took the first refusal on the ladder, and
    // when that refusal is M1's the detail and evidence must come from
    // it too — a note whose rung names the capability query and whose
    // evidence is a stale download index describes two different
    // failures at once, which is the defect
    // `res/NETWORK_C2_TO_PING.md` §2 records twice. C2 cannot have
    // refused here: M1 is only driven over a part whose firmware is
    // running, so `c2_err` is `None` whenever `phy_err` is `Some`.
    match phy_err {
        Some(refusal) => phy_note_numbers(refusal, irq),
        None => (mcu_detail, mcu_evidence),
    }
}

/// **M1's refusal, as the two numbers a note carries.**
///
/// Detail and evidence for a `PhyRefusal`, so the note's rung and its
/// numbers describe the same failure. A note whose rung names the
/// capability query and whose evidence is a stale download index
/// describes two failures at once — the defect
/// `res/NETWORK_C2_TO_PING.md` §2 records twice, both times found on
/// metal rather than here.
///
/// Pure and separate from `bind`, which needs a part, because this is
/// the only thing a photograph of a failed M1 will have to go on.
/// **M1's transport evidence word: where the silence was.**
///
/// The 2026-11-18 photograph of M1b read `[raw 0x71656]` — rung 22, the
/// capability query, and evidence 7, `NoAnswer` — and nothing else. The
/// command matched upstream byte for byte, so the next question was not
/// *what* was sent but how far it got, and the word that should have said
/// carried only the rung. Three readings stayed indistinguishable:
///
/// - the engine never fetched the command (`tx_idx` behind `tx_next`);
/// - it did, and the part wrote nothing (`rx_pending` zero, `skipped` zero);
/// - the part answered and the triage handed the answer to somebody else
///   (`skipped` non-zero), or wrote a slot whose done bit the host never saw
///   (`rx_pending` non-zero, `skipped` zero).
///
/// The word was re-cut twice after that. The second photograph settled the
/// first two readings (`0xd0d7`: 13 fetched of 13, nothing pending, nothing
/// skipped), and the third put the firmware's state in the detail byte;
/// the fourth (top nibble `8`) showed receive ring 2 programmed and empty.
/// Six flashes have printed the same command cursor, so it gives up its
/// byte to **what the interrupt route saw** — the question once the
/// vendor's mask is armed and every wait parks on the route:
///
/// ```text
/// bits  0–3   inner rung              (NoAnswer = 7)
/// bits  4–7   tx_lag, saturating 15   doorbell minus engine index: 0 is
///                                     "the engine fetched the command"
/// bits  8–13  rx_pending, sat. 63     ring-0 slots past the host's cursor
/// bits 14–15  skipped, saturating 3   replies the triage gave away
/// bits 16–19  ring 2                  0 = not programmed; 8 = programmed
///                                     and empty; 9..F = 1..7+ pending
/// bits 20–23  fired, saturating 15    waits the route ended (not the slice)
/// bits 24–31  what was acknowledged:
///               24 RX0 done           25 RX2 done
///               26 TX15/17 done (MCU) 27 TX16 done (download)
///               28 MCU software int   29 any other status bit
///               30 MT_MCU_CMD WAKE_RX_PCIE
///               31 any other MT_MCU_CMD bit
/// ```
///
/// Read off a photograph: the top byte is what the part raised, the next
/// nibble whether any of it was delivered. `fired = 0` beside a non-zero
/// top byte is a part that raised interrupts this machine never delivered;
/// bit 26 alone is a command the engine took and a firmware that said
/// nothing. Every other transport rung refused before anything was posted,
/// so its fields stay zero and the word is the bare rung.
fn transport_evidence(inner: &kore_mt7925::d1::D1Refusal, irq: IrqLog) -> u32 {
    use kore_mt7925::regs::mt7925::{
        HOST_RX_DONE_INT_ENA0, HOST_RX_DONE_INT_ENA2, HOST_TX_DONE_INT_ENA15,
        HOST_TX_DONE_INT_ENA16, HOST_TX_DONE_INT_ENA17, MCU2HOST_SW_INT_ENA,
        MT_MCU_CMD_WAKE_RX_PCIE,
    };
    let rung = u32::from(inner.rung()) & 0xF;
    match *inner {
        kore_mt7925::d1::D1Refusal::NoAnswer {
            tx_idx,
            dma_idx,
            tx_next,
            rx_next,
            skipped,
            data_pending,
            ..
        } => {
            let lag = tx_next.wrapping_sub(tx_idx).wrapping_rem(256).min(15);
            let pending = (dma_idx.wrapping_sub(rx_next) & 0x1FF).min(63);
            let ring2 = data_pending.map_or(0, |p| 0x8 | p.min(7));
            let fired = irq.fired.min(15);
            let named = HOST_RX_DONE_INT_ENA0
                | HOST_RX_DONE_INT_ENA2
                | HOST_TX_DONE_INT_ENA15
                | HOST_TX_DONE_INT_ENA17
                | HOST_TX_DONE_INT_ENA16
                | MCU2HOST_SW_INT_ENA;
            let bit = |set: bool, at: u32| u32::from(set) << at;
            let seen = bit(irq.sta & HOST_RX_DONE_INT_ENA0 != 0, 0)
                | bit(irq.sta & HOST_RX_DONE_INT_ENA2 != 0, 1)
                | bit(irq.sta & (HOST_TX_DONE_INT_ENA15 | HOST_TX_DONE_INT_ENA17) != 0, 2)
                | bit(irq.sta & HOST_TX_DONE_INT_ENA16 != 0, 3)
                | bit(irq.sta & MCU2HOST_SW_INT_ENA != 0, 4)
                | bit(irq.sta & !named != 0, 5)
                | bit(irq.mcu_cmd & MT_MCU_CMD_WAKE_RX_PCIE != 0, 6)
                | bit(irq.mcu_cmd & !MT_MCU_CMD_WAKE_RX_PCIE != 0, 7);
            rung | (lag << 4)
                | (pending << 8)
                | (u32::from(skipped.min(3)) << 14)
                | (ring2 << 16)
                | (fired << 20)
                | (seen << 24)
        }
        _ => rung,
    }
}

fn phy_note_numbers(refusal: &kore_mt7925::d2::PhyRefusal, irq: IrqLog) -> (u8, u32) {
    use kore_mt7925::d2::PhyRefusal;
    match refusal {
        // Which of the three commands is the note's rung already
        // (`PhyStep::rung`), so the detail byte is free for the firmware's
        // own state at the moment of failure — see
        // `kore_mt7925::d2::FirmwareState::byte`; zero means it was never
        // read. The evidence is the transport refusal and, for `NoAnswer`,
        // the cursors that say *where* the silence was
        // ([`transport_evidence`]).
        PhyRefusal::Transport { inner, after, .. } => (
            after.map_or(0, kore_mt7925::d2::FirmwareState::byte),
            transport_evidence(inner, irq),
        ),
        // The walk's own complaint, and the record it broke on. A reply
        // that ran off its end and one carrying a record this driver
        // cannot name are different repairs.
        PhyRefusal::Capability(err) => (err.code(), err.evidence()),
        // The walk succeeded and the part chose not to say who it is.
        // How many records it did carry is the whole report: zero says
        // the reply was empty, a dozen says it answered and omitted this
        // one, and those are faults in different places.
        PhyRefusal::NoMacAddress { elements } | PhyRefusal::NoPhyCapability { elements } => {
            (0, u32::from(*elements))
        }
        // How short. A block that arrived truncated and one that never
        // arrived read identically without this.
        PhyRefusal::EfuseShort { got } => (0, (*got).min(u32::MAX as usize) as u32),
        // Both numbers, because the repair is a size and you cannot
        // choose it from one of them.
        PhyRefusal::ScratchTooSmall { need, have } => (
            (*have).min(255) as u8,
            (*need).min(u32::MAX as usize) as u32,
        ),
    }
}

/// **The MT7925's bring-up ladder, climbed from what the bind and the
/// download actually did.**
///
/// Returns the witness bits for a part that reached at least
/// [`MT7925_LADDER_D0`] — the ordinal, plus the two flags that qualify a
/// step rather than being one. The caller adds `NET_MT7925_FW_PINNED`,
/// which is about the image and not the part.
///
/// **The ladder climbs, it does not accumulate.** Each step is tested
/// only inside the one below it, so what lands in the note is the highest
/// *contiguous* step the part reached. A fact that is true out of order —
/// RAM across a ring the engine never latched — stops the climb instead
/// of being credited, which under-claims rather than over-claims. That is
/// the right direction for a witness: `res/NETWORK_RESTART.md` §1 fact 2
/// is a record of what over-claiming cost, and nothing here is worth
/// repeating it for. The note's evidence word still carries the detail.
///
/// **Here rather than in `entry.rs`, deliberately.** `entry` is
/// `cfg(target_os = "none")`, so a host test can never reach it — and
/// this is the function that decides what a flash reports. A climb with
/// no reachable test is the unreachable seam `AGENTS.md` names, and it
/// was written into `entry.rs` first and moved once the test runner
/// declined to see it.
pub fn climb(
    ring: bool,
    mcu: bool,
    patch: Option<kore_mt7925::c2::PatchOutcome>,
    ram: bool,
    fw_alive: bool,
    phy: bool,
) -> u64 {
    let mut witness = 0u64;
    // **D0, and only D0** to begin with. The bind refuses at
    // `NotThisPart` unless the identity register answered this part's own
    // id through the L1 remap window, and at `Ownership` unless the part
    // then handed over inside the vendor's own budget — so a bound
    // MT7925 *is* those two facts, derived from the bind and never
    // declared. Deliberately not `NET_WIFI_FW_ALIVE`: no firmware has
    // moved.
    let mut ladder = kore_abi::founding::MT7925_LADDER_D0;
    // D1a: the download ring was programmed and the engine read the base
    // and depth back. A part can honestly reach D0 and not this.
    if ring {
        ladder = kore_abi::founding::MT7925_LADDER_D1_RING;
        // D1c: the part answered. A command cannot cross a ring the
        // engine never latched.
        if mcu {
            ladder = kore_abi::founding::MT7925_LADDER_D1_MCU;
            // C2b. **The step says the patch is in place; the bit says
            // who put it there.** A part that already held its patch
            // proves the semaphore was read; only a part we sent one to
            // proves a byte can cross, and reporting the second on the
            // strength of the first would credit this driver with
            // somebody else's download. Both climb, because a part
            // patched before this boot still goes on to take its RAM
            // code.
            if patch.is_some() {
                ladder = kore_abi::founding::MT7925_LADDER_PATCH;
                if matches!(patch, Some(kore_mt7925::c2::PatchOutcome::AlreadyResident)) {
                    witness |= kore_abi::founding::NET_MT7925_PATCH_RESIDENT;
                }
                // **C2c.** One outcome, not two: the RAM code has no
                // semaphore and so no "already loaded" answer, which
                // means this step cannot be reached by a part somebody
                // else prepared.
                if ram {
                    ladder = kore_abi::founding::MT7925_LADDER_RAM_SENT;
                    // **The part's own word that the firmware runs**, a
                    // strictly larger claim than the start command being
                    // answered.
                    if fw_alive {
                        ladder = kore_abi::founding::MT7925_LADDER_FW_ALIVE;
                        // Shared with the rtw89 lineage, where it is
                        // earned from a PHY capability answer instead —
                        // one bit, one meaning, two ways of establishing
                        // it on two parts that climb differently. So it
                        // stays a bit beside the step rather than inside
                        // it.
                        witness |= kore_abi::founding::NET_WIFI_FW_ALIVE;
                        // **M1.** Three unified commands answered: the
                        // part described itself, a block of its own
                        // efuse came back, and it took that efuse as its
                        // EEPROM. Nested under firmware-alive because a
                        // configuration query to a part that never
                        // started is a question asked of nobody.
                        if phy {
                            ladder = kore_abi::founding::MT7925_LADDER_PHY_UP;
                        }
                    }
                }
            }
        }
    }
    witness | kore_abi::founding::mt7925_ladder_bits(ladder)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The window the host places is one this part can actually emit.**
    ///
    /// The regression this pins: `DMA_IOVA_BASE` is `0x1_0000_0000`, which
    /// is 4 GiB *exactly*, so a binding that let the 64-bit default stand
    /// would have the pool begin one byte past a 32-bit master's ceiling.
    /// Nothing about the part's behaviour would say so — a truncated
    /// descriptor address is a DMA write to somebody else's page, or an
    /// IOMMU fault with no obvious author.
    #[test]
    fn the_pool_is_placed_inside_this_parts_reach() {
        use kore_driver::host::{DMA_IOVA_BASE, dma_iova_base, dma_iova_ceiling};
        let bits = <Mt7925NetBinding as PciBindingContract>::DMA_ADDRESS_BITS;
        assert_eq!(bits, 32, "the part's own reach, not the platform's");
        // The per-device carve, `DRIVER_DMA_POOL_BYTES`.
        let len = 0x20_0000u64;
        let base = dma_iova_base(bits, len).expect("a placement this part can reach");
        assert_ne!(base, DMA_IOVA_BASE, "the default base is out of reach");
        assert!(base + len - 1 <= dma_iova_ceiling(bits));
        // And the driver's own reach check agrees with the host's placement,
        // so a buffer drawn from that pool is one a descriptor can carry.
        assert!(kore_mt7925::dma::addressable(base, len));
    }

    /// The binding names BAR0 and one MSI-family route.
    ///
    /// BAR0 is not a guess: `pcim_iomap_regions(pdev, BIT(0), …)` at
    /// `mt7925/pci.c:629` is the only window the vendor driver maps, and the
    /// 2 MiB span is the same number as the direct-addressing limit
    /// (`res/MT7925_EVIDENCE.md` §12).
    #[test]
    fn the_binding_names_bar0_and_one_msi_family_route() {
        assert_eq!(
            <Mt7925NetBinding as PciBindingContract>::BARS,
            kore_rt::service::PciBarMask::BAR0
        );
        assert_eq!(
            <Mt7925NetBinding as PciBindingContract>::INTERRUPTS,
            InterruptPolicy::Required(InterruptBlock::SINGLE)
        );
        assert_eq!(MT7925_REGS_BAR, 0);
    }

    /// **D1c is the first claim a register echo cannot satisfy**, and the
    /// pool must have room for every ring it needs.
    ///
    /// The second half is a real bound, not a formality: the event ring's
    /// receive slab alone is `MT7925_RX_MCU_RING_SIZE * MT_RX_BUF_SIZE`,
    /// and a silent `alloc` failure degrades this rung to D1a with no
    /// explanation. If the carve ever stops fitting, this fails here rather
    /// than on the one machine that can run it.
    #[test]
    fn the_rings_fit_the_pool_the_row_is_endowed() {
        use kore_abi::founding::{
            mt7925_ladder_bits, NET_MT7925_LADDER_MASK, NET_WIFI_FW_ALIVE,
            MT7925_LADDER_D0, MT7925_LADDER_D1_MCU, MT7925_LADDER_D1_RING,
            SERVING_WITNESS_BITS,
        };
        // D1c is strictly above the two steps a register echo satisfies.
        assert!(MT7925_LADDER_D1_MCU > MT7925_LADDER_D1_RING);
        assert!(MT7925_LADDER_D1_RING > MT7925_LADDER_D0);
        // And it is not the firmware-alive bit, which lives outside the
        // ladder because the rtw89 lineage earns it another way.
        assert_eq!(mt7925_ladder_bits(MT7925_LADDER_D1_MCU) & NET_WIFI_FW_ALIVE, 0);
        // The whole ladder rides the carve, so a reader that strips
        // witness bits from a note strips the step too.
        assert_eq!(
            SERVING_WITNESS_BITS & NET_MT7925_LADDER_MASK,
            NET_MT7925_LADDER_MASK
        );

        let rx = kore_mt7925::regs::mt7925::MT_RX_BUF_SIZE as usize;
        let needed = Queue::Fwdl.table_bytes()
            + Queue::EventWm.table_bytes()
            + Queue::McuWm.table_bytes()
            + Queue::EventWm.depth() as usize * rx
            + PAGE_BYTES
            + FWDL_STAGING_BYTES
            + Queue::EventData.table_bytes()
            + Queue::EventData.depth() as usize * rx;
        // `DRIVER_DMA_POOL_BYTES`, the ordinary size class this row is
        // endowed with (`DMA_POOL@2` in `services/net/boot.toml`).
        const POOL: usize = 0x20_0000;
        assert!(
            needed <= POOL,
            "the MT7925's rings need {needed} bytes of a {POOL}-byte pool"
        );
    }

    /// **C2c's download index survives past 255.**
    ///
    /// Every other evidence field in this note is clamped to a byte, which
    /// was correct while 46 was the largest number the ladder could
    /// produce. The RAM code is **267** chunks, and a byte clamp would
    /// print "the download stopped at 260" and "every chunk crossed"
    /// identically as 255 — collapsing the one distinction the number
    /// exists to draw. The evidence word has thirty-two bits; this spends
    /// sixteen of them.
    #[test]
    fn the_ram_downloads_index_is_not_clamped_to_a_byte() {
        use kore_mt7925::c2::C2Refusal;
        use kore_mt7925::d1::D1Refusal;

        let refusal = |fwdl_idx| C2Refusal::StartRefused {
            fwdl_idx,
            inner: D1Refusal::NoAnswer {
                tx_idx: 0,
                seq: 0,
                dma_idx: 0,
                tx_next: 0,
                rx_next: 0,
                skipped: 0,
                data_pending: None,
            },
        };
        // Every chunk of the distribution's RAM code taken.
        assert_eq!(patch_evidence_of(Some(&refusal(267))), 267);
        // A download that stopped late must not read the same as one that
        // finished — which is exactly what a byte clamp would do.
        assert_ne!(
            patch_evidence_of(Some(&refusal(260))),
            patch_evidence_of(Some(&refusal(267))),
            "a clamp has made a stalled download look complete",
        );
        // And the status renderer reads sixteen bits at shift 16, so the
        // value must survive the note's own packing too.
        let note = crate::note::mt7925(true, 19, 0, patch_evidence_of(Some(&refusal(267))));
        assert_eq!((note >> 16) & 0xFFFF, 267, "the note lost the index");
        assert_eq!(note & 0xFF, crate::note::MT7925_BRING_UP_BASE + 19);
    }

    /// Distinct and named is not reachable, and two boots reported nothing
    /// while every test passed.
    ///
    /// So this one checks the composition `entry.rs` actually publishes,
    /// for every rung the ladder can refuse with.
    #[test]
    fn a_refused_rung_becomes_the_note_the_row_publishes() {
        use crate::note;
        use kore_mt7925::c2::C2Refusal;
        for rung in 1..=16u8 {
            let published = note::mt7925(true, rung, 0, 0);
            assert_eq!(
                published,
                note::MT7925_BRING_UP_BASE + u64::from(rung),
                "rung {rung} must reach the note"
            );
            assert_ne!(published, 0, "rung {rung} would report as `up`");
        }
        // And the three ways there is nothing to say stay silent, so a row
        // with no part — or a part that answered — reads as it always did.
        assert_eq!(note::mt7925(false, 7, 1, 0), 0, "no part, no note");
        assert_eq!(note::mt7925(true, 0, 0, 0), 0, "nothing refused");

        // **The regression this test used to miss.** Every call above
        // passed `mcu_answered = false`, and the composition carried an
        // `mcu_answered` term that returned 0 when it was true — so each
        // C2b refusal, which by construction happens *after* the MCU has
        // answered, was discarded and the row printed `up` with no note.
        // A boot was spent discovering that. The parameter is gone; what
        // stands in its place is that a rung reported after a successful
        // D1c still reaches the note.
        for rung in [
            C2Refusal::SemaphoreRefused.rung(),
            C2Refusal::ChunkNotTaken { chunk: 0, dma_idx: 0 }.rung(),
            C2Refusal::FinishRefused { fwdl_idx: 0 }.rung(),
            C2Refusal::FinishStatus { status: 0 }.rung(),
        ] {
            assert_ne!(
                note::mt7925(true, rung, 0, 0),
                0,
                "a C2 rung must survive a D1c that succeeded"
            );
        }
        // The evidence rides above the rung without disturbing it: the
        // status app reads the low byte as the rung and the next as the
        // finding, so a detail that bled into the rung would rename the
        // failure.
        for detail in 0..=2u8 {
            let published = note::mt7925(true, 7, detail, 0);
            assert_eq!(published & 0xFF, note::MT7925_BRING_UP_BASE + 7);
            assert_eq!((published >> 8) & 0xFF, u64::from(detail));
        }
    }

    /// A host with no radio bound reports no rung, through the same
    /// accessor chain `entry.rs` uses — so the chain is exercised, not
    /// merely assumed to exist.
    /// **The evidence word survives the trip into the note, intact.**
    ///
    /// Note 79 arrived from metal on 2026-11-11 saying only "transport
    /// rung 7". The four numbers that would have told us *why* were inside
    /// the refusal and were dropped on the way to the row. This asserts
    /// they are not dropped again, and it asserts it on the composition
    /// the row actually publishes rather than on the packer alone.
    ///
    /// Every byte is a different value on purpose: a packing that shifted
    /// a field by the wrong amount, or swapped two, would still pass if
    /// they all read the same.
    #[test]
    fn the_download_refusal_carries_its_four_numbers_into_the_note() {
        use crate::note;
        use kore_mt7925::c2::C2Refusal;
        use kore_mt7925::d1::D1Refusal;

        let refusal = C2Refusal::DownloadRefused {
            addr: 0x8000_0000,
            section: 1,
            fwdl_idx: 46,
            inner: D1Refusal::NoAnswer {
                tx_idx: 3,
                seq: 9,
                dma_idx: 7,
                tx_next: 0,
                rx_next: 0,
                skipped: 0,
                data_pending: None,
            },
        };
        let evidence = patch_evidence_of(Some(&refusal));
        assert_eq!(evidence & 0xFF, 1, "section");
        assert_eq!((evidence >> 8) & 0xFF, 46, "the engine's Fwdl index");
        assert_eq!((evidence >> 16) & 0xFF, 3, "the transmit index");
        assert_eq!((evidence >> 24) & 0xFF, 7, "the event ring's index");

        // And through the note the row publishes: rung and detail keep
        // their bytes, and the evidence sits above both.
        let published = note::mt7925(true, refusal.rung(), 7, evidence);
        assert_eq!(published & 0xFF, note::MT7925_BRING_UP_BASE + 15);
        assert_eq!((published >> 8) & 0xFF, 7, "the transport rung");
        assert_eq!(published >> note::MT7925_EVIDENCE_SHIFT, u64::from(evidence));

        // **It must not reach the testimony carve.** The witness bits grow
        // downward from bit 63 and the lowest is 48; an evidence word that
        // ran into them would forge a PASS.
        let widest = note::mt7925(true, 15, 0xFF, u32::MAX);
        assert_eq!(
            widest & kore_abi::founding::SERVING_WITNESS_BITS,
            0,
            "the evidence word overlaps a witness bit",
        );
    }

    /// A rung with no numbers of its own reports none — zero stays
    /// reserved for "not instrumented", so an older build cannot read as a
    /// finding.
    #[test]
    fn a_rung_without_numbers_publishes_no_evidence() {
        use kore_mt7925::c2::C2Refusal;

        assert_eq!(patch_evidence_of(None), 0);
        assert_eq!(patch_evidence_of(Some(&C2Refusal::SemaphoreRefused)), 0);
        // The finish reports how far the download got, in the same byte
        // as the download refusal's, so one reading rule covers both.
        assert_eq!(
            patch_evidence_of(Some(&C2Refusal::FinishRefused { fwdl_idx: 46 })) >> 8,
            46,
        );
    }

    #[test]
    fn an_idle_host_carries_no_rung() {
        assert_eq!(crate::metal::MetalHost::IDLE.mt7925_bring_up_rung(), 0);
        assert_eq!(crate::metal::MetalHost::IDLE.mt7925_mcu_answered(), false);
    }

    /// **No rung of the bind may fail silently.**
    ///
    /// The regression test for the flash this cost. `ask_the_part`'s
    /// refusal was carried; `reset_wfsys`'s and `reset_engine`'s were
    /// dropped, so a failure in either reported `up` with no note — the
    /// same silence the note band had just been added to end.
    ///
    /// This cannot check the binding's own control flow without a device,
    /// so it checks the thing that made the silence possible: that every
    /// rung the ladder can refuse with has a distinct, non-zero rung, and
    /// that each one is named by the ABI decoder. A rung the decoder does
    /// not name prints as `(unmapped note)` on the only screen that can
    /// report it.
    ///
    /// **This test did not run until 2026-11-14.** Its `#[test]` and its
    /// doc comment had drifted onto a function thirty lines away, leaving
    /// the body here as an ordinary private fn. Three tests were lost the
    /// same way in one edit; this was the third and the one that guards
    /// the silence a flash was spent discovering.
    #[test]
    fn every_refusal_the_bind_can_carry_is_named() {
        use kore_mt7925::d1::D1Refusal::*;
        let every = [
            Unreachable { iova: 0, len: 0 },
            BaseNotLatched { wrote: 0, read: 0 },
            SizeNotLatched { wrote: 0, read: 0 },
            EngineBusy { glo_cfg: 0 },
            RxNotTaken { dma_idx: 0, posted: 0 },
            CommandTooLong { bytes: 0, room: 0 },
            NoAnswer { seq: 0, dma_idx: 0, tx_idx: 0, tx_next: 0, rx_next: 0, skipped: 0, data_pending: None },
            BadAnswer(kore_mt7925::mcu::McuError::TooShort),
            RingTooSmall { need: 0, have: 0 },
            WfsysNotReady { sw_rst: 0 },
            NotOwned { lpctl: 0 },
        ];
        // C2b continues the same band, so a collision would report the
        // wrong repair on the only screen that can report one.
        use kore_mt7925::c2::C2Refusal;
        let c2 = [
            C2Refusal::SemaphoreRefused,
            C2Refusal::SectionOutOfRange { offs: 0, len: 0 },
            C2Refusal::ChunkNotTaken { chunk: 0, dma_idx: 0 },
            C2Refusal::DownloadRefused {
                addr: 0,
                section: 0,
                fwdl_idx: 0,
                inner: kore_mt7925::d1::D1Refusal::NoAnswer {
                    tx_idx: 0,
                    seq: 0,
                    dma_idx: 0,
                    tx_next: 0,
                    rx_next: 0,
                    skipped: 0,
                    data_pending: None,
                },
            },
            C2Refusal::FinishRefused { fwdl_idx: 0 },
            C2Refusal::FinishStatus { status: 0 },
        ];
        let mut seen = [false; 256];
        for refusal in c2 {
            let rung = refusal.rung();
            assert_ne!(rung, 0, "zero reads as 'no refusal'");
            assert!(!seen[rung as usize], "C2 rung {rung} collides");
            seen[rung as usize] = true;
        }
        for refusal in every {
            let rung = refusal.rung();
            assert_ne!(rung, 0, "zero reads as 'no refusal' in the note");
            assert!(!seen[rung as usize], "two refusals share rung {rung}");
            seen[rung as usize] = true;
            // And it must land inside the band reserved for this part, so
            // it cannot collide with the Ethernet or Realtek ladders that
            // share the row's note byte.
            let note = crate::note::MT7925_BRING_UP_BASE + u64::from(rung);
            assert!(
                (crate::note::MT7925_BRING_UP_BASE..crate::note::MT7925_BRING_UP_BASE + 16)
                    .contains(&note)
            );
        }
        // `kore-abi` owns the other half of this: that each of those notes
        // decodes to words rather than to the catch-all.
    }

    /// **"No firmware" and "the wrong firmware" are different facts**, and
    /// only one of them is a fault.
    ///
    /// Every QEMU boot without the vendored submodule carries neither blob,
    /// and that must not read as tampering. A boot carrying one and not the
    /// other is a third state — a staging bug, not a trust one — and
    /// collapsing all three into a bool would have told an operator to
    /// re-verify bytes that were never shipped.
    #[test]
    fn the_firmware_verdict_tells_absent_from_partial_from_unpinned() {
        use super::FirmwareVerdict as V;
        use kore_mt7925::fw;

        assert_eq!(V::of(None, None), V::Absent);
        let patch = std::vec![0u8; fw::PATCH_PINNED_SIZE];
        let ram = std::vec![0u8; fw::RAM_PINNED_SIZE];
        assert_eq!(V::of(Some(&patch), None), V::Partial);
        assert_eq!(V::of(None, Some(&ram)), V::Partial);
        // Both present, both the right length, neither the right bytes.
        assert_eq!(V::of(Some(&patch), Some(&ram)), V::Unpinned);

        // Only the last of those may ever run a download.
        for v in [V::Absent, V::Partial, V::Unpinned] {
            assert!(!v.is_pinned(), "{v:?} must not gate the download");
        }
        assert!(V::Pinned.is_pinned());
    }

    /// **"We sent it" and "it was already there" must never be one claim.**
    ///
    /// Both are successes and only one says the download path works.
    /// Crediting this driver with a patch somebody else loaded is the
    /// shape of false green this crate has already paid for twice, once
    /// with a witness earned against a fiction and once with reporting
    /// that was never called.
    ///
    /// The repack did not merge them — it moved the question. Both now
    /// reach the same ladder step, because a part patched before this boot
    /// still goes on to take its RAM code and a ladder that stalled would
    /// report it as never having climbed. What distinguishes them is
    /// `NET_MT7925_PATCH_RESIDENT`, which must therefore stay **outside**
    /// the ladder's bits: a step that overlapped it would make "we sent
    /// it" and "it was already there" the same note after all.
    #[test]
    fn the_two_patch_outcomes_are_two_claims() {
        use kore_abi::founding::*;
        assert_eq!(NET_MT7925_PATCH_RESIDENT & NET_MT7925_LADDER_MASK, 0);
        assert_eq!(NET_MT7925_PATCH_RESIDENT & NET_MT7925_FW_PINNED, 0);
        // The step is reached from D1c and is below the RAM code, whichever
        // way the patch got there.
        assert!(MT7925_LADDER_PATCH > MT7925_LADDER_D1_MCU);
        assert!(MT7925_LADDER_PATCH < MT7925_LADDER_RAM_SENT);
        // Both ride the carve, so `status` prints either.
        assert_eq!(
            SERVING_WITNESS_BITS & (NET_MT7925_LADDER_MASK | NET_MT7925_PATCH_RESIDENT),
            NET_MT7925_LADDER_MASK | NET_MT7925_PATCH_RESIDENT
        );
    }

    /// The witness is about the image, and is the only MT7925 bit a boot
    /// with no radio could honestly earn — so it must not collide with
    /// anything that is about the part.
    ///
    /// **This test did not run until 2026-11-14.** Its `#[test]` had
    /// drifted onto the function above during an earlier edit, so it sat
    /// in the file as an ordinary private fn for three flashes. It is
    /// restored here with the ladder it now has to check.
    #[test]
    fn the_firmware_witness_is_its_own_claim() {
        use kore_abi::founding::*;
        // Not a step: a QEMU boot with the modules staged earns this and
        // the part is not involved.
        assert_eq!(NET_MT7925_FW_PINNED & NET_MT7925_LADDER_MASK, 0);
        assert_eq!(NET_MT7925_FW_PINNED & NET_WIFI_FW_ALIVE, 0);
        assert_eq!(NET_MT7925_FW_PINNED & NET_MT7925_PATCH_RESIDENT, 0);
        assert_eq!(SERVING_WITNESS_BITS & NET_MT7925_FW_PINNED, NET_MT7925_FW_PINNED);
    }

    /// **The default reach is the rung proven on metal, and nothing above
    /// it.**
    ///
    /// **This test did not run until 2026-11-14** — same drifted
    /// attribute as the one above, and the more expensive of the two to
    /// have lost, because this is the guard that makes raising `REACH` a
    /// deliberate act. Every raise it should have checked happened to be
    /// deliberate; none of them were checked.
    #[test]
    fn bring_up_rises_one_rung_at_a_time() {
        // **The ceiling moves only after a photograph.** Each raise below
        // is dated by the boot that earned it, in `status services`:
        //
        //   Engine  2026-11-10  (twice: the rung that isolated D1c, then
        //                       again after the bring-up grew by three
        //                       prerequisites a receive-only engine could
        //                       not have shown — the second run counts,
        //                       being the larger claim)
        //   Mcu     2026-11-10  `net-mt7925-d1-mcu PASS`
        //   Patch   2026-11-12  `net-mt7925-patch-sent PASS`
        //   Ram     2026-11-12  `net-mt7925-ram-sent PASS`
        //   Start   2026-11-12  `net-wifi-fw-alive PASS`
        //   Triage  2026-11-14  the same seven witnesses, no note
        //
        // So `Phy` — M1b — is exactly one rung above what metal has
        // confirmed, which is the most this constant ever allows.
        assert!(
            REACH <= Reach::Phy,
            "REACH may be at most one rung above what metal has confirmed; \
             flash and read `status services` before raising it again"
        );
    }

    /// **D0 and D1a are different claims and must not share a bit.**
    ///
    /// The regression this pins is the one `res/NETWORK_RESTART.md` §1
    /// fact 2 records: a rung earning a larger rung's witness. D0 is the
    /// part answering its identity; D1a is the engine latching a ring. A
    /// boot can honestly have the first and not the second — the ring is
    /// programmed after the bind and never refuses it — so a reader must be
    /// able to tell them apart, and both must be masked as testimony.
    #[test]
    fn d0_and_d1_are_separate_testimony() {
        use kore_abi::founding::{
            mt7925_ladder_bits, mt7925_ladder_of, NET_MT7925_LADDER_MASK,
            NET_WIFI_FW_ALIVE, MT7925_LADDER_D0, MT7925_LADDER_D1_RING,
            SERVING_WITNESS_BITS,
        };
        assert_ne!(MT7925_LADDER_D0, MT7925_LADDER_D1_RING);
        // A reader asks `>=`, so D0 must not read as D1a and D1a must
        // carry D0 with it — which is the whole reason an ordinal is
        // honest where a bit pair was not.
        assert!(mt7925_ladder_of(mt7925_ladder_bits(MT7925_LADDER_D0)) < MT7925_LADDER_D1_RING);
        assert!(mt7925_ladder_of(mt7925_ladder_bits(MT7925_LADDER_D1_RING)) >= MT7925_LADDER_D0);
        // Neither is the firmware-alive bit: no blob has moved at either
        // step, and that bit is what the route's cost was justified
        // against.
        assert_eq!(mt7925_ladder_bits(MT7925_LADDER_D1_RING) & NET_WIFI_FW_ALIVE, 0);
        // Masked as testimony, so a reader that strips witness bits from a
        // note strips the step too.
        assert_eq!(
            SERVING_WITNESS_BITS & NET_MT7925_LADDER_MASK,
            NET_MT7925_LADDER_MASK
        );
        // Testimony lives in the high bits; the low byte is the note's own
        // refusal rung, which the ladder must never reach into.
        assert_eq!(NET_MT7925_LADDER_MASK & 0xff, 0);
    }

    /// **The row's selector must admit this part's identity.**
    ///
    /// The regression this pins: the Wi-Fi selector demands an *exact*
    /// vendor:device pair, and for a while the MediaTek part had no selector
    /// of its own — so `claim_first` answered `NoMatch` on the one machine
    /// the lab has, the claim arm never ran, and the boot reported
    /// `note 0x1` with no testimony. A class-and-subclass check passes in
    /// that broken state, which is exactly why it is not the check here.
    #[test]
    fn the_row_asks_for_this_exact_part() {
        use kore_rt::service::DeviceSelector;
        let devices = <crate::metal::NetRow as kore_rt::service::SystemService>::DEVICES;
        let found = devices.iter().any(|d| {
            matches!(d.selector, DeviceSelector::Pci(p)
                if p.class == 0x02
                    && p.subclass == 0x80
                    && p.identity.exact_pair()
                        == Some((
                            kore_mt7925::VENDOR_MEDIATEK,
                            kore_mt7925::DEVICE_MT7925,
                        ))
                    && p.bars == kore_rt::service::PciBarMask::BAR0)
        });
        assert!(
            found,
            "no selector admits 14c3:0717 on BAR0 — the part cannot be claimed"
        );
    }

    /// The ladder's rungs are contiguous from 1 and land in their own note
    /// band, so a photograph of the panel names one failure.
    #[test]
    fn the_refusal_ladder_has_its_own_band() {
        let ladder = [
            Mt7925Error::NotPresent,
            Mt7925Error::NotThisPart,
            Mt7925Error::Ownership,
        ];
        for (i, error) in ladder.iter().enumerate() {
            assert_eq!(usize::from(Mt7925NetBinding::refusal_code(error)), i + 1);
        }
        let base = crate::note::MT7925_BRING_UP_BASE;
        let top = base + u64::from(Mt7925NetBinding::refusal_code(&Mt7925Error::Ownership));
        // Its own band: the Realtek ladder already reaches 60, and the two
        // parts' rungs mean different things.
        assert!(
            base > crate::note::WIFI_BRING_UP_BASE + 15,
            "the Wi-Fi band keeps its fifteen rungs of room"
        );
        // The rung is the note's LOW BYTE (`rung & 0xff` in
        // `kore_rt::service`), so the field runs to 255 and this band is
        // nowhere near the detail bits that start at bit 8.
        assert!(top < 256, "the band stays inside the note's rung byte");
        assert_eq!(top, 67);
    }

    /// **D0 is not firmware-alive**, and a part that has never seen a blob
    /// cannot earn the go/no-go bit.
    ///
    /// After the repack the two are not even the same kind of thing: D0
    /// is a ladder step and firmware-alive is a bit, so the question is
    /// whether the step's bits can reach the bit. They cannot, and this
    /// says so at build time rather than at the one boot nobody can
    /// re-run.
    #[test]
    fn the_d0_witness_is_not_the_firmware_alive_witness() {
        use kore_abi::founding::{
            mt7925_ladder_bits, NET_MT7925_LADDER_MASK, NET_WIFI_FW_ALIVE,
            MT7925_LADDER_D0, SERVING_WITNESS_BITS,
        };
        assert_eq!(NET_MT7925_LADDER_MASK & NET_WIFI_FW_ALIVE, 0);
        assert_ne!(mt7925_ladder_bits(MT7925_LADDER_D0), 0);
        assert_eq!(mt7925_ladder_bits(MT7925_LADDER_D0) & NET_WIFI_FW_ALIVE, 0);
        // And the step is inside the row's carve, so a reader masks it
        // with the rest rather than seeing it as another row's detail.
        assert_ne!(SERVING_WITNESS_BITS & mt7925_ladder_bits(MT7925_LADDER_D0), 0);
        // Never in the rung byte.
        assert_eq!(NET_MT7925_LADDER_MASK & 0xff, 0);
    }
}

#[cfg(test)]
mod climb_tests {
    use super::{IrqLog, bring_up_note_numbers, climb, patch_evidence_of, phy_note_numbers};
    use kore_abi::founding::{
        mt7925_ladder_of, NET_MT7925_PATCH_RESIDENT, NET_WIFI_FW_ALIVE,
        MT7925_LADDER_D0, MT7925_LADDER_D1_MCU, MT7925_LADDER_D1_RING,
        MT7925_LADDER_FW_ALIVE, MT7925_LADDER_PATCH, MT7925_LADDER_PHY_UP,
        MT7925_LADDER_RAM_SENT,
    };
    use kore_mt7925::c2::PatchOutcome;

    /// The patch outcome a download produces, whatever its fields are.
    fn downloaded() -> PatchOutcome {
        PatchOutcome::Downloaded {
            sections: 3,
            chunks: 46,
            bytes: 188_416,
        }
    }

    /// **The boot of 2026-11-14, as the service would encode it.**
    ///
    /// Everything true through firmware-alive: the climb must reach
    /// exactly that step and set the shared bit beside it, and must not
    /// claim the patch was resident.
    #[test]
    fn the_photographed_boot_climbs_to_fw_alive() {
        let w = climb(true, true, Some(downloaded()), true, true, false);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_FW_ALIVE);
        assert_ne!(w & NET_WIFI_FW_ALIVE, 0);
        assert_eq!(w & NET_MT7925_PATCH_RESIDENT, 0);
    }

    /// **A fact out of order stops the climb rather than being credited.**
    ///
    /// This is the property that makes an ordinal safer than six flags,
    /// and it is worth a test because the unsafe version is the one that
    /// reads more naturally: OR every fact you have. A part whose RAM
    /// code "crossed" a ring the engine never latched has a driver bug,
    /// and the right report is the low step, not the high one.
    #[test]
    fn a_fact_out_of_order_does_not_lift_the_ladder() {
        // No ring, but everything above it claims to have happened.
        let w = climb(false, true, Some(downloaded()), true, true, false);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_D0);
        // No MCU answer, but the download claims to have run.
        let w = climb(true, false, Some(downloaded()), true, true, false);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_D1_RING);
        // No patch, but the RAM code claims to have crossed.
        let w = climb(true, true, None, true, true, false);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_D1_MCU);
        // No RAM, but the firmware claims to be alive.
        let w = climb(true, true, Some(downloaded()), false, true, false);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_PATCH);
        // No firmware alive, but the part claims to have described
        // itself — M1 over a part that never started.
        let w = climb(true, true, Some(downloaded()), true, false, true);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_RAM_SENT);
    }

    /// **The firmware-alive bit is never set below its step.**
    ///
    /// It is the bit the whole route's cost was justified against
    /// (`res/NETWORK_RESTART.md` §1 fact 2), so a climb that stops short
    /// must not leave it behind.
    #[test]
    fn the_shared_bit_never_outruns_the_ladder() {
        for (ring, mcu, patch, ram, alive) in [
            (false, true, Some(downloaded()), true, true),
            (true, false, Some(downloaded()), true, true),
            (true, true, None, true, true),
            (true, true, Some(downloaded()), false, true),
            // **The boot this bit exists to refuse.** Every byte of the
            // RAM code crossed and the N9 did not start: the download
            // path worked and the firmware is not alive. A part can take
            // every byte and fail to run it, and crediting the go/no-go
            // bit here is the one claim `res/NETWORK_RESTART.md` §1 fact
            // 2 is a record of.
            (true, true, Some(downloaded()), true, false),
        ] {
            let w = climb(ring, mcu, patch, ram, alive, false);
            assert!(mt7925_ladder_of(w) < MT7925_LADDER_FW_ALIVE);
            assert_eq!(
                w & NET_WIFI_FW_ALIVE,
                0,
                "a stalled climb still claimed the firmware was alive",
            );
        }
    }

    /// A part that already held its patch climbs, and says so.
    #[test]
    fn a_resident_patch_climbs_and_is_marked() {
        let w = climb(true, true, Some(PatchOutcome::AlreadyResident), true, true, false);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_FW_ALIVE);
        assert_ne!(w & NET_MT7925_PATCH_RESIDENT, 0);
    }

    /// **The note's numbers come from the rung that refused.**
    ///
    /// Not from whichever refusal happens to be in scope. The chooser
    /// above is correct; this checks the *composition* calls it, which is
    /// a different claim and the one that has failed here before: a
    /// chooser nothing reaches is the unreachable seam, and three flashes
    /// were spent on one.
    #[test]
    fn the_note_takes_its_numbers_from_the_rung_that_refused() {
        use kore_mt7925::c2::C2Refusal;
        use kore_mt7925::d1::D1Refusal;
        use kore_mt7925::d2::PhyRefusal;

        let phy = PhyRefusal::EfuseShort { got: 9 };
        let c2 = C2Refusal::FinishRefused { fwdl_idx: 46 };

        // M1 refused: its numbers, whatever else is in scope.
        assert_eq!(
            bring_up_note_numbers(Some(&phy), None, None, IrqLog::default()),
            phy_note_numbers(&phy, IrqLog::default()),
        );
        // And M1 still wins if a C2 refusal is somehow present too —
        // which cannot happen on metal, since M1 only runs over running
        // firmware, but the composition must not depend on that to be
        // right.
        assert_eq!(
            bring_up_note_numbers(Some(&phy), Some(&c2), None, IrqLog::default()),
            phy_note_numbers(&phy, IrqLog::default()),
        );
        // No M1 refusal: C2's numbers reach the note unchanged.
        let (detail, evidence) = bring_up_note_numbers(None, Some(&c2), None, IrqLog::default());
        assert_eq!(detail, 46, "the download index stopped reaching the note");
        assert_eq!(evidence, patch_evidence_of(Some(&c2)));
        // Nothing refused anywhere: no numbers to publish.
        assert_eq!(bring_up_note_numbers(None, None, None, IrqLog::default()), (0, 0));
        // A D1 refusal below C2 still reports through its own detail,
        // and must not pick up a patch evidence word that belongs to a
        // download that never ran.
        let (_, evidence) = bring_up_note_numbers(
            None,
            None,
            Some(&Err(D1Refusal::NoAnswer {
                tx_idx: 0,
                seq: 0,
                dma_idx: 0,
                tx_next: 0,
                rx_next: 0,
                skipped: 0,
                data_pending: None,
            })),
            IrqLog::default(),
        );
        assert_eq!(evidence, 0, "a stale patch word attached to a D1 rung");
    }

    /// **An M1 refusal publishes M1's numbers, never C2's.**
    ///
    /// The rung and the evidence must describe one failure. This crate
    /// has shipped the other thing twice — a rung naming one step beside
    /// a number from another — and both times it cost a flash to find,
    /// because a note is read off a photograph and a reader has no way
    /// to tell a stale number from a current one.
    #[test]
    fn an_m1_refusal_publishes_its_own_numbers() {
        use kore_mt7925::d1::D1Refusal;
        use kore_mt7925::d2::{FirmwareState, PhyRefusal, PhyStep};
        use kore_mt7925::mcu::CapaError;

        // A transport failure names the command and the rung beneath it,
        // and a `NoAnswer` says where the silence was — every field
        // distinct, so a field landing in its neighbour's bits fails.
        let (detail, evidence) = phy_note_numbers(&PhyRefusal::Transport {
            step: PhyStep::EfuseRead,
            inner: D1Refusal::NoAnswer {
                tx_idx: 0x05,
                seq: 7,
                dma_idx: 0x2_0C,
                tx_next: 0x1_09,
                rx_next: 0x2_09,
                skipped: 2,
                data_pending: Some(5),
            },
            // N9 ready (`MT_TOP_MISC2_FW_N9_RDY = GENMASK(1, 0)`,
            // `mt792x_regs.h:508`) beside an unrelated high bit, and
            // `MT_MCU_CMD` holding `STOP_DMA` (bit 2, `:277`) and
            // `WAKE_RX_PCIE` (bit 0, `:275`) beside bits outside the
            // error mask that must not leak into the byte.
            after: Some(FirmwareState {
                misc: 0x8000_0003,
                mcu_cmd: 0xC000_0005,
            }),
        }, IrqLog {
            // Six delivered waits; the MCU's command ring completed and the
            // MCU raised its software interrupt with WAKE_RX_PCIE.
            fired: 6,
            sta: (1 << 25) | (1 << 29),
            mcu_cmd: 0x1,
        });
        // Read (7), WAKE_RX_PCIE (6), STOP_DMA verbatim (2), N9 ready (0).
        assert_eq!(detail, 0b1100_0101, "the firmware's state, not the step");
        assert_eq!(evidence & 0xF, 7, "the inner rung, NoAnswer");
        assert_eq!((evidence >> 4) & 0xF, 4, "the doorbell's lead, across a lap");
        assert_eq!((evidence >> 8) & 0x3F, 3, "slots filled past the cursor");
        assert_eq!((evidence >> 14) & 0x3, 2, "replies the triage gave away");
        assert_eq!((evidence >> 16) & 0xF, 0x8 | 5, "ring 2 programmed, five slots past");
        assert_eq!((evidence >> 20) & 0xF, 6, "delivered waits");
        assert_eq!(
            evidence >> 24,
            0b0101_0100,
            "TX15 done (26), the MCU's software interrupt (28), WAKE_RX_PCIE (30)"
        );

        // Pending wraps with the 512-slot event ring: an engine that has
        // lapped zero is still ahead, not 500-odd behind.
        let (detail, evidence) = phy_note_numbers(&PhyRefusal::Transport {
            step: PhyStep::Capability,
            inner: D1Refusal::NoAnswer {
                tx_idx: 0,
                seq: 1,
                dma_idx: 1,
                tx_next: 0,
                rx_next: 511,
                skipped: 200,
                data_pending: None,
            },
            // The firmware gone: N9 no longer ready, nothing raised.
            after: Some(FirmwareState {
                misc: 0x1,
                mcu_cmd: 0,
            }),
        }, IrqLog {
            // Delivery flooding, and status bits this layout does not name.
            fired: 400,
            sta: 1 << 31,
            mcu_cmd: 0x8000_0000,
        });
        assert_eq!((evidence >> 8) & 0x3F, 2, "pending across the wrap");
        assert_eq!(detail, 0x80, "read, and the N9 not ready: PWR_ON alone is not ready");
        assert_eq!((evidence >> 14) & 0x3, 3, "skipped saturates, never spills");
        assert_eq!((evidence >> 16) & 0xF, 0, "a ring never programmed must not read as empty");
        assert_eq!((evidence >> 20) & 0xF, 15, "fired saturates, never spills");
        assert_eq!(evidence >> 24, 0b1010_0000, "only the two 'any other' bits");

        // Ring 2 programmed and empty reads 8; a flood saturates at F and
        // never spills into the bits below. A command the engine never
        // fetched shows a lead, saturating at F.
        let ring2 = |data_pending, tx_idx, tx_next| {
            phy_note_numbers(&PhyRefusal::Transport {
                step: PhyStep::Capability,
                inner: D1Refusal::NoAnswer {
                    tx_idx,
                    seq: 1,
                    dma_idx: 0,
                    tx_next,
                    rx_next: 0,
                    skipped: 0,
                    data_pending,
                },
                after: None,
            }, IrqLog::default())
            .1
        };
        assert_eq!((ring2(Some(0), 13, 13) >> 16) & 0xF, 0x8, "programmed and empty");
        assert_eq!((ring2(Some(31), 13, 13) >> 16) & 0xF, 0xF, "a full ring saturates");
        assert_eq!(ring2(Some(31), 13, 13), 0x000F_0007, "and spills nowhere");
        assert_eq!((ring2(Some(0), 12, 13) >> 4) & 0xF, 1, "one command unfetched");
        assert_eq!((ring2(Some(0), 0, 200) >> 4) & 0xF, 15, "the lead saturates");
        // The metal case: fetched, nothing anywhere, ring 2 empty, no
        // delivery — the word a sixth silent flash would print.
        assert_eq!(ring2(Some(0), 13, 13), 0x0008_0007);

        // The other transport rungs posted nothing: the bare rung, as before.
        let (detail, evidence) = phy_note_numbers(&PhyRefusal::Transport {
            step: PhyStep::Capability,
            inner: D1Refusal::RingTooSmall { need: 9, have: 4 },
            after: None,
        }, IrqLog {
            // A route that fired must not decorate a rung that posted
            // nothing.
            fired: 3,
            sta: 1 << 25,
            mcu_cmd: 1,
        });
        assert_eq!(detail, 0, "never read must not read as read");
        assert_eq!(
            evidence,
            u32::from(D1Refusal::RingTooSmall { need: 0, have: 0 }.rung())
        );

        // A walk failure locates the record: index above, claimed length
        // below. Both numbers survive the packing.
        let (detail, evidence) = phy_note_numbers(&PhyRefusal::Capability(
            CapaError::RecordTooShort {
                index: 5,
                claimed: 2,
            },
        ), IrqLog::default());
        assert_eq!(detail, CapaError::RecordTooShort { index: 5, claimed: 2 }.code());
        assert_eq!(evidence >> 16, 5, "the record index was lost");
        assert_eq!(evidence & 0xFFFF, 2, "the claimed length was lost");

        // An empty reply and a reply that answered and omitted the MAC
        // must not read alike.
        let empty = phy_note_numbers(&PhyRefusal::NoMacAddress { elements: 0 }, IrqLog::default());
        let answered = phy_note_numbers(&PhyRefusal::NoMacAddress { elements: 12 }, IrqLog::default());
        assert_ne!(empty, answered);
        assert_eq!(answered.1, 12);

        // A sizing mistake carries the size you would have to choose.
        let (detail, evidence) =
            phy_note_numbers(&PhyRefusal::ScratchTooSmall { need: 2004, have: 512 }, IrqLog::default());
        assert_eq!(evidence, 2004, "the needed size was lost");
        assert_eq!(detail, 255, "a byte-clamped `have` still reads as large");

        // And every one of them composes into a note whose rung is M1's.
        for refusal in [
            PhyRefusal::NoMacAddress { elements: 0 },
            PhyRefusal::NoPhyCapability { elements: 3 },
            PhyRefusal::EfuseShort { got: 8 },
        ] {
            let (detail, evidence) = phy_note_numbers(&refusal, IrqLog::default());
            let note = crate::note::mt7925(true, refusal.rung(), detail, evidence);
            assert_eq!(
                note & 0xFF,
                crate::note::MT7925_BRING_UP_BASE + u64::from(refusal.rung()),
                "the note lost M1's rung",
            );
        }
    }

    /// **M1 is the seventh step, and only over running firmware.**
    ///
    /// The rung the whole repack was done to make room for. A part that
    /// described itself has necessarily answered every command below,
    /// so the step is reached from firmware-alive and nowhere else.
    #[test]
    fn describing_itself_is_the_seventh_step() {
        let w = climb(true, true, Some(downloaded()), true, true, true);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_PHY_UP);
        // The firmware-alive bit is still set beside it: a part at M1
        // did not stop being a part whose firmware is running.
        assert_ne!(w & NET_WIFI_FW_ALIVE, 0);
    }

    /// **A description from a part that never started does not count.**
    ///
    /// A configuration query to a part whose N9 never came up is a
    /// question asked of nobody, and an answer to it would be the
    /// firmware's predecessor talking — or nothing at all, misread. The
    /// climb refuses to credit it.
    #[test]
    fn a_description_without_running_firmware_does_not_lift_the_ladder() {
        let w = climb(true, true, Some(downloaded()), true, false, true);
        assert_eq!(mt7925_ladder_of(w), MT7925_LADDER_RAM_SENT);
        assert_eq!(w & NET_WIFI_FW_ALIVE, 0);
    }

    /// **The climb never reaches into the note's rung byte.**
    ///
    /// A witness that did would make a proven row decode as a refusal —
    /// the exact inversion the note vocabulary exists to prevent.
    #[test]
    fn the_climb_leaves_the_rung_byte_alone() {
        for alive in [false, true] {
            let w = climb(true, true, Some(downloaded()), true, alive, false);
            assert_eq!(w & 0xff, 0);
        }
    }
}
