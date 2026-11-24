//! **C2b — the ROM patch crosses.**
//!
//! D1c proved the round trip: a descriptor published, an engine that
//! fetched it, a part that read the command out of host memory, an MCU that
//! answered, and a reply written back into a posted buffer. Everything here
//! rides that, and almost all of it is the same transport.
//!
//! **Almost.** Three things about the download are not D1c's shape, and
//! none of them is visible from the function that does the downloading —
//! each lives in a different file, which is why they are stated here rather
//! than left to be rediscovered:
//!
//! 1. **`PATCH_IS_DL` is a success.** `mt76_connac2_load_patch` returns
//!    zero on it (`mt76_connac_mcu.c:3281-3289`): the patch being already
//!    resident is a correct outcome, not a refusal. On a machine that boots
//!    after Linux has had the radio up, it is the likely one.
//! 2. **`FW_SCATTER` carries no MCU header.** `mt7925_mcu_fill_message`
//!    returns early for it, so a chunk crosses as raw payload. Sending it
//!    through [`crate::d1::Core::exchange`] would prepend sixty-four bytes
//!    the MCU is not expecting.
//! 3. **A chunk is not answered.** `__mt76_mcu_send_firmware` sends each
//!    with `wait_resp = false` and reclaims the descriptor. Waiting for a
//!    done bit per chunk on this part's twenty-second budget would make a
//!    forty-six-chunk patch look like a quarter-hour hang.
//!
//! So this module adds exactly one mechanism — [`Core::send_payload`], a
//! publish-and-walk-away — and otherwise composes what D1c already proved.

use kore_driver::seam::{Dma, DmaShared, DmaWrite, Regs};
use kore_driver::wait::Budget;

use crate::d0::Core;
use crate::d1::{D1Refusal, Rings};
use crate::dma::{Descriptor, Queue};
use crate::fw::{Patch, RamCode};
use crate::mcu::{self, Command, Semaphore};

/// How long the engine gets to free a slot once the download ring has
/// lapped.
///
/// **Not a per-chunk wait.** Nothing answers a chunk and nothing
/// acknowledges one: `__mt76_mcu_send_firmware` publishes every chunk with
/// `wait_resp = false` and only reclaims descriptors opportunistically
/// (`mt76_dma_tx_cleanup`). The one thing a ring genuinely owes is that a
/// slot is not rewritten while the engine may still hold it, and that is
/// owed only after the producer has gone all the way round. The ROM patch
/// is 46 chunks on a 128-slot ring, so this budget is unreachable there;
/// the RAM firmware is 266 and will meet it.
pub const CHUNK_BUDGET_NS: u64 = 1_000_000_000;
/// The same wait, for a machine whose clock cannot answer.
pub const CHUNK_SPINS_UNCLOCKED: u32 = 200_000;

/// How long the N9 core gets to come up after `FW_START_REQ`.
///
/// Upstream polls `MT_CONN_ON_MISC` for `MT_TOP_MISC2_FW_N9_RDY` on a
/// 1500 ms budget. The ftrace capture from *this* silicon has the register
/// going `0x1` -> `0x3` in about 455 ms, so the vendor's figure is the
/// right order and is kept rather than tightened to what one machine
/// happened to do once.
pub const N9_BUDGET_NS: u64 = 1_500_000_000;
/// The same wait, for a machine whose clock cannot answer.
pub const N9_SPINS_UNCLOCKED: u32 = 300_000;

/// What the patch download could not do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum C2Refusal {
    /// A rung of the transport refused; the D1 rung is carried through.
    Transport(D1Refusal),
    /// The part would not hand over the patch semaphore.
    ///
    /// Distinct from every other refusal here because it is the one that
    /// says *nothing was attempted* — the download never started, so the
    /// part is exactly as it was.
    SemaphoreRefused,
    /// A section claims bytes the image does not hold.
    SectionOutOfRange {
        /// Where the section says its payload starts.
        offs: u32,
        /// How many bytes it claims.
        len: u32,
    },
    /// The engine did not take a chunk within its budget.
    ///
    /// Carries the chunk's ordinal, because a stall at chunk zero and a
    /// stall at chunk forty are different faults: the first is a transport
    /// that never worked, the second a part that stopped accepting.
    ChunkNotTaken {
        /// Which chunk, counting from zero.
        chunk: u32,
        /// The engine's own index when the budget ran out.
        dma_idx: u32,
    },
    /// The "a region is coming" request did not complete a round trip.
    ///
    /// **Not necessarily a refusal.** The part may have answered no, or
    /// may have said nothing at all — [`D1Refusal::NoAnswer`] is the
    /// second, and it is what metal reported on 2026-11-11. The two are
    /// different repairs, so the transport refusal is carried **whole**
    /// rather than flattened to its rung: the first version kept only
    /// `inner: u8` and the row could say *that* a section failed but not
    /// one number about how.
    DownloadRefused {
        /// The section's target address, which is what the request names.
        addr: u32,
        /// Which section, counting from zero.
        ///
        /// The patch ships two. Section zero fails before any chunk has
        /// been published; section one fails after the whole of section
        /// zero has crossed. Nothing else distinguishes them, and they are
        /// entirely different diagnoses.
        section: u8,
        /// The engine's own `Fwdl` index, read at the moment the request
        /// failed.
        ///
        /// **The measurement this program does not yet have.** Every chunk
        /// is published and none is acknowledged, so this is the only
        /// number that says whether the engine fetched any of them. Zero
        /// means the download ring never moved — and a part still waiting
        /// for a section's bytes has every reason not to answer the next
        /// command, which would make this rung a symptom and not a cause.
        fwdl_idx: u32,
        /// The transport refusal underneath, whole.
        inner: D1Refusal,
    },
    /// The finish command did not complete a round trip.
    ///
    /// **This is where the download's evidence is.** Every chunk before it
    /// is published and not acknowledged, so the finish is the first thing
    /// after the first byte that the part has to answer — and the engine's
    /// own `Fwdl` index, read at this moment, is the only number that says
    /// whether the chunks were fetched at all. Zero means the download ring
    /// never moved; forty-six means every chunk was taken and the finish
    /// itself is what failed. Those are different repairs, and on a machine
    /// with no serial port this byte is the whole difference.
    FinishRefused {
        /// The engine's `Fwdl` index when the finish failed.
        fwdl_idx: u32,
    },
    /// The part answered the finish, and the answer was a refusal.
    ///
    /// Separate from [`Self::FinishRefused`] because a part that answers is
    /// a part whose transport works. `mt7925_mcu_parse_response` reads the
    /// first body byte as the status for `PATCH_FINISH_REQ` exactly as it
    /// does for `PATCH_SEM_CONTROL` (`mt7925/mcu.c:57-60`), and a non-zero
    /// one is the error it returns. Ignoring it — which this did — turns a
    /// patch the part rejected into `patch-sent PASS`.
    FinishStatus {
        /// The part's own status byte.
        status: u8,
    },
    /// `FW_START_REQ` did not complete a round trip.
    StartRefused {
        /// The engine's `Fwdl` index when the start failed — the same
        /// evidence [`Self::FinishRefused`] carries, and for the same
        /// reason: nothing acknowledges a chunk, so this is the only
        /// number that says whether the RAM code was fetched at all.
        fwdl_idx: u32,
        /// The transport refusal underneath, whole.
        inner: D1Refusal,
    },
    /// The part answered `FW_START_REQ`, and the answer was a refusal.
    StartStatus {
        /// The part's own status byte.
        status: u8,
    },
    /// The N9 never reported itself up inside [`N9_BUDGET_NS`].
    ///
    /// **The only rung here whose evidence is a register and not a
    /// message.** `FW_START_REQ` being answered says the MCU took the
    /// command; it says nothing about whether the firmware this driver
    /// just wrote actually runs. `MT_CONN_ON_MISC` going `FW_PWR_ON |
    /// FW_N9_ON` is the part's own report that it does.
    FirmwareNotReady {
        /// What `MT_CONN_ON_MISC` read when the budget ran out.
        ///
        /// The capture from this silicon has it `0x1` -> `0x3`: `0x1` is
        /// powered and not yet up, so a timeout still reading `0x1` is a
        /// firmware that started and died, and `0x0` is one that never
        /// took power at all.
        misc: u32,
    },
}

impl C2Refusal {
    /// The rung this refusal reports, continuing the D1 ladder's numbering
    /// so one note band spans the whole bring-up.
    pub const fn rung(self) -> u8 {
        match self {
            Self::Transport(d1) => d1.rung(),
            Self::SemaphoreRefused => 12,
            Self::SectionOutOfRange { .. } => 13,
            Self::ChunkNotTaken { .. } => 14,
            Self::DownloadRefused { .. } => 15,
            Self::FinishRefused { .. } => 16,
            Self::FinishStatus { .. } => 17,
            // 18 was `OwnershipRefused`, the handshake between the patch
            // and the RAM code that the vendor only does over SDIO. Its
            // note text (82) stays in the ABI; nothing produces it.
            Self::StartRefused { .. } => 19,
            Self::StartStatus { .. } => 20,
            Self::FirmwareNotReady { .. } => 21,
        }
    }
}

/// What the patch rung found when it asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchOutcome {
    /// The part already had the patch; nothing was sent.
    ///
    /// A success, and upstream's own (`load_patch` returns zero on
    /// `PATCH_IS_DL`). Reported separately because "the firmware is there"
    /// and "we put it there" are different facts about the boot, and only
    /// the second proves the download path works.
    AlreadyResident,
    /// The patch crossed.
    Downloaded {
        /// Sections sent.
        sections: u32,
        /// Chunks sent across all of them.
        chunks: u32,
        /// Payload bytes sent, excluding every header.
        bytes: u64,
    },
}

/// What the RAM code rung sent.
///
/// **No `AlreadyResident` arm, unlike [`PatchOutcome`].** The patch is
/// arbitrated by a semaphore that can answer "already loaded"; the RAM
/// code has no such question, so there is no second success to report and
/// no way for this rung to be satisfied by somebody else's download.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamOutcome {
    /// Regions actually sent — the downloaded ones, so four of the
    /// distribution image's five.
    pub regions: u32,
    /// Chunks sent across all of them. **267** on the distribution's blob:
    /// 19 + 94 + 8 + 146, because chunking restarts at each region.
    pub chunks: u32,
    /// Payload bytes sent, excluding every header. **1,087,744** on the
    /// distribution's blob — `RamCode::download_bytes`.
    pub bytes: u64,
    /// The entry point `FW_START_REQ` was given, if a region claimed one.
    pub entry: Option<u32>,
}

impl<'r, R: Regs + ?Sized> Core<'r, R> {
    /// Publish one payload on `queue` and **do not wait for an answer**.
    ///
    /// The mechanism D1c does not have. `exchange` stages a command with its
    /// MCU header and then blocks on the event ring; a firmware chunk has no
    /// header and is never answered, so this writes the descriptor, rings
    /// the doorbell and returns. It blocks only when the ring has lapped
    /// and this slot may still be held by the engine; see
    /// [`CHUNK_BUDGET_NS`].
    ///
    /// `producer` is the running count of descriptors published on this
    /// queue; the doorbell is an absolute index, not an increment.
    pub fn send_payload<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        queue: Queue,
        payload: &[u8],
        producer: u32,
        mut delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(), C2Refusal> {
        let depth = queue.depth() as u32;
        // **The slot advances with the producer.** The doorbell is an
        // absolute index: the engine consumes slot `dma_idx` and moves on,
        // so after the first chunk it reads slot one, not slot zero again.
        // Writing every chunk into slot zero leaves the engine reading
        // zeroed pool memory — `buf0 = 0`, no length, no last-segment bit
        // — and it stops. The same single-use assumption `exchange`
        // carried, written again one layer over while fixing it there.
        let slot = ((producer - 1) % depth) as usize;

        // **The only wait this ring owes, and it is owed before the write,
        // not after it.**
        //
        // The first version waited after every chunk for `dma_idx` to pass
        // the descriptor it had just published — and then reported "the
        // engine stopped taking firmware chunks" when it did not. That is a
        // belief about what `dma_idx` means on a *transmit* queue of this
        // part, and nothing has ever tested it: D1c was proved by an MCU
        // answer, not by an index, and the register never appears in a
        // vendor capture because `Q_READ` is a bare `readl` on
        // `q->regs->field`. Upstream never waits for a chunk at all.
        //
        // What a ring genuinely owes is that this slot's descriptor — and
        // the bytes it names — are not rewritten while the engine may still
        // be fetching them. That is true only once the producer has lapped
        // the ring, so that is the only condition checked, and `dma_idx`
        // gates the download no earlier than the part's own geometry
        // requires.
        // **Refreshed on every publish, not only when the guard runs.**
        // See `Conversation::fwdl_taken`: recovering an absolute count
        // from a wrapping register works only while the engine cannot
        // have advanced a whole lap since the last reading. Sampling it
        // for the first time *at* the boundary is exactly a whole lap to
        // guess across — the engine's index and the producer's are then
        // congruent, and "consumed nothing" and "consumed everything"
        // are the same number.
        rings.conv.fwdl_taken = self.engine_taken(queue, rings.conv.fwdl_taken);
        if producer > depth {
            let mut budget = Budget::new(CHUNK_BUDGET_NS, CHUNK_SPINS_UNCLOCKED, now_ns);
            loop {
                let taken = rings.conv.fwdl_taken;
                if taken.saturating_add(depth) >= producer {
                    break;
                }
                if !budget.may_wait() {
                    return Err(C2Refusal::ChunkNotTaken {
                        chunk: producer.saturating_sub(1),
                        dma_idx: taken,
                    });
                }
                delay();
                rings.conv.fwdl_taken = self.engine_taken(queue, rings.conv.fwdl_taken);
            }
        }

        // This slot's own bytes, at this slot's own address. See
        // [`Rings::staging`]: the descriptor's address is read off the
        // buffer it was given, so the two cannot disagree.
        let staging = rings.staging(queue, slot);
        if payload.len() > staging.room {
            return Err(C2Refusal::Transport(D1Refusal::CommandTooLong {
                bytes: payload.len(),
                room: staging.room,
            }));
        }
        let geometry = |_| {
            C2Refusal::Transport(D1Refusal::RingTooSmall {
                need: staging.at + payload.len(),
                have: staging.bank.len(),
            })
        };
        // Bulk, published by the descriptor and the fence below it.
        staging
            .bank
            .write_bytes_at(staging.at, payload)
            .map_err(geometry)?;

        let iova = staging.bank.iova().0 + staging.at as u64;
        let Some(desc) = Descriptor::transmit(iova, payload.len() as u32) else {
            return Err(C2Refusal::Transport(D1Refusal::Unreachable {
                iova,
                len: payload.len() as u64,
            }));
        };
        let tx = DmaShared::new(rings.bank(queue));
        for (word, value) in desc.words().into_iter().enumerate() {
            tx.write_u32_at(Descriptor::slot_at(slot) + word * 4, value)
                .map_err(|_| {
                    C2Refusal::Transport(D1Refusal::RingTooSmall {
                        need: crate::dma::DESC_BYTES,
                        have: rings.bank(queue).len(),
                    })
                })?;
        }
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        // **The doorbell names the next free slot, not the chunk count.**
        // `mt76_dma_kick_queue` writes `q->head`, which upstream advances
        // as `(head + 1) % ndesc` (`dma.c:232-235`). This wrote the raw
        // producer, which is the same number only while the download is
        // shorter than the ring: the ROM patch's forty-six chunks never
        // reached a hundred and twenty-eight, and the RAM code's chunk
        // 128 wrote `128` into a register whose slots end at 127.
        self.post_receive(queue, (producer % depth) as u16);
        Ok(())
    }

    /// The engine's consumed count for `queue`, absolute, given the last
    /// count read.
    ///
    /// `dma_idx` wraps at the ring's depth; a download does not. The
    /// engine cannot advance a whole lap between two publishes, so the
    /// absolute count is the unique value congruent to the register that
    /// lies at or after `last` — which makes this monotone and makes
    /// `producer - taken` an outstanding-descriptor count again.
    fn engine_taken(&self, queue: Queue, last: u32) -> u32 {
        let depth = u32::from(queue.depth());
        let index = self.engine_index(queue) % depth;
        last + (index + depth - (last % depth)) % depth
    }

    /// Ask the part for the ROM patch semaphore.
    fn patch_semaphore<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        take: bool,
        delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<Semaphore, C2Refusal> {
        let answer = self
            .exchange(
                &mut *rings,
                Command::PatchSem,
                &mcu::patch_sem_body(take),
                delay,
                now_ns,
            )
            .map_err(C2Refusal::Transport)?;
        Semaphore::from_status(answer.status).map_err(|_| C2Refusal::SemaphoreRefused)
    }

    /// **C2b — put the ROM patch into the part.**
    ///
    /// `mt76_connac2_load_patch` (`mt76_connac_mcu.c:3274-3350`), with its
    /// own control flow: take the semaphore, and if the part says the patch
    /// is already there, stop — that is upstream's success path, not a
    /// refusal. Otherwise per section an `init_download` naming the address
    /// and the composed mode word, then the payload in chunks that carry no
    /// header and are not answered, then `PATCH_FINISH_REQ`.
    ///
    /// The semaphore is released on **every** path out, including the
    /// failing ones: a held semaphore makes the next download refuse, and
    /// the next download is the one after a reboot.
    pub fn download_patch<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        patch: &Patch,
        image: &[u8],
        mut delay: impl FnMut() + Copy,
        now_ns: fn() -> Option<u64>,
    ) -> Result<PatchOutcome, C2Refusal> {
        let taken = self.patch_semaphore(&mut *rings, true, delay, now_ns)?;
        match taken {
            // Already in the part. Nothing to send, nothing to release:
            // upstream returns before the release path for this case.
            Semaphore::AlreadyLoaded => return Ok(PatchOutcome::AlreadyResident),
            Semaphore::Granted => {}
            Semaphore::Released | Semaphore::Refused => {
                return Err(C2Refusal::SemaphoreRefused);
            }
        }

        let sent = self.send_patch_sections(&mut *rings, patch, image, &mut delay, now_ns);

        // Release on every path, including the failing one. A held
        // semaphore outlives this boot.
        let released = self.patch_semaphore(&mut *rings, false, delay, now_ns);
        let outcome = sent?;
        match released {
            Ok(Semaphore::Released) => Ok(outcome),
            _ => Err(C2Refusal::SemaphoreRefused),
        }
    }

    /// **C2c — put the N9 RAM code into the part, and start it.**
    ///
    /// `mt76_connac_mcu_send_ram_firmware` (`mt76_connac_mcu.c:3090-3130`)
    /// then `mt76_connac_mcu_start_firmware`. Three differences from C2b,
    /// all of them load-bearing:
    ///
    /// 1. **No semaphore.** The patch is arbitrated between driver
    ///    instances; the RAM code is not. There is nothing to take and
    ///    nothing to release.
    /// 2. **A region may be skipped and still advance the cursor.**
    ///    `FW_FEATURE_NON_DL` regions — the 303,936-byte CLC blob on this
    ///    part — are read by the host and never handed over, but their
    ///    payload still occupies its extent in the image. Upstream reaches
    ///    its `offset += len` by `goto next` so that a skipped region
    ///    advances it too; [`crate::fw::RamRegion::offset`] records it at
    ///    parse time instead, so this loop does no arithmetic it could get
    ///    wrong.
    /// 3. **The start command is answered, and its answer matters.** The
    ///    chunks are not, so `FW_START_REQ` is the download's only
    ///    acknowledgement — the same role `PATCH_FINISH_REQ` plays in C2b,
    ///    and the place C2b's defect hid.
    ///
    /// **And the firmware being started is not the firmware running.** The
    /// MCU answering `FW_START_REQ` says it took the command. Only
    /// `MT_CONN_ON_MISC` reading `FW_PWR_ON | FW_N9_ON` says the code this
    /// driver just wrote actually executes, which is why that poll is part
    /// of this rung and not an optional extra.
    pub fn download_ram<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        ram: &RamCode,
        image: &[u8],
        mut delay: impl FnMut() + Copy,
        now_ns: fn() -> Option<u64>,
    ) -> Result<RamOutcome, C2Refusal> {
        let mut regions = 0u32;
        let mut chunks = 0u32;
        let mut bytes = 0u64;

        for (index, region) in ram.regions().iter().enumerate() {
            // Read but never downloaded, and its extent is already in the
            // next region's `offset`. Nothing to do but not send it.
            if !region.is_downloaded() {
                continue;
            }
            let (offs, len) = (region.offset as usize, region.len as usize);
            let Some(payload) = image.get(offs..offs.saturating_add(len)) else {
                return Err(C2Refusal::SectionOutOfRange {
                    offs: region.offset,
                    len: region.len,
                });
            };

            self.exchange(
                &mut *rings,
                Command::init_download(region.addr),
                &mcu::init_download_body(region.addr, region.len, region.dl_mode()),
                delay,
                now_ns,
            )
            .map_err(|d1| C2Refusal::DownloadRefused {
                addr: region.addr,
                // `enumerate` over at most `MAX_REGIONS`.
                section: index as u8,
                fwdl_idx: self.engine_index(Queue::Fwdl),
                inner: d1,
            })?;

            chunks += self.send_chunks(
                &mut *rings,
                payload,
                (region.offset, region.len),
                &mut delay,
                now_ns,
            )?;
            bytes += payload.len() as u64;
            regions += 1;
        }

        // Read before the command, so a refusal says where the download
        // actually got to rather than where it was aimed.
        let fwdl_idx = self.engine_index(Queue::Fwdl);
        let started = self
            .start_firmware(&mut *rings, ram.override_addr(), delay, now_ns)
            .map_err(|inner| C2Refusal::StartRefused { fwdl_idx, inner })?;
        if started.status != mcu::MCU_STATUS_OK {
            return Err(C2Refusal::StartStatus {
                status: started.status,
            });
        }

        Ok(RamOutcome {
            regions,
            chunks,
            bytes,
            entry: ram.override_addr(),
        })
    }

    /// Tell the part to run what was just written, and read its verdict.
    ///
    /// **A method, for the reason `finish_patch` is one.** The body is
    /// `mcu::fw_start_body(option, addr)` — eight bytes — and
    /// `mcu::fw_start_option` decides the option word from whether any
    /// region claimed `FW_FEATURE_OVERRIDE_ADDR`. Both existed, both were
    /// tested, and neither had a caller until this; that is the exact
    /// shape of the defect note 81 reported, so the call is placed where a
    /// test can reach it and the test asserts on what crosses the ring.
    fn start_firmware<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        entry: Option<u32>,
        delay: impl FnMut() + Copy,
        now_ns: fn() -> Option<u64>,
    ) -> Result<mcu::Response, D1Refusal> {
        // `mt76_connac_mcu_send_ram_firmware` keeps the **last** overriding
        // region's address and passes it with `FW_START_OVERRIDE`; with no
        // such region the command takes address zero and no option.
        let option = mcu::fw_start_option(entry);
        let addr = entry.unwrap_or(0);
        self.exchange(
            rings,
            Command::FwStart,
            &mcu::fw_start_body(option, addr),
            delay,
            now_ns,
        )
    }

    /// Wait for the N9 core to report itself powered and up.
    ///
    /// The part's own word that the firmware runs, and the only evidence
    /// in C2 that is a register rather than a message. Returns what
    /// `MT_CONN_ON_MISC` last read, so a success can be recorded and a
    /// timeout can say *how* it failed: `0x0` is a core that never took
    /// power, `0x1` is one that powered and did not come up.
    pub fn wait_firmware_ready(
        &self,
        mut delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<u32, C2Refusal> {
        let mut budget = Budget::new(N9_BUDGET_NS, N9_SPINS_UNCLOCKED, now_ns);
        loop {
            if self.firmware_ready() {
                return Ok(self.firmware_state());
            }
            if !budget.may_wait() {
                return Err(C2Refusal::FirmwareNotReady {
                    misc: self.firmware_state(),
                });
            }
            delay();
        }
    }

    /// Publish one region's payload as `FW_SCATTER` chunks, and say how
    /// many crossed.
    ///
    /// **The patch's sections and the RAM code's regions are the same
    /// mechanism**, and this is it: chunk at [`mcu::FW_SCATTER_MAX`],
    /// advance the ring's own producer, publish, walk away. C2b had it
    /// inline inside `send_patch_sections`; C2c would have needed a second
    /// copy, and a second copy of a loop that owns a hardware index is how
    /// the index stops being owned.
    ///
    /// `extent` is the region's `(offs, len)` in the image, carried only so
    /// a refusal can name where it was reading.
    fn send_chunks<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        payload: &[u8],
        extent: (u32, u32),
        delay: &mut (impl FnMut() + Copy),
        now_ns: fn() -> Option<u64>,
    ) -> Result<u32, C2Refusal> {
        let mut chunks = 0u32;
        for chunk in payload.chunks(mcu::FW_SCATTER_MAX) {
            let Some(raw) = mcu::scatter(chunk) else {
                return Err(C2Refusal::SectionOutOfRange {
                    offs: extent.0,
                    len: extent.1,
                });
            };
            // **The producer belongs to the ring, not to this call.** See
            // [`crate::d1::Conversation::fwdl_next`]: a download that
            // restarts it writes a doorbell below the engine's own index,
            // and the lapping guard below cannot catch that because a
            // restarted producer satisfies it trivially.
            rings.conv.fwdl_next += 1;
            let producer = rings.conv.fwdl_next;
            self.send_payload(rings, Queue::Fwdl, raw, producer, *delay, now_ns)?;
            chunks += 1;
        }
        Ok(chunks)
    }

    /// The download proper, factored out so the semaphore release above can
    /// be a single unconditional statement rather than repeated on each
    /// early return.
    fn send_patch_sections<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        patch: &Patch,
        image: &[u8],
        delay: &mut (impl FnMut() + Copy),
        now_ns: fn() -> Option<u64>,
    ) -> Result<PatchOutcome, C2Refusal> {
        let mut chunks = 0u32;
        let mut bytes = 0u64;

        for (index, section) in patch.sections().iter().enumerate() {
            // `info.len`, never `size` — the record carries both and
            // upstream downloads the former.
            let (offs, len) = (section.offs as usize, section.len as usize);
            let Some(payload) = image.get(offs..offs.saturating_add(len)) else {
                return Err(C2Refusal::SectionOutOfRange {
                    offs: section.offs,
                    len: section.len,
                });
            };

            // "A region of this length is coming to this address." Answered,
            // so it rides `exchange`.
            self.exchange(
                &mut *rings,
                // The request kind is read off the *address*, not off
                // which blob is being sent — `Command::init_download` owns
                // that decision and a per-phase flag would get it wrong.
                Command::init_download(section.addr),
                &mcu::init_download_body(section.addr, section.len, section.data_mode()),
                *delay,
                now_ns,
            )
            .map_err(|d1| C2Refusal::DownloadRefused {
                addr: section.addr,
                // `enumerate` over at most `MAX_REGIONS`, so the narrowing
                // cannot lose a section this loop can reach.
                section: index as u8,
                // **Read on the failure path, not before it.** The
                // question this rung cannot otherwise answer is whether
                // the engine fetched the chunks published for the
                // *previous* section, and the index is only evidence if it
                // is sampled when the request has already failed.
                fwdl_idx: self.engine_index(Queue::Fwdl),
                inner: d1,
            })?;

            chunks += self.send_chunks(rings, payload, (section.offs, section.len), delay, now_ns)?;
            bytes += payload.len() as u64;
        }

        // **The patch is complete — and this is the download's only
        // acknowledgement.** Nothing answered a chunk, so if the engine
        // never fetched them this is where it shows. The engine's index is
        // read before the command so the refusal carries where the download
        // actually got to rather than where it was aimed.
        let fwdl_idx = self.engine_index(Queue::Fwdl);
        let finished = self
            .finish_patch(&mut *rings, *delay, now_ns)
            .map_err(|_| C2Refusal::FinishRefused { fwdl_idx })?;
        // `mt7925_mcu_parse_response` returns this byte as the command's
        // error for `PATCH_FINISH_REQ`; a part that says no here has not
        // taken the patch, whatever crossed the ring.
        if finished.status != mcu::MCU_STATUS_OK {
            return Err(C2Refusal::FinishStatus {
                status: finished.status,
            });
        }

        Ok(PatchOutcome::Downloaded {
            sections: patch.sections().len() as u32,
            chunks,
            bytes,
        })
    }

    /// Tell the part the patch is complete, and read its verdict.
    ///
    /// **A method, not three lines inside `send_patch_sections`**, because
    /// the body it sends is the thing that was wrong and nothing could
    /// reach it to check. `mcu::patch_finish_body` existed, was tested,
    /// and had no caller: `send_patch_sections` passed `&[]`.
    fn finish_patch<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        delay: impl FnMut() + Copy,
        now_ns: fn() -> Option<u64>,
    ) -> Result<mcu::Response, D1Refusal> {
        self.exchange(
            rings,
            Command::PatchFinish,
                // **Four bytes, not none.** `mt76_connac_mcu_start_patch`
                // sends a `check_crc` byte and three reserved
                // (`mt76_connac_mcu.c:37-49`). `mcu::patch_finish_body`
                // was written for this call and this call sent `&[]`
                // instead — a seam built, tested and never wired to its
                // one caller.
                //
                // An empty body is not merely short. `exchange` stages the
                // body after the descriptor in a buffer it reuses, so the
                // part read whatever the previous command had left at that
                // offset as `check_crc`, and then answered a request to
                // verify a CRC it was never given. That is note 81 on
                // 2026-11-11: status 2, from a part whose transport,
                // semaphore, download and finish round trip all worked.
                &mcu::patch_finish_body(),
            delay,
            now_ns,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d1::Conversation;
    use crate::mock::{MockMt7925, ReachablePool};

    /// Bytes per staging slot in the fixtures.
    ///
    /// Not [`mcu::FW_SCATTER_MAX`]: the slab is `Fwdl.depth()` of these and
    /// a host test does not need half a megabyte to prove slot *n* owns its
    /// own bytes. It must stay larger than any chunk these tests send.
    const TEST_CHUNK: usize = 256;

    /// Build the five buffers a download needs.
    fn rings(pool: &ReachablePool) -> (crate::mock::ReachableBuf, crate::mock::ReachableBuf,
                                       crate::mock::ReachableBuf, crate::mock::ReachableBuf,
                                       crate::mock::ReachableBuf) {
        (
            // **A real ring, not one descriptor.** These fixtures used to
            // allocate a single slot, which was only ever enough because
            // the code under test wrote slot zero forever. A fixture sized
            // to the bug cannot catch it.
            pool.alloc(Queue::Fwdl.table_bytes()),
            // **And a real slab, not one buffer**, for the same reason: a
            // single staging buffer is enough only while every chunk is
            // waited for, and waiting for a chunk is the belief this module
            // stopped holding.
            pool.alloc(Queue::Fwdl.depth() as usize * TEST_CHUNK),
            pool.alloc(mcu::FW_SCATTER_MAX),
            pool.alloc(Queue::EventWm.table_bytes()),
            pool.alloc(2048),
        )
    }

    /// **The start carries its eight-byte body, and the right eight.**
    ///
    /// Note 81's defect, written against the one other command in C2 that
    /// has a body and an answer. `mcu::fw_start_body` and
    /// `mcu::fw_start_option` were in the same condition
    /// `mcu::patch_finish_body` was in when C2b shipped — correct, tested,
    /// and with no caller — so the test that matters is not that they
    /// return the right bytes but that those bytes **cross the ring**.
    ///
    /// The staging buffer is poisoned first, because that is what made the
    /// finish's empty body a wrong value rather than an absent one:
    /// `exchange` stages behind the descriptor in a buffer it reuses, so
    /// an unwritten body is the previous command's bytes.
    #[test]
    fn the_firmware_start_sends_the_body_upstream_sends() {
        let dev = MockMt7925::new();
        let core = crate::d0::Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);

        let answers = DmaShared::new(&rx_buffers);
        answers
            .write_u8_at(36, crate::mcu::Command::FwStart.id())
            .unwrap();
        answers.write_u8_at(37, 1).unwrap();
        answers.write_u8_at(40, crate::mcu::MCU_STATUS_OK).unwrap();
        let posted = Descriptor::receive(rx_buffers.iova().0, 2048).expect("reachable");
        DmaShared::new(&rx_table)
            .write_u32_at(
                Descriptor::ctrl_at(0),
                posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
            )
            .unwrap();

        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        let staged = DmaShared::new(&tx_payload);
        for i in 0..80 {
            staged.write_u8_at(i, 0xAB).unwrap();
        }

        // The distribution's image overrides to `0x90d000`.
        const ENTRY: u32 = 0x0090_d000;
        let answer = core
            .start_firmware(&mut r, Some(ENTRY), || {}, || None)
            .expect("the staged answer");
        assert_eq!(answer.status, crate::mcu::MCU_STATUS_OK);

        let header = crate::mcu::command(crate::mcu::Command::FwStart, 1, 8).len();
        let desc = DmaShared::new(&tx_table);
        let ctrl = desc.read_u32_at(Descriptor::ctrl_at(0)).unwrap();
        assert_eq!(
            Descriptor::from_ctrl(ctrl).written() as usize,
            header + 8,
            "the start crossed the ring without its body",
        );

        // Option first, then the address — the reverse of
        // `init_download_body`'s order, transcribed and not rationalised.
        let want = crate::mcu::fw_start_body(crate::mcu::fw_start_option(Some(ENTRY)), ENTRY);
        for (i, byte) in want.iter().enumerate() {
            assert_eq!(
                staged.read_u8_at(header + i).unwrap(),
                *byte,
                "byte {i} of the start body",
            );
        }
        // And the option word is genuinely the override, not zero: an
        // image that names an entry point and a start that does not say so
        // runs the wrong address.
        assert_ne!(
            crate::mcu::fw_start_option(Some(ENTRY)),
            0,
            "an overriding image must set the option word",
        );
    }

    /// **A region the part does not download is skipped, and the cursor
    /// still passes it.**
    ///
    /// The CLC blob is 303,936 bytes — 21.8% of the file — that the host
    /// reads and the part never sees. The hazard is not skipping it; it is
    /// skipping it *and* leaving the image cursor behind, which would hand
    /// every later region the wrong bytes. Upstream reaches its
    /// `offset += len` by `goto next` precisely so a skipped region still
    /// advances it; this driver records the offset at parse time, and this
    /// asserts that `download_ram` uses that recorded offset rather than a
    /// running sum of its own.
    ///
    /// So the fixture puts the **downloaded** region second: a loop that
    /// counted its own bytes would read region one's payload from offset
    /// zero and send the skipped region's bytes to the part.
    #[test]
    fn a_region_the_part_does_not_download_is_skipped_without_losing_the_cursor() {
        use crate::fw::tests::ram_image_with;

        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();

        // Region 0: `FW_FEATURE_NON_DL`, `FW_TYPE_CLC` — read, never sent.
        // Region 1: downloaded, and the one claiming the entry point.
        const SIZE: u32 = 64;
        let mut raw = [0u8; 1024];
        let len = ram_image_with(&mut raw, 2, SIZE, 0, &[(0x40, 2), (0x21, 0)]);
        // Make the two payloads distinguishable on the wire.
        for i in 0..SIZE as usize {
            raw[i] = 0xCC;
            raw[SIZE as usize + i] = 0x5A;
        }
        let image = &raw[..len];
        let ram = RamCode::parse(image).expect("the fixture's shape");
        assert_eq!(ram.regions().len(), 2);
        assert!(!ram.regions()[0].is_downloaded(), "region 0 is NON_DL");
        assert_eq!(ram.regions()[1].offset, SIZE, "the cursor passed region 0");

        let tx_table = pool.alloc(Queue::McuWm.table_bytes());
        let fwdl_table = pool.alloc(Queue::Fwdl.table_bytes());
        let fwdl_payload = pool.alloc(Queue::Fwdl.depth() as usize * TEST_CHUNK);
        let tx_payload = pool.alloc(mcu::FW_SCATTER_MAX);
        let rx_table = pool.alloc(Queue::EventWm.table_bytes());
        const SLOT: usize = 2048;
        let rx_buffers = pool.alloc(SLOT * 2);

        // Two answers: the region's `init_download`, then `FW_START_REQ`.
        let answers = DmaShared::new(&rx_buffers);
        let table = DmaShared::new(&rx_table);
        let region_addr = ram.regions()[1].addr;
        for (slot, (cmd, seq)) in [
            (Command::init_download(region_addr).id(), 1u8),
            (Command::FwStart.id(), 2u8),
        ]
        .into_iter()
        .enumerate()
        {
            let at = slot * SLOT;
            answers.write_u8_at(at + 36, cmd).unwrap();
            answers.write_u8_at(at + 37, seq).unwrap();
            answers.write_u8_at(at + 40, crate::mcu::MCU_STATUS_OK).unwrap();
            let posted =
                Descriptor::receive(rx_buffers.iova().0 + at as u64, SLOT as u32).expect("reachable");
            table
                .write_u32_at(
                    Descriptor::ctrl_at(slot),
                    posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
                )
                .unwrap();
        }

        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &fwdl_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: SLOT,
            data: None,
            conv: &mut conv,
        };

        let out = core
            .download_ram(&mut r, &ram, image, || {}, || None)
            .expect("the staged answers");

        assert_eq!(out.regions, 1, "only the downloaded region was sent");
        assert_eq!(out.chunks, 1);
        assert_eq!(out.bytes, u64::from(SIZE));
        assert_eq!(out.entry, Some(region_addr), "the overriding region's address");

        // **The bytes that crossed are the downloaded region's**, not the
        // CLC blob's — the whole point of the recorded offset.
        let staged = DmaShared::new(&fwdl_payload);
        for i in 0..SIZE as usize {
            assert_eq!(
                staged.read_u8_at(i).unwrap(),
                0x5A,
                "byte {i} came from the skipped region",
            );
        }
    }

    /// **A firmware chunk crosses with no MCU header at all.**
    ///
    /// The first of the three facts this module exists for.
    /// `mt7925_mcu_fill_message` returns early for `FW_SCATTER`, so the
    /// chunk is the whole payload. Sending it through `exchange` would put
    /// `MCU_TXD_BYTES` of header in front of it, and the part would read
    /// the header as firmware.
    /// **The finish carries its four-byte body.**
    ///
    /// Note 81 on 2026-11-11: the part answered `PATCH_FINISH_REQ` with
    /// status 2, after a semaphore, two download requests, forty-six
    /// chunks and a finish round trip that all worked. The request was
    /// sent with `&[]`. `mt76_connac_mcu_start_patch` sends a `check_crc`
    /// byte and three reserved (`mt76_connac_mcu.c:37-49`), and
    /// `mcu::patch_finish_body` existed for exactly that — tested, and
    /// with no caller.
    ///
    /// An empty body is worse than short: `exchange` stages the body
    /// after the descriptor in a buffer it reuses, so the part read the
    /// *previous* command's bytes as `check_crc` and was asked to verify
    /// a CRC it had never been given.
    ///
    /// So this asserts on what crosses the ring — the descriptor's own
    /// length, and the four bytes behind the command header — because a
    /// test that merely called `patch_finish_body()` is the test that was
    /// already there when this shipped.
    #[test]
    fn the_patch_finish_sends_the_body_upstream_sends() {
        let dev = MockMt7925::new();
        let core = crate::d0::Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);

        // The part's reply: this driver's first sequence, and a verdict of
        // "took it", so the call returns rather than refusing.
        let answers = DmaShared::new(&rx_buffers);
        answers
            .write_u8_at(36, crate::mcu::Command::PatchFinish.id())
            .unwrap();
        answers.write_u8_at(37, 1).unwrap();
        answers.write_u8_at(40, crate::mcu::MCU_STATUS_OK).unwrap();
        let posted = Descriptor::receive(rx_buffers.iova().0, 2048).expect("reachable");
        DmaShared::new(&rx_table)
            .write_u32_at(
                Descriptor::ctrl_at(0),
                posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
            )
            .unwrap();

        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        // **Poison the staging buffer first.** This is what made the empty
        // body a wrong value rather than an absent one: the bytes behind
        // the header are whatever the last command left there.
        let staged = DmaShared::new(&tx_payload);
        for i in 0..72 {
            staged.write_u8_at(i, 0xAB).unwrap();
        }

        let answer = core
            .finish_patch(&mut r, || {}, || None)
            .expect("the staged answer");
        assert_eq!(answer.status, crate::mcu::MCU_STATUS_OK);

        // The descriptor names the header **plus four bytes**.
        let header = crate::mcu::command(crate::mcu::Command::PatchFinish, 1, 4).len();
        let desc = DmaShared::new(&tx_table);
        let ctrl = desc.read_u32_at(Descriptor::ctrl_at(0)).unwrap();
        assert_eq!(
            Descriptor::from_ctrl(ctrl).written() as usize,
            header + 4,
            "the finish crossed the ring with no body",
        );

        // And those four bytes are `check_crc` and its reserved, zeroed —
        // not the poison still sitting around them.
        for i in 0..4 {
            assert_eq!(
                staged.read_u8_at(header + i).unwrap(),
                0,
                "byte {i} of the finish body was never written",
            );
        }
    }

    #[test]
    fn a_firmware_chunk_carries_no_header() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        let chunk: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(3).wrapping_add(7));
        core.send_payload(&mut r, Queue::Fwdl, &chunk, 1, || {}, || None)
            .expect("the engine took it");

        // The staged bytes are the chunk, from byte zero of slot zero's
        // own buffer.
        let staged = DmaShared::new(&fwdl_payload);
        for (i, want) in chunk.iter().enumerate() {
            assert_eq!(
                staged.read_u8_at(i).unwrap(),
                *want,
                "byte {i} is not the chunk's own"
            );
        }
        // And the descriptor describes exactly the chunk's length — not the
        // chunk plus a header.
        let published = DmaShared::new(&tx_table);
        let ctrl = published.read_u32_at(Descriptor::ctrl_at(0)).unwrap();
        assert_eq!(Descriptor::from_ctrl(ctrl).written(), chunk.len() as u32);
    }

    /// **A chunk lands in the table the engine was told to read.**
    ///
    /// The download drives two queues, and `Rings` used to carry one
    /// descriptor table. Chunks were written into `McuWm`'s table while
    /// `Fwdl`'s doorbell was rung, so the engine read the base it had
    /// actually been programmed with, found zeros, and stopped — a stall
    /// at chunk zero, with every unit test passing, because none of them
    /// connected a queue's doorbell to its programmed base.
    ///
    /// This one does: it programs each queue with a distinct table and
    /// then checks the bytes went to the right one.
    #[test]
    fn a_chunk_lands_in_the_table_its_queue_was_programmed_with() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let mcu_table = pool.alloc(Queue::McuWm.table_bytes());
        let fwdl_table = pool.alloc(Queue::Fwdl.table_bytes());
        let fwdl_payload = pool.alloc(Queue::Fwdl.depth() as usize * TEST_CHUNK);
        let tx_payload = pool.alloc(mcu::FW_SCATTER_MAX);
        let rx_table = pool.alloc(Queue::EventWm.table_bytes());
        let rx_buffers = pool.alloc(2048);

        // Program each queue with its own base, as the binding does.
        core.program_queue(Queue::McuWm, mcu_table.iova().0, mcu_table.len() as u64)
            .expect("mcu ring");
        core.program_queue(Queue::Fwdl, fwdl_table.iova().0, fwdl_table.len() as u64)
            .expect("fwdl ring");

        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &mcu_table,
            fwdl_table: &fwdl_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        core.send_payload(&mut r, Queue::Fwdl, b"chunk", 1, || {}, || None)
            .expect("taken");

        // The descriptor is in the download ring...
        let fwdl = DmaShared::new(&fwdl_table);
        assert_eq!(
            fwdl.read_u32_at(Descriptor::slot_at(0)).unwrap(),
            fwdl_payload.iova().0 as u32,
            "the chunk did not land in the queue's own table"
        );
        // ...and the command ring is untouched.
        let mcu_ring = DmaShared::new(&mcu_table);
        assert_eq!(
            mcu_ring.read_u32_at(Descriptor::slot_at(0)).unwrap(),
            0,
            "the chunk was written into the command ring"
        );
    }

    /// **Each chunk goes in its own ring slot.**
    ///
    /// The engine consumes slot `dma_idx` and advances, so after the
    /// first chunk it reads slot one. Every chunk written into slot zero
    /// leaves it reading zeroed pool memory — no address, no length, no
    /// last-segment bit — and it stops. That is note 78, "the engine
    /// stopped taking firmware chunks mid-download", and it cost a boot.
    ///
    /// `the_doorbell_counts_chunks_not_sends` below checks the doorbell
    /// and passed throughout: the doorbell was never the broken half.
    #[test]
    fn each_chunk_lands_in_its_own_slot() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(Queue::Fwdl.table_bytes());
        let fwdl_payload = pool.alloc(Queue::Fwdl.depth() as usize * TEST_CHUNK);
        let tx_payload = pool.alloc(mcu::FW_SCATTER_MAX);
        let rx_table = pool.alloc(crate::dma::DESC_BYTES);
        let rx_buffers = pool.alloc(2048);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        for producer in 1..=3u32 {
            core.send_payload(&mut r, Queue::Fwdl, b"chunk", producer, || {}, || None)
                .expect("taken");
        }

        // Three distinct slots, each describing **its own** staging
        // buffer. One address repeated three times would mean three
        // descriptors naming bytes that were overwritten twice — the part
        // reading the last chunk three times over — which is what a single
        // staging buffer gives once nothing waits per chunk.
        let published = DmaShared::new(&tx_table);
        for n in 0..3usize {
            let buf0 = published.read_u32_at(Descriptor::slot_at(n)).unwrap();
            assert_eq!(
                buf0,
                fwdl_payload.iova().0 as u32 + (n * TEST_CHUNK) as u32,
                "slot {n} does not name its own staging buffer"
            );
            let ctrl = published.read_u32_at(Descriptor::ctrl_at(n)).unwrap();
            assert_eq!(Descriptor::from_ctrl(ctrl).written(), 5);
        }
    }

    /// **The doorbell is an absolute index, not an increment.**
    ///
    /// `cpu_idx` is where the producer has got to, so chunk forty rings
    /// forty and not one. A driver that wrote 1 every time would publish
    /// every chunk into slot zero and advertise one descriptor forever.
    #[test]
    fn the_doorbell_counts_chunks_not_sends() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        for producer in 1..=4u32 {
            core.send_payload(&mut r, Queue::Fwdl, b"chunk", producer, || {}, || None)
                .expect("taken");
        }
        let cell = dev.cell_of(Queue::Fwdl.regs().cpu_idx());
        let rung: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == cell)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(rung, alloc::vec![1, 2, 3, 4]);
    }

    /// **A second download continues the doorbell the first one left.**
    ///
    /// Flash six's defect, one queue over, caught before metal this time.
    /// `send_patch_sections` kept the `Fwdl` producer in a local, which is
    /// right for exactly one download: `enable_engine` has just zeroed the
    /// hardware indices, so zero is the truth. C2c is a second download
    /// over the same started engine, and a local would have begun again at
    /// one while the engine's `dma_idx` sat at forty-six.
    ///
    /// **It would not have refused.** `send_payload` guards a lapped ring
    /// by asking whether `engine_index + depth >= producer`; a producer
    /// that has just restarted passes that trivially, so the download
    /// would have rewritten slots the engine still held and reported every
    /// chunk sent. The only thing that catches it is the counter having an
    /// owner, so that is what this asserts: two passes, one
    /// `Conversation`, and a doorbell that counts 1..=4 rather than
    /// 1, 2, 1, 2.
    #[test]
    fn a_second_download_over_the_same_part_continues_the_doorbell() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        // One chunk per pass — the fixture's staging slot is `TEST_CHUNK`,
        // far below `FW_SCATTER_MAX`. Two passes is the whole point: the
        // patch's download and then the RAM code's, over one started
        // engine, which is all this ring can tell apart.
        let payload = [0u8; 200];
        let mut delay = || {};
        for pass in 1..=2u32 {
            let sent = core
                .send_chunks(&mut r, &payload, (0, payload.len() as u32), &mut delay, || None)
                .unwrap_or_else(|e| panic!("download {pass}: {e:?}"));
            assert_eq!(sent, 1);
        }

        assert_eq!(r.conv.fwdl_next, 2, "the ring's own producer");
        let cell = dev.cell_of(Queue::Fwdl.regs().cpu_idx());
        let rung: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == cell)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(
            rung,
            alloc::vec![1, 2],
            "the second download restarted the doorbell",
        );
    }

    /// **A download longer than the ring keeps its doorbell inside it.**
    ///
    /// C2c's first flash, 2026-11-12: note 79 at region 2, `fwdl idx 125`
    /// against 159 chunks published. 46 of the ROM patch and 113 of the
    /// RAM code had crossed, and the download stopped at the lap.
    ///
    /// `cpu_idx` and `dma_idx` are **ring indices** — `program_queue`
    /// tells the engine `ring_size = depth`, and upstream advances
    /// `q->head = (q->head + 1) % q->ndesc` before writing it
    /// (`dma.c:232-235`). This wrote the raw chunk count, which is the
    /// same number only while the download is shorter than the ring. The
    /// ROM patch is 46 chunks on a 128-slot ring and never reached it;
    /// the RAM code's chunk 128 wrote `128` into a register whose slots
    /// end at 127.
    ///
    /// **Nothing in the ladder could have caught it.** Every fixture sent
    /// fewer chunks than the ring is deep, which is the shape of the bug
    /// written into the tests. So this one sends more.
    #[test]
    fn a_download_longer_than_the_ring_keeps_its_doorbell_inside_it() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        let depth = u32::from(Queue::Fwdl.depth());
        let over = depth + 5;
        let payload = [0u8; 64];
        for _ in 0..over {
            r.conv.fwdl_next += 1;
            let producer = r.conv.fwdl_next;
            core.send_payload(&mut r, Queue::Fwdl, &payload, producer, || {}, || None)
                .unwrap_or_else(|e| panic!("chunk {producer}: {e:?}"));
        }

        let cell = dev.cell_of(Queue::Fwdl.regs().cpu_idx());
        let rung: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == cell)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(rung.len(), over as usize, "one doorbell per chunk");

        // **Every index the part is handed names a slot it owns.** This is
        // the assertion metal made for us: 128 is not a slot of a 128-slot
        // ring.
        for (i, value) in rung.iter().enumerate() {
            assert!(
                *value < depth,
                "chunk {} rang the doorbell at {value}, outside a {depth}-slot ring",
                i + 1,
            );
        }
        // And it wraps where the ring does, rather than merely being
        // clamped: chunk `depth` names slot 0, and the five after it
        // count on from there.
        assert_eq!(rung[(depth - 1) as usize], 0, "the lap returns to slot 0");
        assert_eq!(&rung[depth as usize..], &[1, 2, 3, 4, 5]);
        // The absolute count is untouched — it is what the ladder reports
        // and what `fwdl_taken` is measured against.
        assert_eq!(r.conv.fwdl_next, over);
    }

    /// **An engine that keeps up across two laps is not refused.**
    ///
    /// The other half of the wrapping-index defect, and the one that
    /// would have replaced a working download with a false refusal.
    /// `send_payload`'s guard asks whether `dma_idx + depth >= producer`.
    /// With `dma_idx` read raw it is a slot, so an engine healthily
    /// consuming its 200th chunk reads back 72 — and `72 + 128 >= 260` is
    /// false, which is `ChunkNotTaken` reported against an engine that
    /// had taken everything.
    ///
    /// So the comparison needs a count, and `fwdl_taken` recovers one.
    /// This drives the RAM code's own distance — past two laps — with the
    /// engine consuming, and asserts the recovered count is the real one.
    #[test]
    fn an_engine_that_keeps_up_across_two_laps_is_not_refused() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        // 267 is the distribution's RAM code, and laps the 128-slot ring
        // twice over.
        const CHUNKS: u32 = 267;
        let payload = [0u8; 64];
        for _ in 0..CHUNKS {
            r.conv.fwdl_next += 1;
            let producer = r.conv.fwdl_next;
            core.send_payload(&mut r, Queue::Fwdl, &payload, producer, || {}, || None)
                .unwrap_or_else(|e| panic!("chunk {producer} of {CHUNKS}: {e:?}"));
        }
        assert_eq!(r.conv.fwdl_next, CHUNKS);
        // The engine consumed each chunk as it was handed over, so the
        // recovered count is the whole download — not the slot it happens
        // to sit on, which is `267 % 128 == 11`.
        assert_eq!(
            r.conv.fwdl_taken,
            CHUNKS - 1,
            "the absolute count was lost at a lap",
        );
        assert_ne!(r.conv.fwdl_taken, CHUNKS % 128);
    }

    /// **A chunk is never answered, so nothing waits for one.**
    ///
    /// With the engine consuming, the send returns without the event ring
    /// ever being touched. If this ever starts reading the event queue's
    /// bank, a forty-six-chunk patch has become forty-six twenty-second
    /// waits — a quarter-hour bind that looks exactly like a hang.
    #[test]
    fn nothing_waits_on_the_event_ring_for_a_chunk() {
        let dev = MockMt7925::new();
        dev.tx_consumes.set(true);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        dev.reads.borrow_mut().clear();
        core.send_payload(&mut r, Queue::Fwdl, b"chunk", 1, || {}, || None)
            .expect("published");
        let event_bank = Queue::EventWm.regs();
        for read in dev.reads.borrow().iter() {
            assert_ne!(*read, event_bank.dma_idx(), "waited on the event ring");
            assert_ne!(*read, event_bank.cpu_idx(), "touched the event ring");
            // And not the download ring's own index either. A chunk inside
            // the ring is published and walked away from; reading this
            // register is how forty-six chunks became forty-six one-second
            // waits against a part that never moved it.
            assert_ne!(
                *read,
                Queue::Fwdl.regs().dma_idx(),
                "waited for an engine to acknowledge a chunk"
            );
        }
    }

    /// An engine that frees nothing refuses — but **only once the ring
    /// has lapped**, and then it says which chunk could not be placed.
    #[test]
    fn an_engine_that_frees_nothing_refuses_when_the_ring_laps() {
        let dev = MockMt7925::new();
        // `tx_consumes` left OFF: the honest default, an engine this mock
        // does not claim to know the behaviour of.
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, fwdl_payload, tx_payload, rx_table, rx_buffers) = rings(&pool);
        let mut conv = Conversation::new();
        let mut r = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &fwdl_payload,
            fwdl_chunk_bytes: TEST_CHUNK,
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let depth = Queue::Fwdl.depth() as u32;
        // **Every chunk up to a full ring goes out against an engine that
        // acknowledges nothing.** This is the whole change: the download
        // does not depend on `dma_idx` moving until the ring's geometry
        // says it must, and the ROM patch's forty-six chunks never reach
        // that point.
        let mut probes = 0u32;
        for producer in 1..=depth {
            core.send_payload(&mut r, Queue::Fwdl, b"chunk", producer, || probes += 1, || None)
                .expect("a slot the engine cannot still hold");
        }
        assert_eq!(probes, 0, "waited for a chunk nothing ever answers");

        // One more laps the ring, and that one does have to wait.
        let out = core.send_payload(
            &mut r,
            Queue::Fwdl,
            b"chunk",
            depth + 1,
            || probes += 1,
            || None,
        );
        assert_eq!(
            out,
            Err(C2Refusal::ChunkNotTaken {
                chunk: depth,
                dma_idx: 0
            })
        );
        assert!(probes > 0, "lapped the ring without waiting for a slot");
        assert!(probes <= CHUNK_SPINS_UNCLOCKED, "budgeted, not spun");
    }

    /// A chunk longer than the staging buffer is refused rather than
    /// truncated, and `scatter` bounds it at the vendor's own `max_len`.
    #[test]
    fn an_oversized_chunk_is_refused() {
        assert!(mcu::scatter(&[0u8; mcu::FW_SCATTER_MAX]).is_some());
        assert!(mcu::scatter(&[0u8; mcu::FW_SCATTER_MAX + 1]).is_none());
        assert!(mcu::scatter(&[]).is_none());
    }

    /// **Every C2 rung is distinct, and none collides with a D1 rung.**
    ///
    /// They share one note band, so a collision would report the wrong
    /// repair on the only screen that can report anything.
    #[test]
    fn the_c2_rungs_continue_the_d1_ladder_without_colliding() {
        use crate::d1::D1Refusal;
        let c2 = [
            C2Refusal::SemaphoreRefused,
            C2Refusal::SectionOutOfRange { offs: 0, len: 0 },
            C2Refusal::ChunkNotTaken { chunk: 0, dma_idx: 0 },
            C2Refusal::DownloadRefused {
                addr: 0,
                section: 0,
                fwdl_idx: 0,
                inner: D1Refusal::NoAnswer {
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
            C2Refusal::StartRefused {
                fwdl_idx: 0,
                inner: D1Refusal::NoAnswer {
                    tx_idx: 0,
                    seq: 0,
                    dma_idx: 0,
                    tx_next: 0,
                    rx_next: 0,
                    skipped: 0,
                    data_pending: None,
                },
            },
            C2Refusal::StartStatus { status: 0 },
            C2Refusal::FirmwareNotReady { misc: 0 },
        ];
        let d1 = [
            D1Refusal::Unreachable { iova: 0, len: 0 },
            D1Refusal::BaseNotLatched { wrote: 0, read: 0 },
            D1Refusal::SizeNotLatched { wrote: 0, read: 0 },
            D1Refusal::EngineBusy { glo_cfg: 0 },
            D1Refusal::RxNotTaken { dma_idx: 0, posted: 0 },
            D1Refusal::CommandTooLong { bytes: 0, room: 0 },
            D1Refusal::NoAnswer { seq: 0, dma_idx: 0, tx_idx: 0, tx_next: 0, rx_next: 0, skipped: 0, data_pending: None },
            D1Refusal::BadAnswer(mcu::McuError::TooShort),
            D1Refusal::RingTooSmall { need: 0, have: 0 },
            D1Refusal::WfsysNotReady { sw_rst: 0 },
            D1Refusal::NotOwned { lpctl: 0 },
        ];
        let mut seen = [false; 256];
        for r in d1 {
            seen[r.rung() as usize] = true;
        }
        for r in c2 {
            let n = r.rung();
            assert_ne!(n, 0, "zero reads as `no refusal`");
            assert!(!seen[n as usize], "C2 rung {n} collides with a D1 rung");
            seen[n as usize] = true;
        }
        // A transport refusal keeps the D1 rung it came from, rather than
        // being flattened into one "the download failed".
        assert_eq!(
            C2Refusal::Transport(D1Refusal::NotOwned { lpctl: 0 }).rung(),
            D1Refusal::NotOwned { lpctl: 0 }.rung()
        );
    }
}
