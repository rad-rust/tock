// Licensed under the Apache License, Version 2.0 or the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright Tock Contributors 2022.

//! Lockstep RNG interposer for the qemu_rv32_virt dual-hart configuration.
//!
//! [`LockstepRng`] implements [`kernel::hil::rng::Rng`] on both harts:
//!
//! - **Hart 0**: wraps the real `VirtIORng`. Intercepts each
//!   [`randomness_available`][kernel::hil::rng::Client::randomness_available]
//!   callback, copies the bytes into [`RNG_REPLAY_BUF`][crate::chip::RNG_REPLAY_BUF],
//!   pushes a [`SyncEntry::RngReady`][crate::chip::SyncEntry::RngReady] onto
//!   [`LOCKSTEP_CHAN`][crate::chip::LOCKSTEP_CHAN], then forwards the same bytes
//!   downstream to the unmodified [`RngDriver`][capsules_core::rng::RngDriver].
//!
//! - **Hart 1**: a pure-software stub. `get()` records a `pending` flag.
//!   When the corresponding `RngReady` is popped from the channel in
//!   `main_secondary()`'s drain loop, [`replay_rng_done_for_hart1`] fires
//!   the stored client callback with the forwarded bytes — giving `RngDriver`
//!   on hart 1 the same bytes `RngDriver` on hart 0 already received.
//!
//! Both harts expose an identical `RngDriver` on top, so userspace sees the
//! same driver number and the same call sequence, keeping `KernelActivity`
//! fingerprints in sync across the lockstep barrier.

use core::cell::Cell;
use core::sync::atomic::{AtomicPtr, Ordering};

use kernel::hil::rng;
use kernel::utilities::cells::OptionalCell;
use kernel::ErrorCode;

use crate::chip::{RNG_REPLAY_BUF, RNG_REPLAY_MAX};
use crate::lockstep::RngHooks;

/// Maximum `u32` words that fit in one RNG replay.
/// VirtIORng's internal buffer is 64 bytes → 16 words.
const RNG_REPLAY_WORDS: usize = RNG_REPLAY_MAX / 4;

/// Pointer to Hart 1's [`LockstepRng`] instance.
///
/// Set during `start_secondary()` (via [`AtomicPtr::store`] with
/// [`Ordering::Release`]) so that [`replay_rng_done_for_hart1`] can reach it
/// from the main loop without requiring a `const`-constructible global.
/// Default is null; [`replay_rng_done_for_hart1`] checks and skips if null.
pub static HART1_RNG: AtomicPtr<LockstepRng<'static>> = AtomicPtr::new(core::ptr::null_mut());

/// Hart-aware RNG interposer implementing [`rng::Rng`].
///
/// Hart 0 wraps the real `VirtIORng`; hart 1 is a software replay stub.
/// The two harts are wired symmetrically through the same unmodified
/// `RngDriver` capsule, so both harts execute the same syscall sequence.
pub struct LockstepRng<'a> {
    hart_id: u32,
    /// Hart 0 only: the real VirtIO entropy source. Empty on Hart 1.
    real: OptionalCell<&'a dyn rng::Rng<'a>>,
    /// Hart 1 only: tracks whether a `get()` is outstanding, so `replay()`
    /// knows whether to deliver the callback or silently discard it
    /// (post-`cancel()` or spurious delivery).
    pending: Cell<bool>,
    /// Downstream client (`RngDriver`) on both harts.
    client: OptionalCell<&'a dyn rng::Client>,
    /// Lockstep hook called after each entropy batch is drained.
    hooks: Option<&'static dyn RngHooks>,
}

impl<'a> LockstepRng<'a> {
    /// Construct a new `LockstepRng`.
    ///
    /// Pass `real = Some(virtio_rng)` on hart 0 and `None` on hart 1.
    /// Pass `hooks = Some(...)` to enable cross-hart replay signalling.
    /// Reads `mhartid` once at construction, mirroring `Uart16550::new()`.
    pub fn new(real: Option<&'a dyn rng::Rng<'a>>, hooks: Option<&'static dyn RngHooks>) -> Self {
        LockstepRng {
            hart_id: crate::chip::current_hart(),
            real: match real {
                Some(r) => OptionalCell::new(r),
                None => OptionalCell::empty(),
            },
            pending: Cell::new(false),
            client: OptionalCell::empty(),
            hooks,
        }
    }

    /// Deliver a forwarded RNG batch to the downstream client on Hart 1.
    ///
    /// Called from [`replay_rng_done_for_hart1`] when `SyncEntry::RngReady`
    /// is popped from the lockstep channel. `len` is the byte count carried
    /// in the channel message; bytes are read from
    /// [`RNG_REPLAY_BUF`][crate::chip::RNG_REPLAY_BUF].
    ///
    /// Honors `cancel()` by checking the `pending` flag first — if `cancel()`
    /// was called after `get()`, the flag is false and the replay is discarded
    /// without calling the client.
    fn replay(&self, len: u8) {
        if !self.pending.get() {
            // cancel() cleared the flag; discard without calling the client.
            return;
        }

        let len = (len as usize).min(RNG_REPLAY_MAX);
        // SAFETY: Hart 0 writes RNG_REPLAY_BUF before pushing RngReady onto
        // LOCKSTEP_CHAN (see chip.rs RngReplayBuf SAFETY comment). Hart 1 reads
        // it only after popping that message; the channel's Acquire ordering on
        // read provides the happens-before relationship.
        let buf = unsafe { &*RNG_REPLAY_BUF.0.get() };

        let mut iter = buf[..len].chunks(4).filter_map(|s| {
            if s.len() == 4 {
                Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            } else {
                None
            }
        });

        let cont = self
            .client
            .map(|c| c.randomness_available(&mut iter, Ok(())))
            .unwrap_or(rng::Continue::Done);

        // Keep pending=true if the client wants more (a second RngReady is on
        // the way from hart 0, driven by VirtIORng calling get() again after
        // receiving Continue::More from LockstepRng::randomness_available).
        if let rng::Continue::Done = cont {
            self.pending.set(false);
        }
    }
}

impl<'a> rng::Rng<'a> for LockstepRng<'a> {
    fn get(&self) -> Result<(), ErrorCode> {
        if self.hart_id == 0 {
            // Drive the real hardware. The forwarded copy is emitted later,
            // from randomness_available() when the bytes actually arrive.
            if let Some(r) = self.real.get() {
                r.get()
            } else {
                Err(ErrorCode::FAIL)
            }
        } else {
            // Hart 1: record the pending request. The callback fires when a
            // RngReady arrives from hart 0 (Ok(()) promises it will).
            self.pending.set(true);
            Ok(())
        }
    }

    fn cancel(&self) -> Result<(), ErrorCode> {
        if self.hart_id == 0 {
            if let Some(r) = self.real.get() {
                r.cancel()
            } else {
                Err(ErrorCode::FAIL)
            }
        } else {
            // Hart 1: drop the pending request. replay() checks pending before
            // calling the client, so no randomness_available callback is issued.
            self.pending.set(false);
            Ok(())
        }
    }

    fn set_client(&'a self, client: &'a dyn rng::Client) {
        self.client.set(client);
        if self.hart_id == 0 {
            // Register ourselves as VirtIORng's callback recipient so that
            // randomness_available() below intercepts each entropy delivery.
            self.real.map(|r| r.set_client(self));
        }
    }
}

impl rng::Client for LockstepRng<'_> {
    /// Only called on Hart 0 — registered as VirtIORng's client in `set_client`.
    ///
    /// 1. Drains the incoming `u32` iterator into a stack-local buffer.
    /// 2. Copies bytes into `RNG_REPLAY_BUF` and pushes `SyncEntry::RngReady`
    ///    onto the lockstep channel so Hart 1 can replay the same bytes.
    /// 3. Rebuilds a fresh iterator over those same words and forwards it to the
    ///    downstream `RngDriver`.
    ///
    /// The `Continue` value from the downstream client is returned directly to
    /// `VirtIORng::buffer_chain_callback`, which calls `get()` again if `More`
    /// — triggering another hardware request and a second `RngReady` message.
    fn randomness_available(
        &self,
        randomness: &mut dyn Iterator<Item = u32>,
        error: Result<(), ErrorCode>,
    ) -> rng::Continue {
        // Drain into a stack buffer to avoid holding a borrow on RNG_REPLAY_BUF
        // across the LOCKSTEP_CHAN send.
        let mut tmp = [0u32; RNG_REPLAY_WORDS];
        let mut word_count = 0usize;
        for word in randomness.take(RNG_REPLAY_WORDS) {
            tmp[word_count] = word;
            word_count += 1;
        }
        let byte_len = word_count * 4;

        if byte_len > 0 {
            if let Some(h) = self.hooks {
                h.on_randomness_available(&tmp[..word_count]);
            }
        }

        // Rebuild an iterator from the captured words and forward downstream.
        let mut forward_iter = tmp[..word_count].iter().copied();
        self.client
            .map(|c| c.randomness_available(&mut forward_iter, error))
            .unwrap_or(rng::Continue::Done)
    }
}

/// Called by Hart 1's main loop when it pops [`SyncEntry::RngReady`] from
/// `LOCKSTEP_CHAN`. Mirrors `replay_rx_done_for_hart1` in `uart.rs`.
pub fn replay_rng_done_for_hart1(len: u8) {
    let ptr = HART1_RNG.load(Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr` was stored by `start_secondary()` via `static_init!`, so it
    // points to a valid, fully-initialized `LockstepRng<'static>` that lives for
    // the lifetime of the program.
    unsafe { (*ptr).replay(len) }
}
