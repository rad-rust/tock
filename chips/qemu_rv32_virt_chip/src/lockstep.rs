// Licensed under the Apache License, Version 2.0 or the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer-2 cross-hart lockstep verification for the `qemu_rv32_virt` board.
//!
//! Provides [`QemuUpcallVerifier`], a [`kernel::platform::UpcallVerifier`]
//! implementation that the board installs via
//! [`kernel::Kernel::register_upcall_verifier`] before entering the main loop.
//!
//! The verifier is driven by a static [`DriverUpcallRules`] registry supplied
//! by the board (which knows the concrete capsule driver numbers). For every
//! driver upcall the kernel is about to deliver to a process, `on_upcall` looks
//! up the matching rule and either:
//!
//! - **Compare mode**: exchanges a [`crate::chip::SyncEntry::UpcallDesc`] with
//!   the other hart and panics on any argument divergence.
//! - **Forward mode**: hart 0 sends its value for a live-divergent field (e.g.
//!   `r0 = now` in an alarm upcall) and hart 1 uses that value instead of its
//!   own, so both harts deliver the same arguments to the app.
//!
//! # Staging note
//!
//! The cross-hart channel exchange (`TODO(lockstep-stage0)`) requires the
//! concurrent `lockstep_barrier` primitive introduced in Stage 0 of the lockstep
//! Layer-2 plan. Until that primitive is in place, `on_upcall` returns
//! `UpcallAction::Proceed` for every registered upcall, preserving existing
//! behaviour while the infrastructure is wired up.

use crate::chip::{read_mtime_low, SyncEntry, LOCKSTEP_CHAN};
use kernel::hil;
use kernel::platform::{UpcallAction, UpcallVerifier};
use kernel::upcall::UpcallId;
use kernel::utilities::cells::OptionalCell;

// ---------------------------------------------------------------------------
// Timeout constant
// ---------------------------------------------------------------------------

/// Rendezvous Sync barrier timeout.
///
/// After hart 0 pushes its Sync entry it spins calling `read_mtime_low()` on
/// every iteration while waiting for hart 1's reply.  Each MMIO read advances
/// the QEMU TCG simulation clock by device-emulation overhead (typically
/// 10–100 cycles), so tight-spinning burns mtime fast — the same effect that
/// forced [`DRAIN_TIMEOUT_MTIME_TICKS`] to 1 B.  1 M ticks (nominally 100 ms)
/// proved too small: the timeout fired after ~8000 yield-no-wait iterations
/// even though hart 1 was still making progress.
///
/// Set to 1 000 000 000 (≈ 100 s at 10 MHz) to match the drain timeout and
/// give hart 1 ample time to complete its Phase-2 KW rounds before replying.
/// A future heartbeat mechanism will let us shrink both constants safely.
pub const SYNC_TIMEOUT_MTIME_TICKS: u32 = 1_000_000_000;

/// Drain-wait timeout for hart 1's phase-1 drain loop.
///
/// After receiving an L1 event (UartTxDone / RngReady / UartRxReady) hart 1
/// starts this timer. A matching Sync from hart 0 MUST follow before this
/// expires; if it does not, hart 0 has diverged.
///
/// This needs to be much larger than [`SYNC_TIMEOUT_MTIME_TICKS`] for two
/// reasons:
///
/// 1. **QEMU TCG MMIO overhead**: `read_mtime_low()` is an MMIO read; in
///    TCG mode each MMIO access advances the simulation clock by the device
///    emulation overhead (typically 10–100 cycles). The drain loop calls it
///    every iteration, so tight-spinning burns mtime fast.
/// 2. **KW asymmetry window**: between two consecutive Sync events hart 0 may
///    execute many KernelWork rounds (pconsole timer callbacks, VirtIO, …)
///    each of which advances mtime without sending a Sync.
///
/// 1 000 000 000 ticks ≈ 100 s at 10 MHz — generous enough to survive any
/// expected KW burst while still catching a truly diverged hart 0 eventually.
/// A future heartbeat mechanism will shrink this to milliseconds.
pub const DRAIN_TIMEOUT_MTIME_TICKS: u32 = 1_000_000_000;

// ---------------------------------------------------------------------------
// Layer-1 event dispatch
// ---------------------------------------------------------------------------

/// Replay a single Layer-1 channel event on hart 1.
///
/// Layer-1 events (`UartRxReady`, `UartTxDone`, `RngReady`) carry replayed
/// hardware inputs from hart 0. Hart 0 owns the real peripherals and never
/// receives these, so this function is only meaningful on hart 1. It is the
/// `dispatch` callback passed to [`lockstep_barrier`].
pub fn dispatch_layer1_event(entry: SyncEntry) {
    match entry {
        SyncEntry::UartRxReady { len } => crate::uart::replay_rx_done_for_hart1(len),
        SyncEntry::UartTxDone => crate::uart::replay_tx_done_for_hart1(),
        SyncEntry::RngReady { len } => crate::rng::replay_rng_done_for_hart1(len),
        SyncEntry::Sync { .. } | SyncEntry::UpcallDesc { .. } => {
            unreachable!("lockstep_barrier descriptors must not be dispatched as Layer-1 events")
        }
    }
}

// ---------------------------------------------------------------------------
// Rendezvous primitive
// ---------------------------------------------------------------------------

/// Exchange a synchronization descriptor with the other hart and return its
/// response.
///
/// The loop is symmetric: both sides run identical code; only `hart_id`
/// selects the channel direction (`a_*` for hart 0, `b_*` for hart 1).
///
/// Each iteration:
/// 1. Tries a non-blocking push of `mine` (retries until the queue has space).
/// 2. Tries a non-blocking pop; if an entry arrives:
///    - [`SyncEntry::Sync`] or [`SyncEntry::UpcallDesc`] → returned to caller.
///    - Any other entry (Layer-1 event) → forwarded to `dispatch`.
/// 3. Panics if `mine` was successfully pushed but no counterpart arrives
///    within [`SYNC_TIMEOUT_MTIME_TICKS`] — indicates divergence or SEU.
///
/// The caller is responsible for comparing `mine` against the returned entry
/// and panicking on mismatch; this keeps the primitive free of comparison
/// semantics so it can serve both Sync barriers and upcall descriptors.
pub fn lockstep_barrier(
    hart_id: u32,
    mine: SyncEntry,
    dispatch: impl Fn(SyncEntry),
) -> SyncEntry {
    let mut pushed = false;
    let start = read_mtime_low();
    loop {
        if !pushed {
            pushed = if hart_id == 0 {
                LOCKSTEP_CHAN.a_send(mine)
            } else {
                LOCKSTEP_CHAN.b_send(mine)
            };
        }
        let recv = if hart_id == 0 {
            LOCKSTEP_CHAN.a_recv()
        } else {
            LOCKSTEP_CHAN.b_recv()
        };
        if let Some(e) = recv {
            match e {
                SyncEntry::Sync { .. } | SyncEntry::UpcallDesc { .. } => return e,
                layer1 => dispatch(layer1),
            }
        }
        if pushed && read_mtime_low().wrapping_sub(start) >= SYNC_TIMEOUT_MTIME_TICKS {
            panic!("lockstep: lockstep barrier timeout (possible SEU or divergence)");
        }
        core::hint::spin_loop();
    }
}

// ---------------------------------------------------------------------------
// Registry types (defined here; initialized by the board using capsule consts)
// ---------------------------------------------------------------------------

/// How a particular upcall is verified across harts.
pub enum UpcallMode {
    /// Both harts generate this upcall independently (e.g. console WRITE_DONE
    /// after a Layer-1 TX replay). Compare the masked fields and panic on
    /// mismatch.
    Compare,
    /// Hart 0 holds the authoritative value for one or more masked fields (e.g.
    /// the live `now` timestamp in an alarm upcall). Hart 1 uses hart 0's value
    /// for those fields and its own for the rest.
    Forward,
}

/// Verification rule for a single upcall subscription slot.
pub struct UpcallRule {
    /// Subscribe number (slot within the driver, 0-indexed).
    pub subscribe_num: usize,
    /// Compare or forward semantics (see [`UpcallMode`]).
    pub mode: UpcallMode,
    /// Which of `(r0, r1, r2)` to compare/forward (`true`) vs ignore (`false`).
    /// Unmasked fields are excluded from comparison and left at their
    /// per-hart values.
    pub mask: (bool, bool, bool),
}

/// Verification rules for all intercepted upcalls of a single driver.
pub struct DriverUpcallRules {
    /// Driver number (`capsules_core::xxx::DRIVER_NUM`), set by the board.
    pub driver_num: usize,
    /// Rules for each subscribe slot that needs cross-hart verification.
    pub rules: &'static [UpcallRule],
}

// ---------------------------------------------------------------------------
// QemuUpcallVerifier
// ---------------------------------------------------------------------------

/// Lockstep upcall verifier for the `qemu_rv32_virt` dual-hart configuration.
///
/// Each hart (0 and 1) receives its own instance via `static_init!` in
/// `start()` / `start_secondary()`. The verifier reads `mhartid` at
/// construction so the hart-0 and hart-1 instances branch correctly inside
/// `on_upcall`.
///
/// The `registry` is a `'static` slice of [`DriverUpcallRules`] provided by
/// the board; the chip crate does not depend on capsule crates directly.
pub struct QemuUpcallVerifier {
    hart_id: u32,
    registry: &'static [DriverUpcallRules],
}

impl QemuUpcallVerifier {
    /// Create a new verifier for this hart. Reads `mhartid` from the CSR.
    pub fn new(registry: &'static [DriverUpcallRules]) -> Self {
        Self { hart_id: crate::chip::current_hart(), registry }
    }
}

impl UpcallVerifier for QemuUpcallVerifier {
    fn on_upcall(&self, id: UpcallId, r0: usize, r1: usize, r2: usize) -> UpcallAction {
        // Look up the rule for this (driver_num, subscribe_num).
        let rule = self
            .registry
            .iter()
            .find(|d| d.driver_num == id.driver_num)
            .and_then(|d| d.rules.iter().find(|u| u.subscribe_num == id.subscribe_num));

        let rule = match rule {
            Some(r) => r,
            // No rule for this upcall; pass through unchanged.
            None => return UpcallAction::Proceed,
        };

        // TODO(lockstep-stage3): perform the cross-hart channel exchange here.
        // `lockstep_barrier` (lockstep.rs) is now available for this.
        //
        // Hart 0 path (Compare):
        //   1. Build UpcallDesc with masked fields.
        //   2. lockstep_barrier(UpcallDesc) — drain-while-waiting, exchange with hart 1.
        //   3. Compare received descriptor; panic on mismatch.
        //   4. Return UpcallAction::Proceed.
        //
        // Hart 1 path (Compare):
        //   (same as hart 0 — both generate the upcall independently)
        //
        // Hart 0 path (Forward):
        //   Send the authoritative masked field(s) to hart 1 via UpcallForward,
        //   still using the Compare lockstep_barrier for non-masked fields.
        //   Return UpcallAction::Proceed (hart 0 uses its own value).
        //
        // Hart 1 path (Forward):
        //   Receive UpcallForward from hart 0; return UpcallAction::Overwrite
        //   with hart 0's value substituted for the masked field(s).
        //
        // Requires the concurrent lockstep_barrier loop from Stage 0 so that hart 1
        // is draining the channel while hart 0 blocks inside on_upcall.
        let _ = (rule, r0, r1, r2, self.hart_id);
        UpcallAction::Proceed
    }
}

// ---------------------------------------------------------------------------
// UartHooks — per-HIL sync logic for UART
// ---------------------------------------------------------------------------

/// Callbacks invoked by [`LockstepUart`] at each UART boundary.
///
/// Implement this trait to express the lockstep sync logic for a given
/// UART; [`LockstepUart`] handles all HIL trait plumbing. Hart 0 and hart 1
/// share the same trait impl type but branch internally on `hart_id`.
pub trait UartHooks {
    /// Called just before `transmit_buffer` is forwarded to the inner driver.
    /// Use for verify+gate (fingerprint the payload, lockstep_barrier, compare).
    fn on_transmit(&self, buf: &[u8]);
    /// Called in `transmitted_buffer` (upward callback) before the capsule
    /// client is notified. Use to forward the TX-done signal to the other hart.
    fn on_transmitted(&self, buf: &[u8]);
    /// Called in `received_buffer` (upward callback) before the capsule
    /// client is notified. Use to copy data to the replay buffer and signal
    /// the other hart.
    fn on_received(&self, buf: &[u8]);
}

// ---------------------------------------------------------------------------
// LockstepUart — generic HIL wrapper
// ---------------------------------------------------------------------------

/// Wraps any UART inner driver `U` with lockstep hooks `H`.
///
/// Hart 0 wires `LockstepUart<Uart16550, QemuUartHooks>` between `uart0` and
/// the `MuxUart`; hart 1 wires `LockstepUart<VirtualUartBuffer, QemuUartHooks>`
/// between `HART1_UART_BUF` and the `Console`. The board registers:
///
/// - `inner` as calling the wrapper's [`TransmitClient`] / [`ReceiveClient`]
///   (upward path: inner fires → wrapper hook → capsule).
/// - wrapper as calling the capsule's [`TransmitClient`] / [`ReceiveClient`]
///   (done via the component or explicit `set_*_client` calls).
pub struct LockstepUart<'a, U, H> {
    inner: &'a U,
    hooks: &'a H,
    tx_client: OptionalCell<&'a dyn hil::uart::TransmitClient>,
    rx_client: OptionalCell<&'a dyn hil::uart::ReceiveClient>,
}

impl<'a, U, H> LockstepUart<'a, U, H> {
    pub fn new(inner: &'a U, hooks: &'a H) -> Self {
        Self {
            inner,
            hooks,
            tx_client: OptionalCell::empty(),
            rx_client: OptionalCell::empty(),
        }
    }
}

impl<'a, U: hil::uart::Configure, H: UartHooks> hil::uart::Configure for LockstepUart<'a, U, H> {
    fn configure(&self, params: hil::uart::Parameters) -> Result<(), kernel::ErrorCode> {
        self.inner.configure(params)
    }
}

impl<'a, U: hil::uart::Transmit<'a>, H: UartHooks> hil::uart::Transmit<'a>
    for LockstepUart<'a, U, H>
{
    fn set_transmit_client(&self, client: &'a dyn hil::uart::TransmitClient) {
        self.tx_client.set(client);
    }

    fn transmit_buffer(
        &self,
        tx_buf: &'static mut [u8],
        tx_len: usize,
    ) -> Result<(), (kernel::ErrorCode, &'static mut [u8])> {
        self.hooks.on_transmit(&tx_buf[..tx_len]);
        self.inner.transmit_buffer(tx_buf, tx_len)
    }

    fn transmit_word(&self, word: u32) -> Result<(), kernel::ErrorCode> {
        self.inner.transmit_word(word)
    }

    fn transmit_abort(&self) -> Result<(), kernel::ErrorCode> {
        self.inner.transmit_abort()
    }
}

impl<'a, U: hil::uart::Receive<'a>, H: UartHooks> hil::uart::Receive<'a>
    for LockstepUart<'a, U, H>
{
    fn set_receive_client(&self, client: &'a dyn hil::uart::ReceiveClient) {
        self.rx_client.set(client);
    }

    fn receive_buffer(
        &self,
        rx_buf: &'static mut [u8],
        rx_len: usize,
    ) -> Result<(), (kernel::ErrorCode, &'static mut [u8])> {
        self.inner.receive_buffer(rx_buf, rx_len)
    }

    fn receive_word(&self) -> Result<(), kernel::ErrorCode> {
        self.inner.receive_word()
    }

    fn receive_abort(&self) -> Result<(), kernel::ErrorCode> {
        self.inner.receive_abort()
    }
}

impl<'a, U, H: UartHooks> hil::uart::TransmitClient for LockstepUart<'a, U, H> {
    fn transmitted_buffer(
        &self,
        tx_buf: &'static mut [u8],
        tx_len: usize,
        rcode: Result<(), kernel::ErrorCode>,
    ) {
        self.hooks.on_transmitted(&tx_buf[..tx_len]);
        self.tx_client
            .map(|c| c.transmitted_buffer(tx_buf, tx_len, rcode));
    }

    fn transmitted_word(&self, rcode: Result<(), kernel::ErrorCode>) {
        self.tx_client.map(|c| c.transmitted_word(rcode));
    }
}

impl<'a, U, H: UartHooks> hil::uart::ReceiveClient for LockstepUart<'a, U, H> {
    fn received_buffer(
        &self,
        rx_buf: &'static mut [u8],
        rx_len: usize,
        rcode: Result<(), kernel::ErrorCode>,
        error: hil::uart::Error,
    ) {
        self.hooks.on_received(&rx_buf[..rx_len]);
        self.rx_client
            .map(|c| c.received_buffer(rx_buf, rx_len, rcode, error));
    }

    fn received_word(
        &self,
        word: u32,
        rcode: Result<(), kernel::ErrorCode>,
        error: hil::uart::Error,
    ) {
        self.rx_client.map(|c| c.received_word(word, rcode, error));
    }
}

// ---------------------------------------------------------------------------
// QemuUartHooks — QEMU-specific UART sync implementation
// ---------------------------------------------------------------------------

/// QEMU-specific UART lockstep hooks.
///
/// On hart 0: `on_transmitted` sends `UartTxDone` and wakes hart 1;
/// `on_received` copies to `UART_RX_REPLAY_BUF`, sends `UartRxReady`, and
/// wakes hart 1. `on_transmit` is a TODO for Stage 1/2 verify+gate.
/// On hart 1: all hooks are no-ops (the replay mechanism drives callbacks).
pub struct QemuUartHooks {
    hart_id: u32,
}

impl QemuUartHooks {
    pub fn new() -> Self {
        Self { hart_id: crate::chip::current_hart() }
    }
}

impl UartHooks for QemuUartHooks {
    fn on_transmit(&self, _buf: &[u8]) {
        // TODO(lockstep-stage1): fingerprint _buf, lockstep_barrier with partner
        // hart, panic on mismatch. Gate hart 0 on hart 1's confirmation before
        // forwarding to the inner driver.
    }

    fn on_transmitted(&self, _buf: &[u8]) {
        if self.hart_id == 0 {
            use crate::chip::{CLINT_MSIP1, LOCKSTEP_CHAN, SyncEntry};
            while !LOCKSTEP_CHAN.a_send(SyncEntry::UartTxDone) {
                core::hint::spin_loop();
            }
            unsafe { core::ptr::write_volatile(CLINT_MSIP1, 1) };
        }
    }

    fn on_received(&self, buf: &[u8]) {
        if self.hart_id == 0 {
            use crate::chip::{
                CLINT_MSIP1, LOCKSTEP_CHAN, UART_RX_REPLAY_BUF, UART_RX_REPLAY_MAX,
                SyncEntry,
            };
            let copy_len = buf.len().min(UART_RX_REPLAY_MAX);
            unsafe {
                (&mut *UART_RX_REPLAY_BUF.0.get())[..copy_len]
                    .copy_from_slice(&buf[..copy_len]);
            }
            while !LOCKSTEP_CHAN.a_send(SyncEntry::UartRxReady { len: copy_len as u8 }) {
                core::hint::spin_loop();
            }
            unsafe { core::ptr::write_volatile(CLINT_MSIP1, 1) };
        }
    }
}

// ---------------------------------------------------------------------------
// RngHooks — per-HIL sync logic for RNG
// ---------------------------------------------------------------------------

/// Callback invoked by [`LockstepRng`] after draining an entropy batch.
///
/// `words` contains the `u32` words drained from the VirtIO iterator. On
/// hart 0 the implementor copies them to the replay buffer and notifies hart 1.
/// On hart 1 the implementor is typically a no-op.
pub trait RngHooks {
    fn on_randomness_available(&self, words: &[u32]);
}

// ---------------------------------------------------------------------------
// QemuRngHooks — QEMU-specific RNG sync implementation
// ---------------------------------------------------------------------------

/// QEMU-specific RNG lockstep hooks.
///
/// On hart 0: copies the word batch to `RNG_REPLAY_BUF` in little-endian byte
/// order, pushes `RngReady` onto the channel, and asserts MSIP[1] to wake hart
/// 1. On hart 1: no-op.
pub struct QemuRngHooks {
    hart_id: u32,
}

impl QemuRngHooks {
    pub fn new() -> Self {
        Self { hart_id: crate::chip::current_hart() }
    }
}

impl RngHooks for QemuRngHooks {
    fn on_randomness_available(&self, words: &[u32]) {
        if self.hart_id == 0 && !words.is_empty() {
            use crate::chip::{CLINT_MSIP1, LOCKSTEP_CHAN, RNG_REPLAY_BUF, SyncEntry};
            let byte_len = (words.len() * 4).min(crate::chip::RNG_REPLAY_MAX);
            let word_count = byte_len / 4;
            unsafe {
                let buf = &mut *RNG_REPLAY_BUF.0.get();
                for (i, &w) in words[..word_count].iter().enumerate() {
                    buf[i * 4..(i + 1) * 4].copy_from_slice(&u32::to_le_bytes(w));
                }
            }
            while !LOCKSTEP_CHAN.a_send(SyncEntry::RngReady { len: byte_len as u8 }) {
                core::hint::spin_loop();
            }
            unsafe { core::ptr::write_volatile(CLINT_MSIP1, 1) };
        }
    }
}
