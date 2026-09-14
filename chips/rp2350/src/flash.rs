// Licensed under the Apache License, Version 2.0 or the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RP2350 boot-ROM flash function lookup.
//!
//! Stage 2a of the flash-vs-RAM write benchmark: this module only resolves
//! the boot-ROM function pointers a real flash write will eventually need.
//! Nothing here calls any of them, touches flash, or pauses XIP -- it is
//! safe to run at any time, from ordinary flash-resident code.
//!
//! Algorithm hand-ported from the public pico-sdk's `rom_func_lookup_inline`
//! (`pico/bootrom.h`, ARM/non-RP2040 branch) and the numeric offsets from
//! `bootrom_constants.h` -- this fork has no vendored pico-sdk dependency
//! (see e.g. `chips/rp2040/src/i2c.rs`'s similar "based on the logic from
//! the official pico-sdk" attribution for the existing precedent of this
//! pattern in this tree).
//!
//! Stage 2c adds the actual RAM-resident erase/program calls (hand-ported
//! from pico-sdk's `hardware_flash/flash.c`: `flash_init_boot2_copyout`,
//! `flash_enable_xip_via_boot2`, and the `flash_range_erase`/
//! `flash_range_program` wrappers). These pause XIP chip-wide for their
//! duration -- callers (`boards/raspberry_pi_pico_2/src/main.rs`) are
//! responsible for parking core 1 first (see `FLASH_PARK_REQUEST`/
//! `FLASH_PARK_ACK` there) and must never call these with `data` pointing
//! at flash-resident memory (the source must be RAM-resident too, since
//! flash is unreadable for the whole call).

#![allow(dead_code)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

/// RP2350 has (at least) two silicon revisions with *different* fixed
/// boot-ROM addresses for finding `rom_table_lookup` itself. Revision A1
/// stores a 32-bit pointer at `0x18`; revision A2+ (the current production
/// stepping) stores a 16-bit pointer at `0x16` instead. Cross-checked
/// against `rp-hal`'s `rom_data.rs` (a maintained, real-world Rust HAL for
/// this chip) after hardcoding only the A1 address caused a HardFault on
/// real A2 hardware -- the pico-sdk C header this module was originally
/// ported from only documents the single, version-independent
/// `BOOTROM_TABLE_LOOKUP_OFFSET = 0x18` C macro, which turned out to be
/// A1-only in practice.
const ROM_VERSION_NUMBER_ADDR: *const u8 = 0x0000_0013 as *const u8;
const ROM_TABLE_LOOKUP_A1: *const u32 = 0x0000_0018 as *const u32;
const ROM_TABLE_LOOKUP_A2: *const u16 = 0x0000_0016 as *const u16;

/// Read the boot-ROM version byte at the fixed address `0x13`.
///
/// # Safety
/// Reads a fixed, always-mapped, read-only boot-ROM address. Safe at any
/// time.
unsafe fn rom_version_number() -> u8 {
    // SAFETY: ROM_VERSION_NUMBER_ADDR is a fixed, always-mapped ROM address.
    unsafe { core::ptr::read(ROM_VERSION_NUMBER_ADDR) }
}

/// Secure-ARM function-table mask (pico-sdk `RT_FLAG_FUNC_ARM_SEC`).
///
/// This kernel's `METADATA_BLOCK` declares it runs in Secure mode (see
/// `boards/raspberry_pi_pico_2/src/flash_bootloader.rs`), so we always pass
/// the secure flag -- unlike pico-sdk, which branches on
/// `pico_processor_state_is_nonsecure()` to support both.
const RT_FLAG_FUNC_ARM_SEC: u32 = 0x0004;

const fn rom_table_code(c1: u8, c2: u8) -> u32 {
    (c1 as u32) | ((c2 as u32) << 8)
}

type RomTableLookupFn = unsafe extern "C" fn(u32, u32) -> *const ();

/// Resolve the `rom_table_lookup` entry point itself out of the fixed
/// boot-ROM pointer table.
///
/// # Safety
/// Reads a fixed, always-mapped, read-only boot-ROM address. Does not touch
/// flash or pause XIP -- safe to call from ordinary flash-resident code at
/// any time.
unsafe fn rom_table_lookup_fn() -> RomTableLookupFn {
    // SAFETY: both ROM_TABLE_LOOKUP_A1/A2 and ROM_VERSION_NUMBER_ADDR are
    // fixed, always-mapped ROM addresses; the boot ROM stores a valid,
    // correctly Thumb-tagged function pointer at whichever one applies to
    // this chip's revision, by construction.
    let raw = unsafe {
        if rom_version_number() == 1 {
            core::ptr::read(ROM_TABLE_LOOKUP_A1) as usize
        } else {
            core::ptr::read(ROM_TABLE_LOOKUP_A2) as usize
        }
    };
    // SAFETY: see above.
    unsafe { core::mem::transmute::<usize, RomTableLookupFn>(raw) }
}

/// Read the boot-ROM version byte at the fixed address `0x13` (public
/// wrapper for diagnostics -- e.g. confirming which of the A1/A2 lookup
/// addresses a given board's silicon actually takes).
///
/// # Safety
/// Reads a fixed, always-mapped, read-only boot-ROM address. Safe at any
/// time.
pub unsafe fn rom_version() -> u8 {
    // SAFETY: see `rom_version_number`.
    unsafe { rom_version_number() }
}

/// Look up one boot-ROM function by its 2-character code. Returns a null
/// pointer if the code isn't found.
///
/// # Safety
/// Same as `rom_table_lookup_fn` -- the lookup itself only reads ROM tables
/// and does not touch flash.
unsafe fn rom_func_lookup(c1: u8, c2: u8) -> *const () {
    // SAFETY: see `rom_table_lookup_fn`.
    let lookup = unsafe { rom_table_lookup_fn() };
    // SAFETY: `rom_table_lookup` itself only walks ROM-resident tables; it
    // does not touch flash or require RAM residency to call.
    unsafe { lookup(rom_table_code(c1, c2), RT_FLAG_FUNC_ARM_SEC) }
}

/// Resolved pointers for the six boot-ROM functions a real flash write will
/// need (Stage 2b/2c). Stage 2a only resolves and reports these.
pub struct FlashRomFns {
    pub connect_internal_flash: *const (),
    pub flash_exit_xip: *const (),
    pub flash_range_erase: *const (),
    pub flash_range_program: *const (),
    pub flash_flush_cache: *const (),
    pub flash_enter_cmd_xip: *const (),
}

impl FlashRomFns {
    /// Resolve all six boot-ROM function pointers.
    ///
    /// # Safety
    /// Only performs boot-ROM table lookups (read-only, ROM-resident); does
    /// not call any resolved function and does not touch flash.
    pub unsafe fn lookup() -> Self {
        // SAFETY: `rom_func_lookup` only reads ROM-resident tables.
        unsafe {
            Self {
                connect_internal_flash: rom_func_lookup(b'I', b'F'),
                flash_exit_xip: rom_func_lookup(b'E', b'X'),
                flash_range_erase: rom_func_lookup(b'R', b'E'),
                flash_range_program: rom_func_lookup(b'R', b'P'),
                flash_flush_cache: rom_func_lookup(b'F', b'C'),
                flash_enter_cmd_xip: rom_func_lookup(b'C', b'X'),
            }
        }
    }

    /// True if every lookup found a non-null pointer and no two codes
    /// resolved to the same address -- a cheap sanity check to run (and
    /// report over the console) before Stage 2b/2c ever trust these.
    pub fn all_resolved_and_distinct(&self) -> bool {
        let ptrs = [
            self.connect_internal_flash,
            self.flash_exit_xip,
            self.flash_range_erase,
            self.flash_range_program,
            self.flash_flush_cache,
            self.flash_enter_cmd_xip,
        ];
        if ptrs.iter().any(|p| p.is_null()) {
            return false;
        }
        for i in 0..ptrs.len() {
            for j in (i + 1)..ptrs.len() {
                if ptrs[i] == ptrs[j] {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Stage 2c: real erase/program, and re-entering XIP afterward.
// ---------------------------------------------------------------------------

/// RP2350's dedicated 1KB "boot RAM", where the boot ROM places a copy of
/// the boot2 blob it used to bring the chip up (pico-sdk `BOOTRAM_BASE`).
/// This is *not* `flash_bootloader::FLASH_BOOTLOADER` (this crate has no
/// dependency on the board crate that defines that) -- reading from here
/// instead means the copy we re-run is guaranteed to match whatever the
/// actual hardware in front of us booted with, and needs no access to flash
/// (which is what makes it usable to restore XIP after flash goes offline).
const BOOTRAM_BASE: usize = 0x400e_0000;
const BOOT2_SIZE_WORDS: usize = 256 / 4;

struct Boot2Copy(UnsafeCell<[u32; BOOT2_SIZE_WORDS]>);
// SAFETY: only ever written by `init_boot2_ram_copy` (idempotent, guarded by
// `BOOT2_RAM_COPY_VALID`) and only ever read by `enable_xip_via_boot2_ram`;
// both only ever run on core 0, serialized by construction (core 0 never
// starts a second flash op while one is in flight).
unsafe impl Sync for Boot2Copy {}

static BOOT2_RAM_COPY: Boot2Copy = Boot2Copy(UnsafeCell::new([0u32; BOOT2_SIZE_WORDS]));
static BOOT2_RAM_COPY_VALID: AtomicBool = AtomicBool::new(false);

/// Copy boot2 from `BOOTRAM_BASE` into RAM, if not already done.
///
/// Must be called from ordinary flash-resident code, before ever pausing
/// XIP (mirrors pico-sdk's `flash_init_boot2_copyout`) -- by the time
/// `enable_xip_via_boot2_ram` needs this copy, flash (and the copying code
/// itself, if it weren't already done) would be unreadable.
pub fn init_boot2_ram_copy() {
    if BOOT2_RAM_COPY_VALID.load(Ordering::Acquire) {
        return;
    }
    // SAFETY: BOOTRAM_BASE is a fixed, always-mapped 1KB SRAM region
    // (pico-sdk `BOOTRAM_BASE`/`BOOTRAM_SIZE`); reading the first 256 bytes
    // as `u32`s is in-bounds. `BOOT2_RAM_COPY` is written here and nowhere
    // else concurrently (see `Boot2Copy`'s `Sync` impl comment).
    unsafe {
        let src = BOOTRAM_BASE as *const u32;
        let dst = BOOT2_RAM_COPY.0.get();
        for i in 0..BOOT2_SIZE_WORDS {
            (*dst)[i] = core::ptr::read_volatile(src.add(i));
        }
    }
    BOOT2_RAM_COPY_VALID.store(true, Ordering::Release);
}

/// Re-enter XIP by calling the RAM copy of boot2 -- mirrors pico-sdk's
/// `flash_enable_xip_via_boot2` exactly (RAM address, Thumb bit set via
/// `| 1`, called with no arguments and no return value).
///
/// # Safety
/// `init_boot2_ram_copy` must have already run (from flash-resident code,
/// before flash went offline). RAM-resident: flash is inaccessible until
/// this returns.
#[inline(never)]
#[unsafe(link_section = ".ramfunc")]
unsafe fn enable_xip_via_boot2_ram() {
    let entry = (BOOT2_RAM_COPY.0.get() as usize) | 1;
    // SAFETY: `entry` points at a valid copy of boot2 (a self-contained,
    // no-argument, no-return Thumb routine) with the Thumb bit set, per
    // `init_boot2_ram_copy`'s precondition.
    let f: unsafe extern "C" fn() = unsafe { core::mem::transmute(entry) };
    unsafe { f() };
}

type VoidRomFn = unsafe extern "C" fn();
type EraseRomFn = unsafe extern "C" fn(u32, usize, u32, u8);
type ProgramRomFn = unsafe extern "C" fn(u32, *const u8, usize);

/// RP2350/this board's flash erase granularity in bytes.
pub const FLASH_SECTOR_SIZE: u32 = 4096;
/// Standard JEDEC SPI-NOR "Sector Erase" opcode (universal across SPI-NOR
/// flash, including this board's W25Q32RV).
const FLASH_SECTOR_ERASE_CMD: u8 = 0x20;
/// Flash program granularity in bytes -- `data.len()` passed to
/// `program_from_ram` must be a multiple of this.
pub const FLASH_PAGE_SIZE: usize = 256;

/// Erase one `FLASH_SECTOR_SIZE`-byte sector at `offset` (relative to flash
/// base `0x10000000` -- e.g. pass `0x200000` for absolute address
/// `0x10200000`).
///
/// # Safety
/// - `offset` must be `FLASH_SECTOR_SIZE`-aligned, within the flash chip's
///   actual size, and must never overlap the kernel (`rom`) or app (`prog`)
///   regions -- see `boards/raspberry_pi_pico_2/layout.ld`.
/// - `rom_fns` must be the result of a successful `FlashRomFns::lookup()`
///   (all six pointers non-null and distinct), and `init_boot2_ram_copy`
///   must have already run.
/// - Caller must have already confirmed core 1 is parked (observed
///   `FLASH_PARK_ACK`) and must not release it until this returns -- core 1
///   cannot fetch instructions from flash for this call's whole duration.
#[inline(never)]
#[unsafe(link_section = ".ramfunc")]
pub unsafe fn erase_sector_from_ram(rom_fns: &FlashRomFns, offset: u32) {
    // SAFETY: pointers are valid per `rom_fns`' precondition; called with
    // the argument counts/types pico-sdk's own C prototypes use for these
    // ROM functions.
    unsafe {
        let connect: VoidRomFn = core::mem::transmute(rom_fns.connect_internal_flash);
        let exit_xip: VoidRomFn = core::mem::transmute(rom_fns.flash_exit_xip);
        let erase: EraseRomFn = core::mem::transmute(rom_fns.flash_range_erase);
        let flush: VoidRomFn = core::mem::transmute(rom_fns.flash_flush_cache);

        connect();
        exit_xip();
        erase(
            offset,
            FLASH_SECTOR_SIZE as usize,
            FLASH_SECTOR_SIZE,
            FLASH_SECTOR_ERASE_CMD,
        );
        flush();
        enable_xip_via_boot2_ram();
    }
}

/// Program `data` at `offset` (relative to flash base). `offset` must lie
/// within an already-erased region, and `data.len()` must be a multiple of
/// `FLASH_PAGE_SIZE`.
///
/// # Safety
/// Same preconditions as `erase_sector_from_ram`, plus: `data` must point at
/// RAM-resident memory, never flash-resident memory (flash is unreadable
/// for this call's whole duration, including the source buffer).
#[inline(never)]
#[unsafe(link_section = ".ramfunc")]
pub unsafe fn program_from_ram(rom_fns: &FlashRomFns, offset: u32, data: &[u8]) {
    // SAFETY: see `erase_sector_from_ram`.
    unsafe {
        let connect: VoidRomFn = core::mem::transmute(rom_fns.connect_internal_flash);
        let exit_xip: VoidRomFn = core::mem::transmute(rom_fns.flash_exit_xip);
        let program: ProgramRomFn = core::mem::transmute(rom_fns.flash_range_program);
        let flush: VoidRomFn = core::mem::transmute(rom_fns.flash_flush_cache);

        connect();
        exit_xip();
        program(offset, data.as_ptr(), data.len());
        flush();
        enable_xip_via_boot2_ram();
    }
}
