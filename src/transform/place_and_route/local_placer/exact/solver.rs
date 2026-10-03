//! Thin CaDiCaL binding. The `cadical` crate builds and links the bundled
//! solver, but does not expose options such as `seed`, so the C API is declared
//! here directly.

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use super::cnf::{Cnf, Lit};

// Links the static CaDiCaL library built by the `cadical` crate.
extern crate cadical;

#[repr(C)]
struct CCaDiCaL {
    _private: [u8; 0],
}

extern "C" {
    fn ccadical_init() -> *mut CCaDiCaL;
    fn ccadical_release(solver: *mut CCaDiCaL);
    fn ccadical_add(solver: *mut CCaDiCaL, lit: c_int);
    fn ccadical_assume(solver: *mut CCaDiCaL, lit: c_int);
    fn ccadical_solve(solver: *mut CCaDiCaL) -> c_int;
    fn ccadical_val(solver: *mut CCaDiCaL, lit: c_int) -> c_int;
    fn ccadical_failed(solver: *mut CCaDiCaL, lit: c_int) -> c_int;
    fn ccadical_set_option(solver: *mut CCaDiCaL, name: *const c_char, val: c_int);
    fn ccadical_set_terminate(
        solver: *mut CCaDiCaL,
        state: *mut c_void,
        terminate: Option<extern "C" fn(*mut c_void) -> c_int>,
    );
}

/// Stops a running search when another worker succeeds or the deadline passes.
pub(super) struct StopSignal<'a> {
    pub(super) stop: &'a AtomicBool,
    pub(super) deadline: Option<Instant>,
    /// Also stop once this counter moves past the given value (another worker
    /// found a cheaper layout, so the current bound is stale).
    pub(super) restart: Option<(&'a AtomicUsize, usize)>,
}

impl StopSignal<'_> {
    /// Stopped for good (not merely asked to restart with a new bound).
    pub(super) fn finished(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }
}

extern "C" fn terminate_callback(state: *mut c_void) -> c_int {
    // SAFETY: `state` points to the `StopSignal` borrowed for the whole solve call.
    let signal = unsafe { &*(state as *const StopSignal) };
    let stale = signal
        .restart
        .is_some_and(|(counter, seen)| counter.load(Ordering::Relaxed) != seen);
    c_int::from(stale || signal.finished())
}

pub(super) struct SatSolver {
    raw: *mut CCaDiCaL,
}

// The solver is used from one thread at a time.
unsafe impl Send for SatSolver {}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum SolveResult {
    Sat,
    Unsat,
    Interrupted,
}

impl SatSolver {
    pub(super) fn new(seed: u32) -> Self {
        // SAFETY: plain constructor from the bundled C API.
        let raw = unsafe { ccadical_init() };
        let solver = Self { raw };
        solver.set_option("seed", seed as i32);
        solver
    }

    pub(super) fn set_option(&self, name: &str, value: i32) {
        let name = CString::new(name).expect("option names have no NUL");
        // SAFETY: valid solver pointer and NUL-terminated option name.
        unsafe { ccadical_set_option(self.raw, name.as_ptr(), value) };
    }

    pub(super) fn add_cnf(&mut self, cnf: &Cnf) {
        for &lit in cnf.literals() {
            // SAFETY: literals are nonzero except clause terminators.
            unsafe { ccadical_add(self.raw, lit) };
        }
    }

    pub(super) fn add_clause(&mut self, clause: &[Lit]) {
        for &lit in clause {
            // SAFETY: see `add_cnf`.
            unsafe { ccadical_add(self.raw, lit) };
        }
        // SAFETY: terminates the clause.
        unsafe { ccadical_add(self.raw, 0) };
    }

    pub(super) fn solve(&mut self, assumptions: &[Lit], signal: &StopSignal) -> SolveResult {
        for &lit in assumptions {
            // SAFETY: valid nonzero literal.
            unsafe { ccadical_assume(self.raw, lit) };
        }
        let state = signal as *const StopSignal as *mut c_void;
        // SAFETY: `signal` outlives this call; the callback is removed below.
        let result = unsafe {
            ccadical_set_terminate(self.raw, state, Some(terminate_callback));
            let result = ccadical_solve(self.raw);
            ccadical_set_terminate(self.raw, std::ptr::null_mut(), None);
            result
        };
        match result {
            10 => SolveResult::Sat,
            20 => SolveResult::Unsat,
            _ => SolveResult::Interrupted,
        }
    }

    pub(super) fn value(&self, lit: Lit) -> bool {
        // SAFETY: only called after a satisfiable solve.
        unsafe { ccadical_val(self.raw, lit) > 0 }
    }
}

impl SatSolver {
    /// After an unsatisfiable solve: whether assumption `lit` was in the core.
    pub(super) fn failed(&self, lit: Lit) -> bool {
        // SAFETY: only called after an unsatisfiable solve with assumptions.
        unsafe { ccadical_failed(self.raw, lit) != 0 }
    }
}

impl Drop for SatSolver {
    fn drop(&mut self) {
        // SAFETY: the pointer was created by `ccadical_init` and is released once.
        unsafe { ccadical_release(self.raw) };
    }
}
