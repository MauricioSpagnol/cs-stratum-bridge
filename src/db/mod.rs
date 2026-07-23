//! Persistence layer for the OPoI pool-side bridge.
//!
//! Not yet wired into the binary — whoever integrates this module adds
//! `mod db;` to `src/main.rs`.

pub mod models;
pub mod repo;

pub use models::{RevealCandidate, Submission, SubmissionStatus};
