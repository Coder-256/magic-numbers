#![warn(clippy::all)]
#![doc(html_root_url = "https://docs.rs/magic-numbers")]
#![cfg_attr(not(feature = "std"), no_std)]

// Only impls, so not public
mod real;

/// Math operations
pub mod ops;
