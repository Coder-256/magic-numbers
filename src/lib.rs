#![warn(clippy::all)]
#![deny(unconditional_recursion)]
#![doc(html_root_url = "https://docs.rs/magic-numbers")]
#![cfg_attr(not(feature = "std"), no_std)]

mod impls;
pub use impls::*;

mod ops;
pub use ops::*;
