# magic-numbers

[![crate
](https://img.shields.io/crates/v/magic-numbers.svg)
](https://crates.io/crates/magic-numbers)
[![documentation
](https://docs.rs/magic-numbers/badge.svg)
](https://docs.rs/magic-numbers)
[![rust 2018 edition
](https://img.shields.io/badge/rust-2018%20edition-brightgreen.svg)
](https://rust-lang-nursery.github.io/edition-guide/rust-2018/index.html)
[![Travis Status
](https://travis-ci.com/Coder-256/magic-numbers.svg?branch=master)
](https://travis-ci.com/Coder-256/magic-numbers)

A library for generic mathematics in Rust. This project started as a fork of the
[`num-traits` crate](https://github.com/rust-num/num-traits) created to
implement many postponed breaking changes, along with reorganizing the entire
system of crates into individual operations to allow more flexible generic
requirements. Since then, it has grown into a general-purpose math library.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
magic-numbers = "0.1"
# Uncomment this line to disable building with `std`:
# default-features = false
```

## Releases

Release notes are available in [RELEASES.md](RELEASES.md).

## Compatibility

The `magic-numbers` crate uses Rust 2018.

# Acknowledgements

This project includes code from the following projects:

- [`num-traits`](https://github.com/rust-num/num-traits)
- [`num-complex`](https://github.com/rust-num/num-complex)s

Thanks to everybody who contributed to these projects! `magic-numbers` would
have never existed without their help.
