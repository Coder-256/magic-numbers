use super::super::ops::*;
use core::num::{NonZeroU8, ParseIntError};

macro_rules! signed {
    ($t:ty) => {
        impl Normalize for $t {
            fn normalize(self) -> Self {
                self.signum()
            }
        }

        impl PNorm for $t {
            type NormOutput = $t;

            fn pnorm(self, p: NonZeroU8) -> Self::NormOutput {
                if p.get() % 2 == 0 && self.is_negative() {
                    -self
                } else {
                    self
                }
            }
        }
    };
}

macro_rules! unsigned {
    ($t:ty) => {
        impl Normalize for $t {
            fn normalize(self) -> Self {
                match self {
                    0 => 0,
                    _ => 1,
                }
            }
        }

        impl PNorm for $t {
            type NormOutput = $t;

            fn pnorm(self, _: NonZeroU8) -> Self::NormOutput {
                self
            }
        }
    };
}

macro_rules! int_shared {
    ($t:ty, $s:ty, $u:ty, $c:expr) => {
        impl Bitwise for $t {
            type Bytes = [u8; $c];
            forward! {
                Self::count_ones(self) -> u32;
                Self::count_zeros(self) -> u32;
                Self::leading_zeros(self) -> u32;
                Self::trailing_zeros(self) -> u32;
                Self::rotate_left(self, n: u32) -> Self;
                Self::rotate_right(self, n: u32) -> Self;
                Self::swap_bytes(self) -> Self;
                Self::from_be(x: Self) -> Self;
                Self::from_le(x: Self) -> Self;
                Self::to_be(self) -> Self;
                Self::to_le(self) -> Self;
                Self::pow(self, exp: u32) -> Self;
                Self::to_be_bytes(self) -> Self::Bytes;
                Self::to_le_bytes(self) -> Self::Bytes;
                Self::to_ne_bytes(self) -> Self::Bytes;
                Self::from_be_bytes(bytes: Self::Bytes) -> Self;
                Self::from_le_bytes(bytes: Self::Bytes) -> Self;
                Self::from_ne_bytes(bytes: Self::Bytes) -> Self;
            }

            fn signed_shl(self, n: u32) -> Self {
                ((self as $s) << n) as $t
            }
            fn signed_shr(self, n: u32) -> Self {
                ((self as $s) >> n) as $t
            }
            fn unsigned_shl(self, n: u32) -> Self {
                ((self as $u) << n) as $t
            }
            fn unsigned_shr(self, n: u32) -> Self {
                ((self as $u) >> n) as $t
            }
        }

        impl Bounded for $t {
            forward! {
                Self::min_value() -> Self;
                Self::max_value() -> Self;
            }
        }

        impl FromStrRadix for $t {
            type ParseError = ParseIntError;
            forward! {
                Self::from_str_radix(src: &str,
                    radix: u32) -> Result<Self, Self::ParseError>;
            }
        }

        impl MulAdd for $t {
            type Output = Self;
            fn mul_add(self, a: Self, b: Self) -> Self {
                (self * a) + b
            }
        }

        impl MulAddAssign for $t {
            fn mul_add_assign(&mut self, a: Self, b: Self) {
                *self = (*self * a) + b
            }
        }

        impl One for $t {
            fn one() -> Self {
                1
            }

            fn is_one(self) -> bool {
                self == 1
            }
        }

        impl Power<u32> for $t {
            fn pow(self, exp: u32) -> Self {
                self.pow(exp)
            }
        }

        impl Zero for $t {
            fn zero() -> Self {
                0
            }

            fn is_zero(self) -> bool {
                self == 0
            }
        }
    };
}

macro_rules! int_impl {
    ($s:ty, $u:ty, $c:expr) => {
        signed!($s);
        unsigned!($u);
        int_shared!($s, $s, $u, $c);
        int_shared!($u, $s, $u, $c);
    };
}

int_impl!(isize, usize, 8);
int_impl!(i8, u8, 1);
int_impl!(i16, u16, 2);
int_impl!(i32, u32, 4);
int_impl!(i64, u64, 8);
int_impl!(i128, u128, 16);
