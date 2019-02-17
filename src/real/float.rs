use super::super::ops::*;
#[cfg(feature = "std")]
use core::num::NonZeroU8;
use core::{f32, f64, mem, num::FpCategory};

macro_rules! float_impl {
    ($t:ident, $decode:ident, $signal:expr) => {
        impl Angles for $t {
            forward! {
                Self::to_degrees(self) -> Self;
                Self::to_radians(self) -> Self;
            }
        }

        impl Bounded for $t {
            constant! {
                min_value() -> $t::MIN;
                max_value() -> $t::MAX;
            }
        }

        impl Epsilon for $t {
            constant! {
                epsilon() -> $t::EPSILON;
            }
        }

        #[cfg(feature = "std")]
        impl Exponential for $t {
            forward! {
                Self::sqrt(self) -> Self;
                Self::cbrt(self) -> Self;
                Self::exp(self) -> Self;
                Self::exp2(self) -> Self;
                Self::hypot(self, other: Self) -> Self;
                Self::exp_m1(self) -> Self;
            }
        }

        impl FloatCore for $t {
            constant! {
                nan() -> $t::NAN;
                infinity() -> $t::INFINITY;
                neg_infinity() -> $t::NEG_INFINITY;
                neg_zero() -> -0.0;
                min_positive_value() -> $t::MIN_POSITIVE;
            }

            forward! {
                Self::is_nan(self) -> bool;
                Self::is_infinite(self) -> bool;
                Self::is_finite(self) -> bool;
                Self::is_normal(self) -> bool;
                Self::classify(self) -> FpCategory;
            }

            fn is_signaling(self) -> bool {
                (self.to_bits() & $signal) == 0
            }
        }

        #[cfg(feature = "std")]
        impl FloatRuntime for $t {
            forward! {
                Self::floor(self) -> Self;
                Self::ceil(self) -> Self;
                Self::round(self) -> Self;
                Self::trunc(self) -> Self;
                Self::fract(self) -> Self;
            }
        }

        #[cfg(feature = "std")]
        impl Hyperbolic for $t {
            forward! {
                Self::sinh(self) -> Self;
                Self::cosh(self) -> Self;
                Self::tanh(self) -> Self;
                Self::asinh(self) -> Self;
                Self::acosh(self) -> Self;
                Self::atanh(self) -> Self;
            }
        }

        impl Inv for $t {
            fn inv(self) -> Self {
                self.recip()
            }
        }

        #[cfg(feature = "std")]
        impl Logarithmic for $t {
            forward! {
                Self::ln(self) -> Self;
                Self::log(self, base: Self) -> Self;
                Self::log2(self) -> Self;
                Self::log10(self) -> Self;
                Self::ln_1p(self) -> Self;
            }
        }

        #[cfg(feature = "std")]
        impl MulAdd for $t {
            type Output = Self;
            forward! {
                Self::mul_add(self, a: Self, b: Self) -> Self;
            }
        }

        #[cfg(feature = "std")]
        impl Normalize for $t {
            fn normalize(self) -> Self {
                self.signum()
            }
        }

        impl One for $t {
            fn one() -> Self {
                1.0
            }

            #[allow(clippy::float_cmp)]
            fn is_one(self) -> bool {
                self == 1.0
            }
        }

        #[cfg(feature = "std")]
        impl PNorm for $t {
            type NormOutput = $t;

            fn pnorm(self, p: NonZeroU8) -> Self::NormOutput {
                if p.get() % 2 == 0 {
                    self.abs()
                } else {
                    self
                }
            }
        }

        #[cfg(feature = "std")]
        impl Power<i32> for $t {
            fn pow(self, exp: i32) -> Self {
                self.powi(exp)
            }
        }

        #[cfg(feature = "std")]
        impl Power<$t> for $t {
            fn pow(self, exp: Self) -> Self {
                self.powf(exp)
            }
        }

        #[cfg(feature = "std")]
        impl Signed for $t {
            forward! {
                Self::signum(self) -> Self;
                Self::is_sign_positive(self) -> bool;
                Self::is_sign_negative(self) -> bool;
            }
        }

        #[cfg(feature = "std")]
        impl Trigonometric for $t {
            forward! {
                Self::sin(self) -> Self;
                Self::cos(self) -> Self;
                Self::tan(self) -> Self;
                Self::asin(self) -> Self;
                Self::acos(self) -> Self;
                Self::atan(self) -> Self;
                Self::atan2(self, other: Self) -> Self;
                Self::sin_cos(self) -> (Self, Self);
            }
        }

        impl Zero for $t {
            fn zero() -> Self {
                0.0
            }

            fn is_zero(self) -> bool {
                self == 0.0
            }
        }
    };
}

impl FloatIntDecode<u32, i16> for f32 {
    fn integer_decode(self) -> (u32, i16, i8) {
        let bits: u32 = unsafe { mem::transmute(self) };
        let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x007f_ffff) << 1
        } else {
            (bits & 0x007f_ffff) | 0x0080_0000
        };
        // Exponent bias + mantissa shift
        exponent -= 127 + 23;
        (mantissa, exponent, sign)
    }
}

impl FloatIntDecode<u64, i16> for f64 {
    fn integer_decode(self) -> (u64, i16, i8) {
        let bits: u64 = unsafe { mem::transmute(self) };
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x000f_ffff_ffff_ffff) << 1
        } else {
            (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
        };
        // Exponent bias + mantissa shift
        exponent -= 1023 + 52;
        (mantissa, exponent, sign)
    }
}

// According to IEEE 754-2008, binary floats have (in order of MSB to LSB) 1
// sign bit, `w` bits for the exponent, and the remaining `t` bits are used for
// the trailing significand; t=23 and t=52 for 32- and 64-bit floats
// respectively (§5.4). The signaling bit is the first (aka most significant)
// bit of the trailing significand (the signaling bit is cleared for signaling
// NaNs and set for quiet NaNs) (§8.2.1). Therefore the signaling bitmask is
// `1 << (t - 1)`.
float_impl!(f32, integer_decode_f32, 1 << (23 - 1));
float_impl!(f64, integer_decode_f64, 1 << (52 - 1));
