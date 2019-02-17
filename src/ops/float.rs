use core::num::FpCategory;

/// Operations for floating-point numbers.
pub trait FloatCore {
    /// Returns positive infinity.
    fn infinity() -> Self;

    /// Returns negative infinity.
    fn neg_infinity() -> Self;

    /// Returns NaN.
    fn nan() -> Self;

    /// Returns `-0.0`.
    fn neg_zero() -> Self;

    /// Returns the smallest positive, normalized value that this type can
    /// represent.
    fn min_positive_value() -> Self;

    /// Returns `true` if the number is NaN.
    fn is_nan(self) -> bool;

    /// Returns `true` if the number (assumed to be NaN) is signaling (rather
    /// than quiet). This method only makes sense for `NaN` floats. [See here
    /// for caveats](https://en.wikipedia.org/wiki/NaN#Encoding).
    fn is_signaling(self) -> bool;

    /// Returns `true` if the number is infinite.
    fn is_infinite(self) -> bool;

    /// Returns `true` if the number is neither infinite nor NaN.
    fn is_finite(self) -> bool;

    /// Returns `true` if the number is neither zero, infinite, subnormal nor
    /// NaN.
    fn is_normal(self) -> bool;

    /// Returns the floating point category of the number. When testing only one
    /// category, it is generally faster to use the specific `is_*()` method
    /// instead.
    fn classify(self) -> FpCategory;
}

/// Operations for floating-point numbers. Implementations for `f32` and `f64`
/// rely on `std`.
pub trait FloatRuntime {
    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> Self;

    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> Self;

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    fn round(self) -> Self;

    /// Return the integer part of a number.
    fn trunc(self) -> Self;

    /// Returns the fractional part of a number.
    fn fract(self) -> Self;
}

/// Decode a float into its mantissa, exponent, and sign.
pub trait FloatIntDecode<Mantissa, Exponent> {
    /// Returns the mantissa, base 2 exponent, and sign as integers,
    /// respectively. The original number can be recovered by
    /// `sign * mantissa * 2 ^ exponent`.
    fn integer_decode(self) -> (Mantissa, Exponent, i8);
}
