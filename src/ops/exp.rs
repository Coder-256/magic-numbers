/// Exponential functions.
pub trait Exponential {
    /// Take the square root of a number.
    fn sqrt(self) -> Self;

    /// Take the cubic root of a number.
    fn cbrt(self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> Self;

    /// Returns `2^(self)`.
    fn exp2(self) -> Self;

    /// Calculate the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    fn hypot(self, other: Self) -> Self;

    /// Returns `e^(self) - 1` in a way that is accurate even if the number is
    /// close to zero.
    fn exp_m1(self) -> Self;
}

/// Raise a number to a power.
pub trait Power<T> {
    /// Raise a number to a power.
    fn pow(self, exp: T) -> Self;
}
