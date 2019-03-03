/// Hyperbolic functions.
pub trait Hyperbolic {
    /// Hyperbolic sine.
    fn sinh(self) -> Self;

    /// Hyperbolic cosine.
    fn cosh(self) -> Self;

    /// Hyperbolic tangent.
    fn tanh(self) -> Self;

    /// Inverse hyperbolic sine.
    fn asinh(self) -> Self;

    /// Inverse hyperbolic cosine.
    fn acosh(self) -> Self;

    /// Inverse hyperbolic tangent.
    fn atanh(self) -> Self;

    /// Simultaneously computes the hyperbolic sine and cosine of the number,
    /// `x`. Returns `(sinh(x), cosh(x))`.
    fn sinh_cosh(self) -> (Self, Self)
    where
        Self: Sized;
}
