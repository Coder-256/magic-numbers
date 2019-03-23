/// Trigonometric functions
pub trait Trigonometric {
    /// Computes the sine of a number (in radians).
    fn sin(self) -> Self;

    /// Computes the cosine of a number (in radians).
    fn cos(self) -> Self;

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    fn sin_cos(self) -> (Self, Self)
    where
        Self: Sized;

    /// Computes the tangent of a number (in radians).
    fn tan(self) -> Self;

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range `[-π/2, π/2]`, or NaN if the number is outside the range
    /// `[-1, 1]`.
    fn asin(self) -> Self;

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range `[0, π]` or NaN if the number is outside the range`[-1, 1]`.
    fn acos(self) -> Self;

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range `[-π/2, π/2]`
    fn atan(self) -> Self;
}

/// The [`atan2`](https://en.wikipedia.org/wiki/Atan2) function.
pub trait Atan2 {
    /// Computes the four-quadrant arctangent of `self` (`y`) and `other` (`x`).
    ///
    /// * `x = 0`, `y = 0`: `0`
    /// * `x >= 0`: `arctan(y/x)` -> `[-π/2, π/2]`
    /// * `y >= 0`: `arctan(y/x) + pi` -> `(π/2, π]`
    /// * `y < 0`: `arctan(y/x) - pi` -> `(-π, -π/2)`
    fn atan2(self, other: Self) -> Self;
}
