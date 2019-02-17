/// Functions for signed numbers.
pub trait Signed {
    /// Returns a number that represents the sign of `self`.
    ///
    /// - `+1` if the number is positive, `+0`, or `+∞`
    /// - `-1` if the number is negative, `-0`, or `-∞`
    /// - `NaN` if the number is `NaN`
    fn signum(self) -> Self;

    /// Returns `true` if the sign of `self` is positive, including `+0`, `+∞`,
    /// and `NaN`.
    fn is_sign_positive(self) -> bool;

    /// Returns `true` if the sign of `self` is negative, including `-0`, `-∞`,
    /// and `-NaN`.
    fn is_sign_negative(self) -> bool;
}
