/// Unary operator for retrieving the multiplicative inverse, or reciprocal, of a value.
pub trait Inv {
    /// Returns the multiplicative inverse (reciprocal) of `self`.
    fn inv(self) -> Self;
}
