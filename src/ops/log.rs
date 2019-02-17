/// Logarithmic functions.
pub trait Logarithmic {
    /// Returns the natural logarithm of the number.
    fn ln(self) -> Self;

    /// Returns the logarithm of the number with respect to an arbitrary base.
    fn log(self, base: Self) -> Self;

    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> Self;

    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> Self;

    /// Returns `ln(1+n)` (natural logarithm) more accurately than performing
    /// the operations separately.
    fn ln_1p(self) -> Self;
}
