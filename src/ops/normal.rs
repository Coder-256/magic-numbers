/// Normalize a value.
pub trait Normalize {
    /// Normalize a value. For vectors, returns the normalized vector (a zero
    /// vector returns a zero vector). **For scalars, this returns the value of
    /// the `signum` function.** This means that for floats, `0.0` returns `1`
    /// and `-0.0` returns `-1`. See the documentation for `signum` (in `std`)
    /// for more info.
    fn normalize(self) -> Self;
}
