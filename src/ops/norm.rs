use core::num::NonZeroU8;

/// Calculate the p-norm of a number.
pub trait PNorm {
    type Output;

    /// Calculate the p-norm of a number. `p` is the `L^p` space.
    fn pnorm(self, p: NonZeroU8) -> Self::Output;
}

/// Calculate the euclidean norm (absolute value) of a number.
pub trait EuclideanNorm {
    type Output;

    /// Calculate the Euclidean norm of a number. Also known as the `L^2` norm,
    /// magnitude, absolute value, modulus, or just "norm". This is the same
    /// as `n.pnorm(2)` but may be implemented more efficiently.
    fn abs(self) -> Self::Output;
}
