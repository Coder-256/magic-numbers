use core::num::NonZeroU8;

/// Calculate the p-norm of a number.
pub trait PNorm {
    type NormOutput;

    /// Calculate the p-norm of a number. `p` is the `L^p` space.
    fn pnorm(self, p: NonZeroU8) -> Self::NormOutput;
}

/// Calculate the euclidean norm (absolute value) of a number.
pub trait EuclideanNorm {
    type NormOutput;

    /// Calculate the Euclidean norm of a number. Also known as the `L^2` norm,
    /// magnitude, absolute value, modulus, or just "norm".
    fn abs(self) -> Self::NormOutput;
}

impl<T: PNorm> EuclideanNorm for T {
    type NormOutput = T::NormOutput;

    fn abs(self) -> T::NormOutput {
        self.pnorm(unsafe { NonZeroU8::new_unchecked(2) })
    }
}
