/// The additive identity
pub trait Zero {
    /// Returns the additive identity of `Self`: `0`.
    ///
    /// # Laws
    ///
    /// ```{.text}
    /// a + 0 = a       ∀ a ∈ Self
    /// 0 + a = a       ∀ a ∈ Self
    /// ```
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    fn zero() -> Self;

    /// Returns `true` if `self` is equal to the additive identity.
    ///
    /// In some cases, this function is faster than comparing to the result of
    /// `zero()`.
    fn is_zero(self) -> bool;
}

/// Defines a multiplicative identity element for `Self`.
pub trait One {
    /// Returns the multiplicative identity element of `Self`, `1`.
    ///
    /// # Laws
    ///
    /// ```{.text}
    /// a * 1 = a       ∀ a ∈ Self
    /// 1 * a = a       ∀ a ∈ Self
    /// ```
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    fn one() -> Self;

    /// Returns `true` if `self` is equal to the multiplicative identity.
    ///
    /// In some cases, this function is faster than comparing to the result of
    /// `one()`.
    fn is_one(self) -> bool;
}
