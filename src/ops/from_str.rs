/// Create a number from a string and radix.
pub trait FromStrRadix: Sized {
    type ParseError;

    /// Converts a string slice in a given base to an integer.
    ///
    /// The string is expected to be an optional `+` or `-` sign followed by
    /// digits. Leading and trailing whitespace represent an error. Digits are a
    /// subset of these characters, depending on `radix`:
    ///
    /// * `0-9`
    /// * `a-z`
    /// * `A-Z`
    ///
    /// #  Panics
    ///
    /// This function may panic if `radix` is not 2 to 36.
    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::ParseError>;
}
