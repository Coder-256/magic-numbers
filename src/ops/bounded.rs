/// Numbers which have upper and lower bounds
pub trait Bounded {
    /// returns the smallest finite number this type can represent
    fn min_value() -> Self;
    /// returns the largest finite number this type can represent
    fn max_value() -> Self;
}
