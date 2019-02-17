/// Functions to convert between radians and degrees.
pub trait Angles {
    /// Converts radians to degrees.
    fn to_degrees(self) -> Self;

    /// Converts degrees to radians.
    fn to_radians(self) -> Self;
}
