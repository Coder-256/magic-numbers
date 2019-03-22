use crate::ops::*;
use core::ops::*;

/// Basic operations that are required for the parts of a complex number.
pub trait ComplexPart:
    Sized
    + Copy
    + PartialEq
    + Signed
    + One
    + Zero
    + Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    // + Rem<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + DivAssign
    + MulAssign
    // + RemAssign
    + SubAssign
{
}

impl<T> ComplexPart for T where
    T: Copy
        + PartialEq
        + Signed
        + One
        + Zero
        + Add<Output = Self>
        + Div<Output = Self>
        + Mul<Output = Self>
        + Neg<Output = Self>
        + Sub<Output = Self>
        + AddAssign
        + DivAssign
        + MulAssign
        + SubAssign
{
}

/// A complex number in Cartesian form.
///
/// ## Representation and Foreign Function Interface Compatibility
///
/// `Complex<T>` is memory layout compatible with an array `[T; 2]`.
///
/// `Complex<F>` where F is a floating point type is guaranteed to be memory
/// layout compatible with C, but it may or may not be calling convention
/// compatible. Please test this on all target platforms if you pass a complex
/// number by value! Alternatively, `Complex` should always work well in C
/// behind a pointer.
#[derive(PartialEq, Copy, Clone, Hash, Debug, Default)]
#[repr(C)]
pub struct Complex<T: ComplexPart> {
    /// Real part of the complex number
    pub re: T,
    /// Imaginary part of the complex number
    pub im: T,
}

impl<T: ComplexPart> Complex<T> {
    pub fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }

    /// Returns the imaginary unit
    pub fn i() -> Self {
        Complex::new(T::zero(), T::one())
    }

    /// Returns the square of the norm, i.e. `re^2 + im^2`.
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// Returns the complex conjugate.
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }
}

impl<T: ComplexPart> Complex<T>
where
    T: Atan2,
{
    /// Calculate the argument (also called phase or angle).
    pub fn arg(self) -> T {
        // https://en.wikipedia.org/wiki/Atan2
        self.im.atan2(self.re)
    }
}

impl<T: ComplexPart> Complex<T>
where
    T: Exponential + Atan2,
{
    /// Convert to polar form (r, theta), such that
    /// `self = r * exp(i * theta)`
    pub fn to_polar(self) -> (T, T) {
        (self.abs(), self.arg())
    }
}

impl<T: ComplexPart> Complex<T>
where
    T: Copy + Exponential + Atan2 + Trigonometric + Mul<Output = T>,
{
    /// Convert a polar representation into a complex number.
    pub fn from_polar(r: T, theta: T) -> Complex<T> {
        Complex::new(r * theta.cos(), r * theta.sin())
    }

    /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
    pub fn exp(self) -> Complex<T> {
        // formula: e^(a + bi) = e^a (cos(b) + i*sin(b))
        // = from_polar(e^a, b)
        Complex::from_polar(self.re.exp(), self.im)
    }
}

impl<T: ComplexPart> Complex<T>
where
    T: Exponential + Atan2 + Logarithmic,
{
    /// Computes the **principal** value of natural logarithm of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 0]`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ arg(ln(z)) ≤ π`.
    pub fn ln(self) -> Complex<T> {
        // formula: ln(z) = ln(r) + i*theta
        let (r, theta) = self.to_polar();
        Complex::new(r.ln(), theta)
    }

    /// Returns the logarithm of `self` with respect to an arbitrary base.
    pub fn log(&self, base: T) -> Complex<T> {
        // formula: log_y(x) = log_y(ρ e^(i θ))
        // = log_y(ρ) + log_y(e^(i θ)) = log_y(ρ) + ln(e^(i θ)) / ln(y)
        // = log_y(ρ) + i θ / ln(y)
        let (r, theta) = self.to_polar();
        Complex::new(r.log(base), theta / base.ln())
    }
}

impl<T: ComplexPart> Complex<T>
where
    T: Exponential + Atan2 + Trigonometric,
{
    /// Computes the principal value of the square root of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 0)`, continuous from above.
    ///
    /// The branch satisfies `-π/2 ≤ arg(sqrt(z)) ≤ π/2`.
    pub fn sqrt(self) -> Complex<T> {
        // formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
        // = (sqrt(r), theta/2) in polar
        let two = T::one() + T::one();
        let (r, theta) = self.to_polar();
        Complex::from_polar(r.sqrt(), theta / two)
    }
}

impl<T: ComplexPart> Complex<T>
where
    T: Exponential + Trigonometric + Atan2 + Logarithmic + Power<T>,
{
    /// Raises a floating point number to the complex power `self`.
    pub fn expf(self, base: T) -> Complex<T> {
        // formula: x^(a+bi) = x^a x^bi = x^a e^(b ln(x) i)
        // = from_polar(x^a, b ln(x))
        Complex::from_polar(base.pow(self.re), self.im * base.ln())
    }
}

impl<T: ComplexPart> Complex<T>
where
    T: FloatCore,
{
    /// Checks if the given complex number is NaN
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Checks if the given complex number is infinite
    pub fn is_infinite(self) -> bool {
        !self.is_nan() && (self.re.is_infinite() || self.im.is_infinite())
    }

    /// Checks if the given complex number is finite
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Checks if the given complex number is normal
    pub fn is_normal(self) -> bool {
        self.re.is_normal() && self.im.is_normal()
    }
}

// -----------
// Trait impls
// -----------

impl<T: ComplexPart> Add<Self> for Complex<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

impl<T: ComplexPart> Add<T> for Complex<T> {
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        Complex::new(self.re + other, self.im)
    }
}

impl<T: ComplexPart> AddAssign<Self> for Complex<T> {
    fn add_assign(&mut self, other: Self) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl<T: ComplexPart> AddAssign<T> for Complex<T> {
    fn add_assign(&mut self, other: T) {
        self.re += other;
    }
}

impl<T: ComplexPart> Div<Self> for Complex<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        // formula: (a + bi)/(c + di)
        // = (a + bi)(c - di)/(c^2 + d^2)
        // = (ac - adi + bci + bd)/(c^2 + d^2)
        // = ((ac + bd) + (bc - ad)i)/(c^2 + d^2)
        let den = other.norm_sqr();
        Complex::new(
            (self.re * other.re + self.im * other.im) / den,
            (self.im * other.re - self.re * other.im) / den,
        )
    }
}

impl<T: ComplexPart> Div<T> for Complex<T> {
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        Complex::new(self.re / other, self.im / other)
    }
}

impl<T: ComplexPart> DivAssign<Self> for Complex<T> {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl<T: ComplexPart> DivAssign<T> for Complex<T> {
    fn div_assign(&mut self, other: T) {
        self.re /= other;
        self.im /= other;
    }
}

impl<T: ComplexPart> AbsoluteValue for Complex<T>
where
    T: Exponential,
{
    type Output = T;

    /// This should be equal to the square root of `norm_sqr()`.
    fn abs(self) -> T {
        self.re.hypot(self.im)
    }
}

impl<T: ComplexPart> From<T> for Complex<T> {
    fn from(value: T) -> Self {
        Complex::new(value, T::zero())
    }
}

impl<T: ComplexPart> Inv for Complex<T> {
    /// Returns the reciprocal.
    fn inv(self) -> Self {
        // formula: 1/(a+bi) = (a-bi)/(a^2+b^2)
        let norm_sqr = self.norm_sqr();
        Complex::new(self.re / norm_sqr, -self.im / norm_sqr)
    }
}

impl<T: ComplexPart> Mul<Self> for Complex<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        // formula: (a + bi)(c + di)
        // = ac + adi + bci - bd
        // = (ac - bd) + (ad + bc)i
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl<T: ComplexPart> Mul<T> for Complex<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Complex::new(self.re * other, self.im * other)
    }
}

impl<T: ComplexPart> MulAssign<Self> for Complex<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<T: ComplexPart> MulAssign<T> for Complex<T> {
    fn mul_assign(&mut self, other: T) {
        self.re *= other;
        self.im *= other;
    }
}

impl<T: ComplexPart> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: ComplexPart> One for Complex<T> {
    fn one() -> Self {
        Complex::new(T::one(), T::zero())
    }

    fn is_one(self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }
}

impl<T: ComplexPart> PartialEq<T> for Complex<T> {
    fn eq(&self, other: &T) -> bool {
        self.im.is_zero() && self.re == *other
    }
}

impl<T: ComplexPart> PNorm for Complex<T>
where
    T: Inv + From<u8> + Power<u8> + Power<T>,
{
    type Output = T;

    fn pnorm(self, p: core::num::NonZeroU8) -> Self::Output {
        let pnum = p.get();
        (self.re.pow(pnum) + self.im.pow(pnum)).pow(T::from(pnum).inv())
    }
}

impl<T: ComplexPart> Power<T> for Complex<T>
where
    T: Power<T> + Exponential + Atan2 + Trigonometric,
{
    /// Raises `self` to a power.
    fn pow(self, exp: T) -> Complex<T> {
        // formula: x^y = (ρ e^(i θ))^y = ρ^y e^(i θ y)
        // = from_polar(ρ^y, θ y)
        let (r, theta) = self.to_polar();
        Complex::from_polar(r.pow(exp), theta * exp)
    }
}

impl<T: ComplexPart> Power<Complex<T>> for Complex<T>
where
    T: Copy + Exponential + Atan2 + Trigonometric + Logarithmic + Power<T>,
{
    /// Raises `self` to a complex power.
    fn pow(self, exp: Complex<T>) -> Complex<T> {
        // formula: x^y = (a + i b)^(c + i d)
        // = (ρ e^(i θ))^c (ρ e^(i θ))^(i d)
        //    where ρ=|x| and θ=arg(x)
        // = ρ^c e^(−d θ) e^(i c θ) ρ^(i d)
        // = p^c e^(−d θ) (cos(c θ)
        //   + i sin(c θ)) (cos(d ln(ρ)) + i sin(d ln(ρ)))
        // = p^c e^(−d θ) (
        //   cos(c θ) cos(d ln(ρ)) − sin(c θ) sin(d ln(ρ))
        //   + i(cos(c θ) sin(d ln(ρ)) + sin(c θ) cos(d ln(ρ))))
        // = p^c e^(−d θ) (cos(c θ + d ln(ρ)) + i sin(c θ + d ln(ρ)))
        // = from_polar(p^c e^(−d θ), c θ + d ln(ρ))
        let (r, theta) = self.to_polar();
        Complex::from_polar(
            r.pow(exp.re) * (-exp.im * theta).exp(),
            exp.re * theta + exp.im * r.ln(),
        )
    }
}

// TODO: Rem?

impl<T: ComplexPart> Sub<Self> for Complex<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

impl<T: ComplexPart> Sub<T> for Complex<T> {
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        Complex::new(self.re - other, self.im)
    }
}

impl<T: ComplexPart> SubAssign<Self> for Complex<T> {
    fn sub_assign(&mut self, other: Self) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl<T: ComplexPart> SubAssign<T> for Complex<T> {
    fn sub_assign(&mut self, other: T) {
        self.re -= other;
    }
}

impl<T: ComplexPart> Trigonometric for Complex<T>
where
    T: Copy + Trigonometric + Hyperbolic + Exponential + Atan2 + Logarithmic + FloatCore,
{
    /// Computes the sine of `self`.
    fn sin(self) -> Self {
        // formula: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        let (re_sin, re_cos) = self.re.sin_cos();
        let (im_sinh, im_cosh) = self.im.sinh_cosh();
        Complex::new(re_sin * im_cosh, re_cos * im_sinh)
    }

    /// Computes the cosine of `self`.
    fn cos(self) -> Self {
        // formula: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        let (re_sin, re_cos) = self.re.sin_cos();
        let (im_sinh, im_cosh) = self.im.sinh_cosh();
        Complex::new(re_cos * im_cosh, -re_sin * im_sinh)
    }

    /// Computes the tangent of `self`.
    fn tan(self) -> Self {
        // formula: tan(a + bi) = (sin(2a) + i*sinh(2b))/(cos(2a) + cosh(2b))
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);
        let (two_re_sin, two_re_cos) = two_re.sin_cos();
        let (two_im_sinh, two_im_cosh) = two_im.sinh_cosh();
        let den = two_re_cos + two_im_cosh;
        Complex::new(two_re_sin / den, two_im_sinh / den)
    }

    /// Computes the principal value of the inverse sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Re(asin(z)) ≤ π/2`.
    fn asin(self) -> Self {
        // formula: arcsin(z) = -i ln(sqrt(1-z^2) + iz)
        let i = Complex::<T>::i();
        -i * ((Complex::<T>::one() - self * self).sqrt() + i * self).ln()
    }

    /// Computes the principal value of the inverse cosine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `0 ≤ Re(acos(z)) ≤ π`.
    fn acos(self) -> Self {
        // formula: arccos(z) = -i ln(i sqrt(1-z^2) + z)
        let i = Complex::<T>::i();
        -i * (i * (Complex::<T>::one() - self * self).sqrt() + self).ln()
    }

    /// Computes the principal value of the inverse tangent of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞i, -i]`, continuous from the left.
    /// * `[i, ∞i)`, continuous from the right.
    ///
    /// The branch satisfies `-π/2 ≤ Re(atan(z)) ≤ π/2`.
    fn atan(self) -> Self {
        // formula: arctan(z) = (ln(1+iz) - ln(1-iz))/(2i)
        let i = Complex::<T>::i();
        let one = Complex::<T>::one();
        let two = one + one;
        if self.re.is_zero() {
            if self.im.is_one() {
                return Complex::new(T::zero(), T::infinity());
            } else if (-self.im).is_one() {
                return Complex::new(T::zero(), T::neg_infinity());
            }
        }
        ((one + i * self).ln() - (one - i * self).ln()) / (two * i)
    }

    fn sin_cos(self) -> (Self, Self) {
        // formula: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        // formula: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)

        let (re_sin, re_cos) = self.re.sin_cos();
        let (im_sinh, im_cosh) = self.im.sinh_cosh();

        (
            Complex::new(re_sin * im_cosh, re_cos * im_sinh),
            Complex::new(re_cos * im_cosh, -re_sin * im_cosh),
        )
    }
}

impl<T: ComplexPart> Zero for Complex<T> {
    fn zero() -> Self {
        Complex::new(T::zero(), T::zero())
    }

    fn is_zero(self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
}
