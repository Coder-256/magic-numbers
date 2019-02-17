/// Create a method that returns a constant.
macro_rules! constant {
    ($( $method:ident () -> $ret:expr ; )*)
        => {$(
            fn $method() -> Self {
                $ret
            }
        )*};
}

/// Forward a method to an inherent method or a base trait method.
macro_rules! forward {
    ($( Self :: $method:ident ( $( $arg:ident $( : $ty:ty )? ),* ) -> $ret:ty ; )*)
        => {$(
            fn $method($( $arg $( : $ty )? ),*) -> $ret {
                Self::$method($( $arg ),*)
            }
        )*};
}
