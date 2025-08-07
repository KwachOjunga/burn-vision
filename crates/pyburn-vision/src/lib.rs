use pyo3::prelude::*;
pub mod layers;
pub mod models;

#[macro_export]
macro_rules! implement_send_and_sync {
    ($name:ty) => {
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
    };
}

#[pymodule]
mod pyburn_vision {

    use super::*;

    #[pymodule_export]
    use super::models::Alexnet;
}