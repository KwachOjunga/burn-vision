#![recursion_limit = "256"]

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

    #[pymodule_export]
    use super::models::Alexnet;
    #[pymodule_export]
    use super::models::Densenet121;
    #[pymodule_export]
    use super::models::Efficientnetb0;
    #[pymodule_export]
    use super::models::Googlenet;
    #[pymodule_export]
    use super::models::Inceptionv3;
    #[pymodule_export]
    use super::models::Regnetx;

}
