use pyo3::prelude::*;
// use pyo3::wrap_pymodule;
pub mod layers;
pub mod models;

/// A Python module implemented in Rust.
#[pymodule]
fn pyburn_vision(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(&layers::layers)?;
    Ok(())
}
