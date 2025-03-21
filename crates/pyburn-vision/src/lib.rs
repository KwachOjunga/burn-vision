use pyo3::prelude::*;
use pyo3::wrap_pymodule;
pub mod layers;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyburn_vision(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(&layers::layers)?;
    Ok(())
}
