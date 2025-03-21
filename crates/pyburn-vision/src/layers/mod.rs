#![allow(unused)]

use burn_vision::burn::{nn, optim::*};
use pyo3::prelude::*;
use pyo3::pymodule;

// pub mod activations;
// pub mod optimizations;

#[pymodule]
pub mod layers {
    /// core layer types
    use crate::pymodule;

    #[pymodule]
    mod activations {
        //e crate::layers::activations;
        use crate::pymodule;
    }
    #[pymodule]
    mod optimizers {
        //use crate::layers::optimizations;
        use crate::pymodule;
    }
}
