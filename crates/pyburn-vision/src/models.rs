#![allow(dead_code)]
use pyo3::prelude::*;

use burn_vision::alexnet::AlexNet;

#[pyclass]
struct Alexnet(AlwxNet);
