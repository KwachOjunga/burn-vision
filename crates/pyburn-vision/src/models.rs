#![allow(dead_code)]
use pyo3::prelude::*;
use crate::implement_send_and_sync;
use burnvision::burn::backend::Wgpu;

#[pyclass]
pub struct Alexnet( burnvision::alexnet::AlexNet<Wgpu>);

implement_send_and_sync!(Alexnet);
