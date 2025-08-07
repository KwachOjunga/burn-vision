#![allow(dead_code)]

use crate::implement_send_and_sync;
use burnvision::burn::backend::Wgpu;
use pyo3::prelude::*;

#[pyclass]
pub struct Alexnet(burnvision::alexnet::AlexNet<Wgpu>);

#[pyclass]
pub struct Densenet121(burnvision::densenet121::Model<Wgpu>);

#[pyclass]
pub struct Efficientnetb0(burnvision::efficientnet_b0::Model<Wgpu>);

#[pyclass]
pub struct Googlenet(burnvision::googlenet::Model<Wgpu>);

#[pyclass]
pub struct Inceptionv3(burnvision::inceptionv3::Model<Wgpu>);

#[pyclass]
pub struct Regnetx(burnvision::regnet_x_16gf::Model<Wgpu>);

implement_send_and_sync!(Alexnet);
implement_send_and_sync!(Densenet121);
implement_send_and_sync!(Efficientnetb0);
implement_send_and_sync!(Googlenet);
implement_send_and_sync!(Inceptionv3);
implement_send_and_sync!(Regnetx);
