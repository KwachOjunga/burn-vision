use burn::nn::Relu;
use burn::prelude::*;

#[derive(Module, Debug, Clone)]
pub enum Activations {
    RELU(Relu),
}

impl Activations {
    pub fn forward<B: Backend>(&self, in_tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Activations::RELU(func) => func.forward(in_tensor),
        }
    }
}
