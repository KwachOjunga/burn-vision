use burn::prelude::*;
use burn::nn::Relu;


#[derive(Module, Debug, Clone)]
pub enum Activations {
    RELU(Relu)
}

impl Activations {
    pub fn forward<B : Backend>(&self, in_tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Activations::RELU(act) => act.forward(in_tensor)
        }
    }
}
