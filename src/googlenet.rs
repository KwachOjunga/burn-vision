#![allow(unused)]

use burn::nn::{
    Relu,
    conv::Conv2dConfig,
    pool::{AvgPool2dConfig, MaxPool2dConfig},
};
/// GoogLeNet architecture as described in the the [paper](../docs/)


pub struct GoogLeNet {}

pub struct GoogLeNetConfig {}

struct BasicConV2D {}

//
struct Inception {}

// contains a conv2d layer and two fullly connected layers,
// a dropout layer and uses a relu function as its activation function
struct InceptionAux {}
