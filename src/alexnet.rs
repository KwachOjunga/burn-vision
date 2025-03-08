use core::usize;

use burn::{
    nn::{conv::{Conv2d, Conv2dConfig}, pool::{AdaptiveAvgPool2d,AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig}, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu
    },
    prelude::*, 
};

#[derive(Module, Debug)]
pub struct AlexNet<B: Backend> {
    feature_extractor: FeatureExtractor<B>,
    avg_pool: AdaptiveAvgPool2d,
    classifier: AlexNetClassifier<B>    
}

#[derive(Config, Debug)]
pub struct AlexNetConfig{
    #[config(default = "1000")]
    num_classes: usize,
    hidden_size: usize,
    #[config( default = "0.5")]
    dropout: f64
}

impl AlexNetConfig {
    pub fn init<B: Backend>(&self, device: &<B as Backend>::Device, ) -> AlexNet<B> {
        let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
        let _reshaped = tensor.reshape([2, -1]);
        let features = FeatureExtractorConfig::init::<B>(device);
        let classifier = AlexNetClassifierConfig::init::<B>(device, self.num_classes);
        AlexNet {
            feature_extractor: features,
            avg_pool: AdaptiveAvgPool2dConfig::new([6,6]).init(),
            classifier
        }
    }
}

//[TODO]: remove the preprocessing steps
impl<B: Backend> AlexNet<B> {
    pub fn forward(&self, data: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, height, width] = data.dims();
        let x = data.reshape([batch_size, 1, height, width]);
        let x = self.feature_extractor.forward(x);
        let x = self.avg_pool.forward(x);
        let x = self.classifier.forward(x);
        x
    }
}

#[derive(Module, Debug)]
pub struct FeatureExtractor<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    maxpool1: MaxPool2d,
    maxpool2: MaxPool2d,
    maxpool3: MaxPool2d,
    activation: Relu,
}

pub struct FeatureExtractorConfig;

impl FeatureExtractorConfig {
    pub fn init<B: Backend>(device: &B::Device) -> FeatureExtractor<B>{
        FeatureExtractor {
            conv1: Conv2dConfig::new([3,64],[11,11]).with_padding(PaddingConfig2d::Explicit(2, 2)).init(device),
            conv2: Conv2dConfig::new([64,192],[5,5]).with_padding(PaddingConfig2d::Explicit(2, 2)).init(device),
            conv3: Conv2dConfig::new([192,384], [3,3]).init(device),
            conv4: Conv2dConfig::new([384,256], [3,3]).init(device),
            conv5: Conv2dConfig::new([256,256], [3,3]).init(device),
            maxpool1: MaxPool2dConfig::new([3,3]).with_strides([2,2]).init(),
            maxpool2: MaxPool2dConfig::new([3,3]).with_strides([2,2]).init(),
            maxpool3: MaxPool2dConfig::new([3,3]).with_strides([2,2]).init(),
            activation: Relu::new()
        }
    }
}
impl<B: Backend> FeatureExtractor<B>{
    pub fn forward(&self, data: Tensor<B, 4>) -> Tensor<B ,4>{
        // the nature of the output is explained by the return type of the final layer in self
        let x = self.conv1.forward(data);
        let x = self.activation.forward(x);
        let x = self.maxpool1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.maxpool2.forward(x);
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv4.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv5.forward(x);
        let x = self.activation.forward(x);
        self.maxpool3.forward(x)
    }
}
   

#[derive(Module, Debug)]
struct AlexNetClassifier<B: Backend>{
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

struct AlexNetClassifierConfig;

impl AlexNetClassifierConfig {
    pub fn init<B: Backend>(device: &B::Device, num_classes: usize) -> AlexNetClassifier<B>{
        AlexNetClassifier{
            linear1: LinearConfig::new(9216,4096).with_bias(true).init(device),
            linear2: LinearConfig::new(4096,4096).with_bias(true).init(device),
            linear3: LinearConfig::new(4096, num_classes).with_bias(true).init(device),
            dropout: DropoutConfig::new(0.5).init(),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> AlexNetClassifier<B>{
    pub fn forward(&self, data: Tensor<B, 4>)-> Tensor<B, 4>{
        let x = self.dropout.forward(data);
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
       let x = self.linear3.forward(x);
       x
    }
}
