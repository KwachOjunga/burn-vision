// Generated from ONNX "densenet121.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    batchnormalization1: BatchNorm<B, 2>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    batchnormalization2: BatchNorm<B, 2>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    batchnormalization3: BatchNorm<B, 2>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    batchnormalization4: BatchNorm<B, 2>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    batchnormalization5: BatchNorm<B, 2>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    batchnormalization6: BatchNorm<B, 2>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    batchnormalization7: BatchNorm<B, 2>,
    conv2d14: Conv2d<B>,
    averagepool2d1: AvgPool2d,
    batchnormalization8: BatchNorm<B, 2>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    batchnormalization9: BatchNorm<B, 2>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    batchnormalization10: BatchNorm<B, 2>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    batchnormalization11: BatchNorm<B, 2>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    batchnormalization12: BatchNorm<B, 2>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    batchnormalization13: BatchNorm<B, 2>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    batchnormalization14: BatchNorm<B, 2>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    batchnormalization15: BatchNorm<B, 2>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    batchnormalization16: BatchNorm<B, 2>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    batchnormalization17: BatchNorm<B, 2>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    batchnormalization18: BatchNorm<B, 2>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    batchnormalization19: BatchNorm<B, 2>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    batchnormalization20: BatchNorm<B, 2>,
    conv2d39: Conv2d<B>,
    averagepool2d2: AvgPool2d,
    batchnormalization21: BatchNorm<B, 2>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    batchnormalization22: BatchNorm<B, 2>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    batchnormalization23: BatchNorm<B, 2>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    batchnormalization24: BatchNorm<B, 2>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    batchnormalization25: BatchNorm<B, 2>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    batchnormalization26: BatchNorm<B, 2>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    batchnormalization27: BatchNorm<B, 2>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    batchnormalization28: BatchNorm<B, 2>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    batchnormalization29: BatchNorm<B, 2>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    batchnormalization30: BatchNorm<B, 2>,
    conv2d58: Conv2d<B>,
    conv2d59: Conv2d<B>,
    batchnormalization31: BatchNorm<B, 2>,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    batchnormalization32: BatchNorm<B, 2>,
    conv2d62: Conv2d<B>,
    conv2d63: Conv2d<B>,
    batchnormalization33: BatchNorm<B, 2>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    batchnormalization34: BatchNorm<B, 2>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    batchnormalization35: BatchNorm<B, 2>,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    batchnormalization36: BatchNorm<B, 2>,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    batchnormalization37: BatchNorm<B, 2>,
    conv2d72: Conv2d<B>,
    conv2d73: Conv2d<B>,
    batchnormalization38: BatchNorm<B, 2>,
    conv2d74: Conv2d<B>,
    conv2d75: Conv2d<B>,
    batchnormalization39: BatchNorm<B, 2>,
    conv2d76: Conv2d<B>,
    conv2d77: Conv2d<B>,
    batchnormalization40: BatchNorm<B, 2>,
    conv2d78: Conv2d<B>,
    conv2d79: Conv2d<B>,
    batchnormalization41: BatchNorm<B, 2>,
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    batchnormalization42: BatchNorm<B, 2>,
    conv2d82: Conv2d<B>,
    conv2d83: Conv2d<B>,
    batchnormalization43: BatchNorm<B, 2>,
    conv2d84: Conv2d<B>,
    conv2d85: Conv2d<B>,
    batchnormalization44: BatchNorm<B, 2>,
    conv2d86: Conv2d<B>,
    conv2d87: Conv2d<B>,
    batchnormalization45: BatchNorm<B, 2>,
    conv2d88: Conv2d<B>,
    averagepool2d3: AvgPool2d,
    batchnormalization46: BatchNorm<B, 2>,
    conv2d89: Conv2d<B>,
    conv2d90: Conv2d<B>,
    batchnormalization47: BatchNorm<B, 2>,
    conv2d91: Conv2d<B>,
    conv2d92: Conv2d<B>,
    batchnormalization48: BatchNorm<B, 2>,
    conv2d93: Conv2d<B>,
    conv2d94: Conv2d<B>,
    batchnormalization49: BatchNorm<B, 2>,
    conv2d95: Conv2d<B>,
    conv2d96: Conv2d<B>,
    batchnormalization50: BatchNorm<B, 2>,
    conv2d97: Conv2d<B>,
    conv2d98: Conv2d<B>,
    batchnormalization51: BatchNorm<B, 2>,
    conv2d99: Conv2d<B>,
    conv2d100: Conv2d<B>,
    batchnormalization52: BatchNorm<B, 2>,
    conv2d101: Conv2d<B>,
    conv2d102: Conv2d<B>,
    batchnormalization53: BatchNorm<B, 2>,
    conv2d103: Conv2d<B>,
    conv2d104: Conv2d<B>,
    batchnormalization54: BatchNorm<B, 2>,
    conv2d105: Conv2d<B>,
    conv2d106: Conv2d<B>,
    batchnormalization55: BatchNorm<B, 2>,
    conv2d107: Conv2d<B>,
    conv2d108: Conv2d<B>,
    batchnormalization56: BatchNorm<B, 2>,
    conv2d109: Conv2d<B>,
    conv2d110: Conv2d<B>,
    batchnormalization57: BatchNorm<B, 2>,
    conv2d111: Conv2d<B>,
    conv2d112: Conv2d<B>,
    batchnormalization58: BatchNorm<B, 2>,
    conv2d113: Conv2d<B>,
    conv2d114: Conv2d<B>,
    batchnormalization59: BatchNorm<B, 2>,
    conv2d115: Conv2d<B>,
    conv2d116: Conv2d<B>,
    batchnormalization60: BatchNorm<B, 2>,
    conv2d117: Conv2d<B>,
    conv2d118: Conv2d<B>,
    batchnormalization61: BatchNorm<B, 2>,
    conv2d119: Conv2d<B>,
    conv2d120: Conv2d<B>,
    batchnormalization62: BatchNorm<B, 2>,
    globalaveragepool1: AdaptiveAvgPool2d,
    gemm1: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("../../models/onnx_dir/densenet121", &Default::default())
    }
}

impl<B: Backend> Model<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Record file to exist.");
        Self::new(device).load_record(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let batchnormalization1 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(96)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d4 = Conv2dConfig::new([96, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d6 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d8 = Conv2dConfig::new([160, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(192)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d10 = Conv2dConfig::new([192, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(224)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d12 = Conv2dConfig::new([224, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization7 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d14 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool2d1 = AvgPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let batchnormalization8 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d15 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization9 = BatchNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d17 = Conv2dConfig::new([160, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization10 = BatchNormConfig::new(192)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d19 = Conv2dConfig::new([192, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization11 = BatchNormConfig::new(224)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d21 = Conv2dConfig::new([224, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization12 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d23 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization13 = BatchNormConfig::new(288)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d25 = Conv2dConfig::new([288, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization14 = BatchNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d27 = Conv2dConfig::new([320, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization15 = BatchNormConfig::new(352)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d29 = Conv2dConfig::new([352, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization16 = BatchNormConfig::new(384)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d31 = Conv2dConfig::new([384, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization17 = BatchNormConfig::new(416)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d33 = Conv2dConfig::new([416, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization18 = BatchNormConfig::new(448)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d35 = Conv2dConfig::new([448, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization19 = BatchNormConfig::new(480)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d37 = Conv2dConfig::new([480, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization20 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d39 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool2d2 = AvgPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let batchnormalization21 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d40 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization22 = BatchNormConfig::new(288)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d42 = Conv2dConfig::new([288, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization23 = BatchNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d44 = Conv2dConfig::new([320, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization24 = BatchNormConfig::new(352)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d46 = Conv2dConfig::new([352, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization25 = BatchNormConfig::new(384)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d48 = Conv2dConfig::new([384, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization26 = BatchNormConfig::new(416)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d50 = Conv2dConfig::new([416, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization27 = BatchNormConfig::new(448)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d52 = Conv2dConfig::new([448, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization28 = BatchNormConfig::new(480)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d54 = Conv2dConfig::new([480, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization29 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d56 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization30 = BatchNormConfig::new(544)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d58 = Conv2dConfig::new([544, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization31 = BatchNormConfig::new(576)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d60 = Conv2dConfig::new([576, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization32 = BatchNormConfig::new(608)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d62 = Conv2dConfig::new([608, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization33 = BatchNormConfig::new(640)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d64 = Conv2dConfig::new([640, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization34 = BatchNormConfig::new(672)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d66 = Conv2dConfig::new([672, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization35 = BatchNormConfig::new(704)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d68 = Conv2dConfig::new([704, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization36 = BatchNormConfig::new(736)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d70 = Conv2dConfig::new([736, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization37 = BatchNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d72 = Conv2dConfig::new([768, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization38 = BatchNormConfig::new(800)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d74 = Conv2dConfig::new([800, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d75 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization39 = BatchNormConfig::new(832)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d76 = Conv2dConfig::new([832, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization40 = BatchNormConfig::new(864)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d78 = Conv2dConfig::new([864, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization41 = BatchNormConfig::new(896)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d80 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization42 = BatchNormConfig::new(928)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d82 = Conv2dConfig::new([928, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization43 = BatchNormConfig::new(960)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d84 = Conv2dConfig::new([960, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d85 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization44 = BatchNormConfig::new(992)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d86 = Conv2dConfig::new([992, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d87 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization45 = BatchNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d88 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool2d3 = AvgPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let batchnormalization46 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d89 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d90 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization47 = BatchNormConfig::new(544)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d91 = Conv2dConfig::new([544, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization48 = BatchNormConfig::new(576)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d93 = Conv2dConfig::new([576, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d94 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization49 = BatchNormConfig::new(608)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d95 = Conv2dConfig::new([608, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d96 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization50 = BatchNormConfig::new(640)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d97 = Conv2dConfig::new([640, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d98 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization51 = BatchNormConfig::new(672)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d99 = Conv2dConfig::new([672, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d100 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization52 = BatchNormConfig::new(704)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d101 = Conv2dConfig::new([704, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d102 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization53 = BatchNormConfig::new(736)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d103 = Conv2dConfig::new([736, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d104 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization54 = BatchNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d105 = Conv2dConfig::new([768, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d106 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization55 = BatchNormConfig::new(800)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d107 = Conv2dConfig::new([800, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d108 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization56 = BatchNormConfig::new(832)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d109 = Conv2dConfig::new([832, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d110 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization57 = BatchNormConfig::new(864)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d111 = Conv2dConfig::new([864, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d112 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization58 = BatchNormConfig::new(896)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d113 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d114 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization59 = BatchNormConfig::new(928)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d115 = Conv2dConfig::new([928, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d116 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization60 = BatchNormConfig::new(960)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d117 = Conv2dConfig::new([960, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d118 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization61 = BatchNormConfig::new(992)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d119 = Conv2dConfig::new([992, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d120 = Conv2dConfig::new([128, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization62 = BatchNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let gemm1 = LinearConfig::new(1024, 1000).with_bias(true).init(device);
        Self {
            conv2d1,
            maxpool2d1,
            batchnormalization1,
            conv2d2,
            conv2d3,
            batchnormalization2,
            conv2d4,
            conv2d5,
            batchnormalization3,
            conv2d6,
            conv2d7,
            batchnormalization4,
            conv2d8,
            conv2d9,
            batchnormalization5,
            conv2d10,
            conv2d11,
            batchnormalization6,
            conv2d12,
            conv2d13,
            batchnormalization7,
            conv2d14,
            averagepool2d1,
            batchnormalization8,
            conv2d15,
            conv2d16,
            batchnormalization9,
            conv2d17,
            conv2d18,
            batchnormalization10,
            conv2d19,
            conv2d20,
            batchnormalization11,
            conv2d21,
            conv2d22,
            batchnormalization12,
            conv2d23,
            conv2d24,
            batchnormalization13,
            conv2d25,
            conv2d26,
            batchnormalization14,
            conv2d27,
            conv2d28,
            batchnormalization15,
            conv2d29,
            conv2d30,
            batchnormalization16,
            conv2d31,
            conv2d32,
            batchnormalization17,
            conv2d33,
            conv2d34,
            batchnormalization18,
            conv2d35,
            conv2d36,
            batchnormalization19,
            conv2d37,
            conv2d38,
            batchnormalization20,
            conv2d39,
            averagepool2d2,
            batchnormalization21,
            conv2d40,
            conv2d41,
            batchnormalization22,
            conv2d42,
            conv2d43,
            batchnormalization23,
            conv2d44,
            conv2d45,
            batchnormalization24,
            conv2d46,
            conv2d47,
            batchnormalization25,
            conv2d48,
            conv2d49,
            batchnormalization26,
            conv2d50,
            conv2d51,
            batchnormalization27,
            conv2d52,
            conv2d53,
            batchnormalization28,
            conv2d54,
            conv2d55,
            batchnormalization29,
            conv2d56,
            conv2d57,
            batchnormalization30,
            conv2d58,
            conv2d59,
            batchnormalization31,
            conv2d60,
            conv2d61,
            batchnormalization32,
            conv2d62,
            conv2d63,
            batchnormalization33,
            conv2d64,
            conv2d65,
            batchnormalization34,
            conv2d66,
            conv2d67,
            batchnormalization35,
            conv2d68,
            conv2d69,
            batchnormalization36,
            conv2d70,
            conv2d71,
            batchnormalization37,
            conv2d72,
            conv2d73,
            batchnormalization38,
            conv2d74,
            conv2d75,
            batchnormalization39,
            conv2d76,
            conv2d77,
            batchnormalization40,
            conv2d78,
            conv2d79,
            batchnormalization41,
            conv2d80,
            conv2d81,
            batchnormalization42,
            conv2d82,
            conv2d83,
            batchnormalization43,
            conv2d84,
            conv2d85,
            batchnormalization44,
            conv2d86,
            conv2d87,
            batchnormalization45,
            conv2d88,
            averagepool2d3,
            batchnormalization46,
            conv2d89,
            conv2d90,
            batchnormalization47,
            conv2d91,
            conv2d92,
            batchnormalization48,
            conv2d93,
            conv2d94,
            batchnormalization49,
            conv2d95,
            conv2d96,
            batchnormalization50,
            conv2d97,
            conv2d98,
            batchnormalization51,
            conv2d99,
            conv2d100,
            batchnormalization52,
            conv2d101,
            conv2d102,
            batchnormalization53,
            conv2d103,
            conv2d104,
            batchnormalization54,
            conv2d105,
            conv2d106,
            batchnormalization55,
            conv2d107,
            conv2d108,
            batchnormalization56,
            conv2d109,
            conv2d110,
            batchnormalization57,
            conv2d111,
            conv2d112,
            batchnormalization58,
            conv2d113,
            conv2d114,
            batchnormalization59,
            conv2d115,
            conv2d116,
            batchnormalization60,
            conv2d117,
            conv2d118,
            batchnormalization61,
            conv2d119,
            conv2d120,
            batchnormalization62,
            globalaveragepool1,
            gemm1,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu1_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([maxpool2d1_out1.clone()].into(), 1);
        let batchnormalization1_out1 = self.batchnormalization1.forward(concat1_out1);
        let relu2_out1 = burn::tensor::activation::relu(batchnormalization1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu3_out1);
        let concat2_out1 =
            burn::tensor::Tensor::cat([maxpool2d1_out1.clone(), conv2d3_out1.clone()].into(), 1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(concat2_out1);
        let relu4_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu5_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [
                maxpool2d1_out1.clone(),
                conv2d3_out1.clone(),
                conv2d5_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization3_out1 = self.batchnormalization3.forward(concat3_out1);
        let relu6_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu7_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [
                maxpool2d1_out1.clone(),
                conv2d3_out1.clone(),
                conv2d5_out1.clone(),
                conv2d7_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization4_out1 = self.batchnormalization4.forward(concat4_out1);
        let relu8_out1 = burn::tensor::activation::relu(batchnormalization4_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu9_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [
                maxpool2d1_out1.clone(),
                conv2d3_out1.clone(),
                conv2d5_out1.clone(),
                conv2d7_out1.clone(),
                conv2d9_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization5_out1 = self.batchnormalization5.forward(concat5_out1);
        let relu10_out1 = burn::tensor::activation::relu(batchnormalization5_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu11_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [
                maxpool2d1_out1.clone(),
                conv2d3_out1.clone(),
                conv2d5_out1.clone(),
                conv2d7_out1.clone(),
                conv2d9_out1.clone(),
                conv2d11_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization6_out1 = self.batchnormalization6.forward(concat6_out1);
        let relu12_out1 = burn::tensor::activation::relu(batchnormalization6_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu13_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [
                maxpool2d1_out1,
                conv2d3_out1,
                conv2d5_out1,
                conv2d7_out1,
                conv2d9_out1,
                conv2d11_out1,
                conv2d13_out1,
            ]
            .into(),
            1,
        );
        let batchnormalization7_out1 = self.batchnormalization7.forward(concat7_out1);
        let relu14_out1 = burn::tensor::activation::relu(batchnormalization7_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu14_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(conv2d14_out1);
        let concat8_out1 = burn::tensor::Tensor::cat([averagepool2d1_out1.clone()].into(), 1);
        let batchnormalization8_out1 = self.batchnormalization8.forward(concat8_out1);
        let relu15_out1 = burn::tensor::activation::relu(batchnormalization8_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu16_out1);
        let concat9_out1 = burn::tensor::Tensor::cat(
            [averagepool2d1_out1.clone(), conv2d16_out1.clone()].into(),
            1,
        );
        let batchnormalization9_out1 = self.batchnormalization9.forward(concat9_out1);
        let relu17_out1 = burn::tensor::activation::relu(batchnormalization9_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu17_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu18_out1);
        let concat10_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization10_out1 = self.batchnormalization10.forward(concat10_out1);
        let relu19_out1 = burn::tensor::activation::relu(batchnormalization10_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu19_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu20_out1);
        let concat11_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization11_out1 = self.batchnormalization11.forward(concat11_out1);
        let relu21_out1 = burn::tensor::activation::relu(batchnormalization11_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu21_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu22_out1);
        let concat12_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization12_out1 = self.batchnormalization12.forward(concat12_out1);
        let relu23_out1 = burn::tensor::activation::relu(batchnormalization12_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu23_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu24_out1);
        let concat13_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
                conv2d24_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization13_out1 = self.batchnormalization13.forward(concat13_out1);
        let relu25_out1 = burn::tensor::activation::relu(batchnormalization13_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu26_out1);
        let concat14_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
                conv2d24_out1.clone(),
                conv2d26_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization14_out1 = self.batchnormalization14.forward(concat14_out1);
        let relu27_out1 = burn::tensor::activation::relu(batchnormalization14_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu27_out1);
        let relu28_out1 = burn::tensor::activation::relu(conv2d27_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu28_out1);
        let concat15_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
                conv2d24_out1.clone(),
                conv2d26_out1.clone(),
                conv2d28_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization15_out1 = self.batchnormalization15.forward(concat15_out1);
        let relu29_out1 = burn::tensor::activation::relu(batchnormalization15_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu29_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu30_out1);
        let concat16_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
                conv2d24_out1.clone(),
                conv2d26_out1.clone(),
                conv2d28_out1.clone(),
                conv2d30_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization16_out1 = self.batchnormalization16.forward(concat16_out1);
        let relu31_out1 = burn::tensor::activation::relu(batchnormalization16_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu31_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu32_out1);
        let concat17_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
                conv2d24_out1.clone(),
                conv2d26_out1.clone(),
                conv2d28_out1.clone(),
                conv2d30_out1.clone(),
                conv2d32_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization17_out1 = self.batchnormalization17.forward(concat17_out1);
        let relu33_out1 = burn::tensor::activation::relu(batchnormalization17_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu33_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu34_out1);
        let concat18_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
                conv2d24_out1.clone(),
                conv2d26_out1.clone(),
                conv2d28_out1.clone(),
                conv2d30_out1.clone(),
                conv2d32_out1.clone(),
                conv2d34_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization18_out1 = self.batchnormalization18.forward(concat18_out1);
        let relu35_out1 = burn::tensor::activation::relu(batchnormalization18_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu35_out1);
        let relu36_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu36_out1);
        let concat19_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1.clone(),
                conv2d16_out1.clone(),
                conv2d18_out1.clone(),
                conv2d20_out1.clone(),
                conv2d22_out1.clone(),
                conv2d24_out1.clone(),
                conv2d26_out1.clone(),
                conv2d28_out1.clone(),
                conv2d30_out1.clone(),
                conv2d32_out1.clone(),
                conv2d34_out1.clone(),
                conv2d36_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization19_out1 = self.batchnormalization19.forward(concat19_out1);
        let relu37_out1 = burn::tensor::activation::relu(batchnormalization19_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu37_out1);
        let relu38_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu38_out1);
        let concat20_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d1_out1,
                conv2d16_out1,
                conv2d18_out1,
                conv2d20_out1,
                conv2d22_out1,
                conv2d24_out1,
                conv2d26_out1,
                conv2d28_out1,
                conv2d30_out1,
                conv2d32_out1,
                conv2d34_out1,
                conv2d36_out1,
                conv2d38_out1,
            ]
            .into(),
            1,
        );
        let batchnormalization20_out1 = self.batchnormalization20.forward(concat20_out1);
        let relu39_out1 = burn::tensor::activation::relu(batchnormalization20_out1);
        let conv2d39_out1 = self.conv2d39.forward(relu39_out1);
        let averagepool2d2_out1 = self.averagepool2d2.forward(conv2d39_out1);
        let concat21_out1 = burn::tensor::Tensor::cat([averagepool2d2_out1.clone()].into(), 1);
        let batchnormalization21_out1 = self.batchnormalization21.forward(concat21_out1);
        let relu40_out1 = burn::tensor::activation::relu(batchnormalization21_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu40_out1);
        let relu41_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let conv2d41_out1 = self.conv2d41.forward(relu41_out1);
        let concat22_out1 = burn::tensor::Tensor::cat(
            [averagepool2d2_out1.clone(), conv2d41_out1.clone()].into(),
            1,
        );
        let batchnormalization22_out1 = self.batchnormalization22.forward(concat22_out1);
        let relu42_out1 = burn::tensor::activation::relu(batchnormalization22_out1);
        let conv2d42_out1 = self.conv2d42.forward(relu42_out1);
        let relu43_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu43_out1);
        let concat23_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization23_out1 = self.batchnormalization23.forward(concat23_out1);
        let relu44_out1 = burn::tensor::activation::relu(batchnormalization23_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = self.conv2d45.forward(relu45_out1);
        let concat24_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization24_out1 = self.batchnormalization24.forward(concat24_out1);
        let relu46_out1 = burn::tensor::activation::relu(batchnormalization24_out1);
        let conv2d46_out1 = self.conv2d46.forward(relu46_out1);
        let relu47_out1 = burn::tensor::activation::relu(conv2d46_out1);
        let conv2d47_out1 = self.conv2d47.forward(relu47_out1);
        let concat25_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization25_out1 = self.batchnormalization25.forward(concat25_out1);
        let relu48_out1 = burn::tensor::activation::relu(batchnormalization25_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu48_out1);
        let relu49_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(relu49_out1);
        let concat26_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization26_out1 = self.batchnormalization26.forward(concat26_out1);
        let relu50_out1 = burn::tensor::activation::relu(batchnormalization26_out1);
        let conv2d50_out1 = self.conv2d50.forward(relu50_out1);
        let relu51_out1 = burn::tensor::activation::relu(conv2d50_out1);
        let conv2d51_out1 = self.conv2d51.forward(relu51_out1);
        let concat27_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization27_out1 = self.batchnormalization27.forward(concat27_out1);
        let relu52_out1 = burn::tensor::activation::relu(batchnormalization27_out1);
        let conv2d52_out1 = self.conv2d52.forward(relu52_out1);
        let relu53_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let conv2d53_out1 = self.conv2d53.forward(relu53_out1);
        let concat28_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization28_out1 = self.batchnormalization28.forward(concat28_out1);
        let relu54_out1 = burn::tensor::activation::relu(batchnormalization28_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu54_out1);
        let relu55_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let conv2d55_out1 = self.conv2d55.forward(relu55_out1);
        let concat29_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization29_out1 = self.batchnormalization29.forward(concat29_out1);
        let relu56_out1 = burn::tensor::activation::relu(batchnormalization29_out1);
        let conv2d56_out1 = self.conv2d56.forward(relu56_out1);
        let relu57_out1 = burn::tensor::activation::relu(conv2d56_out1);
        let conv2d57_out1 = self.conv2d57.forward(relu57_out1);
        let concat30_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization30_out1 = self.batchnormalization30.forward(concat30_out1);
        let relu58_out1 = burn::tensor::activation::relu(batchnormalization30_out1);
        let conv2d58_out1 = self.conv2d58.forward(relu58_out1);
        let relu59_out1 = burn::tensor::activation::relu(conv2d58_out1);
        let conv2d59_out1 = self.conv2d59.forward(relu59_out1);
        let concat31_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization31_out1 = self.batchnormalization31.forward(concat31_out1);
        let relu60_out1 = burn::tensor::activation::relu(batchnormalization31_out1);
        let conv2d60_out1 = self.conv2d60.forward(relu60_out1);
        let relu61_out1 = burn::tensor::activation::relu(conv2d60_out1);
        let conv2d61_out1 = self.conv2d61.forward(relu61_out1);
        let concat32_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization32_out1 = self.batchnormalization32.forward(concat32_out1);
        let relu62_out1 = burn::tensor::activation::relu(batchnormalization32_out1);
        let conv2d62_out1 = self.conv2d62.forward(relu62_out1);
        let relu63_out1 = burn::tensor::activation::relu(conv2d62_out1);
        let conv2d63_out1 = self.conv2d63.forward(relu63_out1);
        let concat33_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization33_out1 = self.batchnormalization33.forward(concat33_out1);
        let relu64_out1 = burn::tensor::activation::relu(batchnormalization33_out1);
        let conv2d64_out1 = self.conv2d64.forward(relu64_out1);
        let relu65_out1 = burn::tensor::activation::relu(conv2d64_out1);
        let conv2d65_out1 = self.conv2d65.forward(relu65_out1);
        let concat34_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization34_out1 = self.batchnormalization34.forward(concat34_out1);
        let relu66_out1 = burn::tensor::activation::relu(batchnormalization34_out1);
        let conv2d66_out1 = self.conv2d66.forward(relu66_out1);
        let relu67_out1 = burn::tensor::activation::relu(conv2d66_out1);
        let conv2d67_out1 = self.conv2d67.forward(relu67_out1);
        let concat35_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization35_out1 = self.batchnormalization35.forward(concat35_out1);
        let relu68_out1 = burn::tensor::activation::relu(batchnormalization35_out1);
        let conv2d68_out1 = self.conv2d68.forward(relu68_out1);
        let relu69_out1 = burn::tensor::activation::relu(conv2d68_out1);
        let conv2d69_out1 = self.conv2d69.forward(relu69_out1);
        let concat36_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization36_out1 = self.batchnormalization36.forward(concat36_out1);
        let relu70_out1 = burn::tensor::activation::relu(batchnormalization36_out1);
        let conv2d70_out1 = self.conv2d70.forward(relu70_out1);
        let relu71_out1 = burn::tensor::activation::relu(conv2d70_out1);
        let conv2d71_out1 = self.conv2d71.forward(relu71_out1);
        let concat37_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization37_out1 = self.batchnormalization37.forward(concat37_out1);
        let relu72_out1 = burn::tensor::activation::relu(batchnormalization37_out1);
        let conv2d72_out1 = self.conv2d72.forward(relu72_out1);
        let relu73_out1 = burn::tensor::activation::relu(conv2d72_out1);
        let conv2d73_out1 = self.conv2d73.forward(relu73_out1);
        let concat38_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
                conv2d73_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization38_out1 = self.batchnormalization38.forward(concat38_out1);
        let relu74_out1 = burn::tensor::activation::relu(batchnormalization38_out1);
        let conv2d74_out1 = self.conv2d74.forward(relu74_out1);
        let relu75_out1 = burn::tensor::activation::relu(conv2d74_out1);
        let conv2d75_out1 = self.conv2d75.forward(relu75_out1);
        let concat39_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
                conv2d73_out1.clone(),
                conv2d75_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization39_out1 = self.batchnormalization39.forward(concat39_out1);
        let relu76_out1 = burn::tensor::activation::relu(batchnormalization39_out1);
        let conv2d76_out1 = self.conv2d76.forward(relu76_out1);
        let relu77_out1 = burn::tensor::activation::relu(conv2d76_out1);
        let conv2d77_out1 = self.conv2d77.forward(relu77_out1);
        let concat40_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
                conv2d73_out1.clone(),
                conv2d75_out1.clone(),
                conv2d77_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization40_out1 = self.batchnormalization40.forward(concat40_out1);
        let relu78_out1 = burn::tensor::activation::relu(batchnormalization40_out1);
        let conv2d78_out1 = self.conv2d78.forward(relu78_out1);
        let relu79_out1 = burn::tensor::activation::relu(conv2d78_out1);
        let conv2d79_out1 = self.conv2d79.forward(relu79_out1);
        let concat41_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
                conv2d73_out1.clone(),
                conv2d75_out1.clone(),
                conv2d77_out1.clone(),
                conv2d79_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization41_out1 = self.batchnormalization41.forward(concat41_out1);
        let relu80_out1 = burn::tensor::activation::relu(batchnormalization41_out1);
        let conv2d80_out1 = self.conv2d80.forward(relu80_out1);
        let relu81_out1 = burn::tensor::activation::relu(conv2d80_out1);
        let conv2d81_out1 = self.conv2d81.forward(relu81_out1);
        let concat42_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
                conv2d73_out1.clone(),
                conv2d75_out1.clone(),
                conv2d77_out1.clone(),
                conv2d79_out1.clone(),
                conv2d81_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization42_out1 = self.batchnormalization42.forward(concat42_out1);
        let relu82_out1 = burn::tensor::activation::relu(batchnormalization42_out1);
        let conv2d82_out1 = self.conv2d82.forward(relu82_out1);
        let relu83_out1 = burn::tensor::activation::relu(conv2d82_out1);
        let conv2d83_out1 = self.conv2d83.forward(relu83_out1);
        let concat43_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
                conv2d73_out1.clone(),
                conv2d75_out1.clone(),
                conv2d77_out1.clone(),
                conv2d79_out1.clone(),
                conv2d81_out1.clone(),
                conv2d83_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization43_out1 = self.batchnormalization43.forward(concat43_out1);
        let relu84_out1 = burn::tensor::activation::relu(batchnormalization43_out1);
        let conv2d84_out1 = self.conv2d84.forward(relu84_out1);
        let relu85_out1 = burn::tensor::activation::relu(conv2d84_out1);
        let conv2d85_out1 = self.conv2d85.forward(relu85_out1);
        let concat44_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1.clone(),
                conv2d41_out1.clone(),
                conv2d43_out1.clone(),
                conv2d45_out1.clone(),
                conv2d47_out1.clone(),
                conv2d49_out1.clone(),
                conv2d51_out1.clone(),
                conv2d53_out1.clone(),
                conv2d55_out1.clone(),
                conv2d57_out1.clone(),
                conv2d59_out1.clone(),
                conv2d61_out1.clone(),
                conv2d63_out1.clone(),
                conv2d65_out1.clone(),
                conv2d67_out1.clone(),
                conv2d69_out1.clone(),
                conv2d71_out1.clone(),
                conv2d73_out1.clone(),
                conv2d75_out1.clone(),
                conv2d77_out1.clone(),
                conv2d79_out1.clone(),
                conv2d81_out1.clone(),
                conv2d83_out1.clone(),
                conv2d85_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization44_out1 = self.batchnormalization44.forward(concat44_out1);
        let relu86_out1 = burn::tensor::activation::relu(batchnormalization44_out1);
        let conv2d86_out1 = self.conv2d86.forward(relu86_out1);
        let relu87_out1 = burn::tensor::activation::relu(conv2d86_out1);
        let conv2d87_out1 = self.conv2d87.forward(relu87_out1);
        let concat45_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d2_out1,
                conv2d41_out1,
                conv2d43_out1,
                conv2d45_out1,
                conv2d47_out1,
                conv2d49_out1,
                conv2d51_out1,
                conv2d53_out1,
                conv2d55_out1,
                conv2d57_out1,
                conv2d59_out1,
                conv2d61_out1,
                conv2d63_out1,
                conv2d65_out1,
                conv2d67_out1,
                conv2d69_out1,
                conv2d71_out1,
                conv2d73_out1,
                conv2d75_out1,
                conv2d77_out1,
                conv2d79_out1,
                conv2d81_out1,
                conv2d83_out1,
                conv2d85_out1,
                conv2d87_out1,
            ]
            .into(),
            1,
        );
        let batchnormalization45_out1 = self.batchnormalization45.forward(concat45_out1);
        let relu88_out1 = burn::tensor::activation::relu(batchnormalization45_out1);
        let conv2d88_out1 = self.conv2d88.forward(relu88_out1);
        let averagepool2d3_out1 = self.averagepool2d3.forward(conv2d88_out1);
        let concat46_out1 = burn::tensor::Tensor::cat([averagepool2d3_out1.clone()].into(), 1);
        let batchnormalization46_out1 = self.batchnormalization46.forward(concat46_out1);
        let relu89_out1 = burn::tensor::activation::relu(batchnormalization46_out1);
        let conv2d89_out1 = self.conv2d89.forward(relu89_out1);
        let relu90_out1 = burn::tensor::activation::relu(conv2d89_out1);
        let conv2d90_out1 = self.conv2d90.forward(relu90_out1);
        let concat47_out1 = burn::tensor::Tensor::cat(
            [averagepool2d3_out1.clone(), conv2d90_out1.clone()].into(),
            1,
        );
        let batchnormalization47_out1 = self.batchnormalization47.forward(concat47_out1);
        let relu91_out1 = burn::tensor::activation::relu(batchnormalization47_out1);
        let conv2d91_out1 = self.conv2d91.forward(relu91_out1);
        let relu92_out1 = burn::tensor::activation::relu(conv2d91_out1);
        let conv2d92_out1 = self.conv2d92.forward(relu92_out1);
        let concat48_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization48_out1 = self.batchnormalization48.forward(concat48_out1);
        let relu93_out1 = burn::tensor::activation::relu(batchnormalization48_out1);
        let conv2d93_out1 = self.conv2d93.forward(relu93_out1);
        let relu94_out1 = burn::tensor::activation::relu(conv2d93_out1);
        let conv2d94_out1 = self.conv2d94.forward(relu94_out1);
        let concat49_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization49_out1 = self.batchnormalization49.forward(concat49_out1);
        let relu95_out1 = burn::tensor::activation::relu(batchnormalization49_out1);
        let conv2d95_out1 = self.conv2d95.forward(relu95_out1);
        let relu96_out1 = burn::tensor::activation::relu(conv2d95_out1);
        let conv2d96_out1 = self.conv2d96.forward(relu96_out1);
        let concat50_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization50_out1 = self.batchnormalization50.forward(concat50_out1);
        let relu97_out1 = burn::tensor::activation::relu(batchnormalization50_out1);
        let conv2d97_out1 = self.conv2d97.forward(relu97_out1);
        let relu98_out1 = burn::tensor::activation::relu(conv2d97_out1);
        let conv2d98_out1 = self.conv2d98.forward(relu98_out1);
        let concat51_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization51_out1 = self.batchnormalization51.forward(concat51_out1);
        let relu99_out1 = burn::tensor::activation::relu(batchnormalization51_out1);
        let conv2d99_out1 = self.conv2d99.forward(relu99_out1);
        let relu100_out1 = burn::tensor::activation::relu(conv2d99_out1);
        let conv2d100_out1 = self.conv2d100.forward(relu100_out1);
        let concat52_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization52_out1 = self.batchnormalization52.forward(concat52_out1);
        let relu101_out1 = burn::tensor::activation::relu(batchnormalization52_out1);
        let conv2d101_out1 = self.conv2d101.forward(relu101_out1);
        let relu102_out1 = burn::tensor::activation::relu(conv2d101_out1);
        let conv2d102_out1 = self.conv2d102.forward(relu102_out1);
        let concat53_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization53_out1 = self.batchnormalization53.forward(concat53_out1);
        let relu103_out1 = burn::tensor::activation::relu(batchnormalization53_out1);
        let conv2d103_out1 = self.conv2d103.forward(relu103_out1);
        let relu104_out1 = burn::tensor::activation::relu(conv2d103_out1);
        let conv2d104_out1 = self.conv2d104.forward(relu104_out1);
        let concat54_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization54_out1 = self.batchnormalization54.forward(concat54_out1);
        let relu105_out1 = burn::tensor::activation::relu(batchnormalization54_out1);
        let conv2d105_out1 = self.conv2d105.forward(relu105_out1);
        let relu106_out1 = burn::tensor::activation::relu(conv2d105_out1);
        let conv2d106_out1 = self.conv2d106.forward(relu106_out1);
        let concat55_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
                conv2d106_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization55_out1 = self.batchnormalization55.forward(concat55_out1);
        let relu107_out1 = burn::tensor::activation::relu(batchnormalization55_out1);
        let conv2d107_out1 = self.conv2d107.forward(relu107_out1);
        let relu108_out1 = burn::tensor::activation::relu(conv2d107_out1);
        let conv2d108_out1 = self.conv2d108.forward(relu108_out1);
        let concat56_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
                conv2d106_out1.clone(),
                conv2d108_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization56_out1 = self.batchnormalization56.forward(concat56_out1);
        let relu109_out1 = burn::tensor::activation::relu(batchnormalization56_out1);
        let conv2d109_out1 = self.conv2d109.forward(relu109_out1);
        let relu110_out1 = burn::tensor::activation::relu(conv2d109_out1);
        let conv2d110_out1 = self.conv2d110.forward(relu110_out1);
        let concat57_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
                conv2d106_out1.clone(),
                conv2d108_out1.clone(),
                conv2d110_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization57_out1 = self.batchnormalization57.forward(concat57_out1);
        let relu111_out1 = burn::tensor::activation::relu(batchnormalization57_out1);
        let conv2d111_out1 = self.conv2d111.forward(relu111_out1);
        let relu112_out1 = burn::tensor::activation::relu(conv2d111_out1);
        let conv2d112_out1 = self.conv2d112.forward(relu112_out1);
        let concat58_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
                conv2d106_out1.clone(),
                conv2d108_out1.clone(),
                conv2d110_out1.clone(),
                conv2d112_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization58_out1 = self.batchnormalization58.forward(concat58_out1);
        let relu113_out1 = burn::tensor::activation::relu(batchnormalization58_out1);
        let conv2d113_out1 = self.conv2d113.forward(relu113_out1);
        let relu114_out1 = burn::tensor::activation::relu(conv2d113_out1);
        let conv2d114_out1 = self.conv2d114.forward(relu114_out1);
        let concat59_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
                conv2d106_out1.clone(),
                conv2d108_out1.clone(),
                conv2d110_out1.clone(),
                conv2d112_out1.clone(),
                conv2d114_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization59_out1 = self.batchnormalization59.forward(concat59_out1);
        let relu115_out1 = burn::tensor::activation::relu(batchnormalization59_out1);
        let conv2d115_out1 = self.conv2d115.forward(relu115_out1);
        let relu116_out1 = burn::tensor::activation::relu(conv2d115_out1);
        let conv2d116_out1 = self.conv2d116.forward(relu116_out1);
        let concat60_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
                conv2d106_out1.clone(),
                conv2d108_out1.clone(),
                conv2d110_out1.clone(),
                conv2d112_out1.clone(),
                conv2d114_out1.clone(),
                conv2d116_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization60_out1 = self.batchnormalization60.forward(concat60_out1);
        let relu117_out1 = burn::tensor::activation::relu(batchnormalization60_out1);
        let conv2d117_out1 = self.conv2d117.forward(relu117_out1);
        let relu118_out1 = burn::tensor::activation::relu(conv2d117_out1);
        let conv2d118_out1 = self.conv2d118.forward(relu118_out1);
        let concat61_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1.clone(),
                conv2d90_out1.clone(),
                conv2d92_out1.clone(),
                conv2d94_out1.clone(),
                conv2d96_out1.clone(),
                conv2d98_out1.clone(),
                conv2d100_out1.clone(),
                conv2d102_out1.clone(),
                conv2d104_out1.clone(),
                conv2d106_out1.clone(),
                conv2d108_out1.clone(),
                conv2d110_out1.clone(),
                conv2d112_out1.clone(),
                conv2d114_out1.clone(),
                conv2d116_out1.clone(),
                conv2d118_out1.clone(),
            ]
            .into(),
            1,
        );
        let batchnormalization61_out1 = self.batchnormalization61.forward(concat61_out1);
        let relu119_out1 = burn::tensor::activation::relu(batchnormalization61_out1);
        let conv2d119_out1 = self.conv2d119.forward(relu119_out1);
        let relu120_out1 = burn::tensor::activation::relu(conv2d119_out1);
        let conv2d120_out1 = self.conv2d120.forward(relu120_out1);
        let concat62_out1 = burn::tensor::Tensor::cat(
            [
                averagepool2d3_out1,
                conv2d90_out1,
                conv2d92_out1,
                conv2d94_out1,
                conv2d96_out1,
                conv2d98_out1,
                conv2d100_out1,
                conv2d102_out1,
                conv2d104_out1,
                conv2d106_out1,
                conv2d108_out1,
                conv2d110_out1,
                conv2d112_out1,
                conv2d114_out1,
                conv2d116_out1,
                conv2d118_out1,
                conv2d120_out1,
            ]
            .into(),
            1,
        );
        let batchnormalization62_out1 = self.batchnormalization62.forward(concat62_out1);
        let relu121_out1 = burn::tensor::activation::relu(batchnormalization62_out1);
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(relu121_out1);
        let flatten1_out1 = globalaveragepool1_out1.flatten(1, 3);
        let gemm1_out1 = self.gemm1.forward(flatten1_out1);
        gemm1_out1
    }
}
