// Generated from ONNX "../../models/onnx_dir/inceptionv3.onnx" by burn-import
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    averagepool2d1: AvgPool2d,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    averagepool2d2: AvgPool2d,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    averagepool2d3: AvgPool2d,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    averagepool2d4: AvgPool2d,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    averagepool2d5: AvgPool2d,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    conv2d58: Conv2d<B>,
    conv2d59: Conv2d<B>,
    averagepool2d6: AvgPool2d,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    conv2d62: Conv2d<B>,
    conv2d63: Conv2d<B>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    averagepool2d7: AvgPool2d,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    conv2d72: Conv2d<B>,
    conv2d73: Conv2d<B>,
    conv2d74: Conv2d<B>,
    conv2d75: Conv2d<B>,
    conv2d76: Conv2d<B>,
    maxpool2d4: MaxPool2d,
    conv2d77: Conv2d<B>,
    conv2d78: Conv2d<B>,
    conv2d79: Conv2d<B>,
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    conv2d82: Conv2d<B>,
    conv2d83: Conv2d<B>,
    conv2d84: Conv2d<B>,
    averagepool2d8: AvgPool2d,
    conv2d85: Conv2d<B>,
    conv2d86: Conv2d<B>,
    conv2d87: Conv2d<B>,
    conv2d88: Conv2d<B>,
    conv2d89: Conv2d<B>,
    conv2d90: Conv2d<B>,
    conv2d91: Conv2d<B>,
    conv2d92: Conv2d<B>,
    conv2d93: Conv2d<B>,
    averagepool2d9: AvgPool2d,
    conv2d94: Conv2d<B>,
    globalaveragepool1: AdaptiveAvgPool2d,
    gemm1: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("../../models/onnx_dir/inceptionv3", &Default::default())
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
        let conv2d1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d4 = Conv2dConfig::new([64, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([80, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d6 = Conv2dConfig::new([192, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([192, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([48, 64], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([192, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([64, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d1 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d12 = Conv2dConfig::new([192, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([256, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([48, 64], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([64, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d2 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d19 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([288, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([288, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([48, 64], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([288, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([64, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d3 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d26 = Conv2dConfig::new([288, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([288, 384], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([288, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([64, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d31 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([768, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([128, 192], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([768, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([128, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d4 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d40 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([768, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([160, 160], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([160, 192], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([768, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([160, 160], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([160, 160], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([160, 160], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([160, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d5 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d50 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([768, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([160, 160], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([160, 192], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([768, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([160, 160], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([160, 160], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([160, 160], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([160, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d6 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d60 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([192, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([192, 192], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([192, 192], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([192, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([192, 192], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([192, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d7 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d70 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([192, 320], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d74 = Conv2dConfig::new([192, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d75 = Conv2dConfig::new([192, 192], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d4 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d77 = Conv2dConfig::new([1280, 320], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d78 = Conv2dConfig::new([1280, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([384, 384], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d80 = Conv2dConfig::new([384, 384], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([1280, 448], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d82 = Conv2dConfig::new([448, 384], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([384, 384], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d84 = Conv2dConfig::new([384, 384], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d8 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d85 = Conv2dConfig::new([1280, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d86 = Conv2dConfig::new([2048, 320], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d87 = Conv2dConfig::new([2048, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d88 = Conv2dConfig::new([384, 384], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d89 = Conv2dConfig::new([384, 384], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d90 = Conv2dConfig::new([2048, 448], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d91 = Conv2dConfig::new([448, 384], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([384, 384], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d93 = Conv2dConfig::new([384, 384], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d9 = AvgPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_count_include_pad(true)
            .init();
        let conv2d94 = Conv2dConfig::new([2048, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let gemm1 = LinearConfig::new(2048, 1000).with_bias(true).init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            maxpool2d1,
            conv2d4,
            conv2d5,
            maxpool2d2,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            averagepool2d1,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            averagepool2d2,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            averagepool2d3,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            maxpool2d3,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            averagepool2d4,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            averagepool2d5,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            averagepool2d6,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            averagepool2d7,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            conv2d74,
            conv2d75,
            conv2d76,
            maxpool2d4,
            conv2d77,
            conv2d78,
            conv2d79,
            conv2d80,
            conv2d81,
            conv2d82,
            conv2d83,
            conv2d84,
            averagepool2d8,
            conv2d85,
            conv2d86,
            conv2d87,
            conv2d88,
            conv2d89,
            conv2d90,
            conv2d91,
            conv2d92,
            conv2d93,
            averagepool2d9,
            conv2d94,
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
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu3_out1);
        let conv2d4_out1 = self.conv2d4.forward(maxpool2d1_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu5_out1);
        let conv2d6_out1 = self.conv2d6.forward(maxpool2d2_out1.clone());
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(maxpool2d2_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(maxpool2d2_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(maxpool2d2_out1);
        let conv2d12_out1 = self.conv2d12.forward(averagepool2d1_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let concat1_out1 =
            burn::tensor::Tensor::cat([relu6_out1, relu8_out1, relu11_out1, relu12_out1].into(), 1);
        let conv2d13_out1 = self.conv2d13.forward(concat1_out1.clone());
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(concat1_out1.clone());
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu14_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(concat1_out1.clone());
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu16_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let averagepool2d2_out1 = self.averagepool2d2.forward(concat1_out1);
        let conv2d19_out1 = self.conv2d19.forward(averagepool2d2_out1);
        let relu19_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [relu13_out1, relu15_out1, relu18_out1, relu19_out1].into(),
            1,
        );
        let conv2d20_out1 = self.conv2d20.forward(concat2_out1.clone());
        let relu20_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = self.conv2d21.forward(concat2_out1.clone());
        let relu21_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu21_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv2d22_out1);
        let conv2d23_out1 = self.conv2d23.forward(concat2_out1.clone());
        let relu23_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu23_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu24_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let averagepool2d3_out1 = self.averagepool2d3.forward(concat2_out1);
        let conv2d26_out1 = self.conv2d26.forward(averagepool2d3_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [relu20_out1, relu22_out1, relu25_out1, relu26_out1].into(),
            1,
        );
        let conv2d27_out1 = self.conv2d27.forward(concat3_out1.clone());
        let relu27_out1 = burn::tensor::activation::relu(conv2d27_out1);
        let conv2d28_out1 = self.conv2d28.forward(concat3_out1.clone());
        let relu28_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu28_out1);
        let relu29_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu29_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(concat3_out1);
        let concat4_out1 =
            burn::tensor::Tensor::cat([relu27_out1, relu30_out1, maxpool2d3_out1].into(), 1);
        let conv2d31_out1 = self.conv2d31.forward(concat4_out1.clone());
        let relu31_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = self.conv2d32.forward(concat4_out1.clone());
        let relu32_out1 = burn::tensor::activation::relu(conv2d32_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu32_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu33_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv2d34_out1);
        let conv2d35_out1 = self.conv2d35.forward(concat4_out1.clone());
        let relu35_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu35_out1);
        let relu36_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu37_out1);
        let relu38_out1 = burn::tensor::activation::relu(conv2d38_out1);
        let conv2d39_out1 = self.conv2d39.forward(relu38_out1);
        let relu39_out1 = burn::tensor::activation::relu(conv2d39_out1);
        let averagepool2d4_out1 = self.averagepool2d4.forward(concat4_out1);
        let conv2d40_out1 = self.conv2d40.forward(averagepool2d4_out1);
        let relu40_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [relu31_out1, relu34_out1, relu39_out1, relu40_out1].into(),
            1,
        );
        let conv2d41_out1 = self.conv2d41.forward(concat5_out1.clone());
        let relu41_out1 = burn::tensor::activation::relu(conv2d41_out1);
        let conv2d42_out1 = self.conv2d42.forward(concat5_out1.clone());
        let relu42_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu42_out1);
        let relu43_out1 = burn::tensor::activation::relu(conv2d43_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu43_out1);
        let relu44_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = self.conv2d45.forward(concat5_out1.clone());
        let relu45_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let conv2d46_out1 = self.conv2d46.forward(relu45_out1);
        let relu46_out1 = burn::tensor::activation::relu(conv2d46_out1);
        let conv2d47_out1 = self.conv2d47.forward(relu46_out1);
        let relu47_out1 = burn::tensor::activation::relu(conv2d47_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu47_out1);
        let relu48_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(relu48_out1);
        let relu49_out1 = burn::tensor::activation::relu(conv2d49_out1);
        let averagepool2d5_out1 = self.averagepool2d5.forward(concat5_out1);
        let conv2d50_out1 = self.conv2d50.forward(averagepool2d5_out1);
        let relu50_out1 = burn::tensor::activation::relu(conv2d50_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [relu41_out1, relu44_out1, relu49_out1, relu50_out1].into(),
            1,
        );
        let conv2d51_out1 = self.conv2d51.forward(concat6_out1.clone());
        let relu51_out1 = burn::tensor::activation::relu(conv2d51_out1);
        let conv2d52_out1 = self.conv2d52.forward(concat6_out1.clone());
        let relu52_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let conv2d53_out1 = self.conv2d53.forward(relu52_out1);
        let relu53_out1 = burn::tensor::activation::relu(conv2d53_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu53_out1);
        let relu54_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let conv2d55_out1 = self.conv2d55.forward(concat6_out1.clone());
        let relu55_out1 = burn::tensor::activation::relu(conv2d55_out1);
        let conv2d56_out1 = self.conv2d56.forward(relu55_out1);
        let relu56_out1 = burn::tensor::activation::relu(conv2d56_out1);
        let conv2d57_out1 = self.conv2d57.forward(relu56_out1);
        let relu57_out1 = burn::tensor::activation::relu(conv2d57_out1);
        let conv2d58_out1 = self.conv2d58.forward(relu57_out1);
        let relu58_out1 = burn::tensor::activation::relu(conv2d58_out1);
        let conv2d59_out1 = self.conv2d59.forward(relu58_out1);
        let relu59_out1 = burn::tensor::activation::relu(conv2d59_out1);
        let averagepool2d6_out1 = self.averagepool2d6.forward(concat6_out1);
        let conv2d60_out1 = self.conv2d60.forward(averagepool2d6_out1);
        let relu60_out1 = burn::tensor::activation::relu(conv2d60_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [relu51_out1, relu54_out1, relu59_out1, relu60_out1].into(),
            1,
        );
        let conv2d61_out1 = self.conv2d61.forward(concat7_out1.clone());
        let relu61_out1 = burn::tensor::activation::relu(conv2d61_out1);
        let conv2d62_out1 = self.conv2d62.forward(concat7_out1.clone());
        let relu62_out1 = burn::tensor::activation::relu(conv2d62_out1);
        let conv2d63_out1 = self.conv2d63.forward(relu62_out1);
        let relu63_out1 = burn::tensor::activation::relu(conv2d63_out1);
        let conv2d64_out1 = self.conv2d64.forward(relu63_out1);
        let relu64_out1 = burn::tensor::activation::relu(conv2d64_out1);
        let conv2d65_out1 = self.conv2d65.forward(concat7_out1.clone());
        let relu65_out1 = burn::tensor::activation::relu(conv2d65_out1);
        let conv2d66_out1 = self.conv2d66.forward(relu65_out1);
        let relu66_out1 = burn::tensor::activation::relu(conv2d66_out1);
        let conv2d67_out1 = self.conv2d67.forward(relu66_out1);
        let relu67_out1 = burn::tensor::activation::relu(conv2d67_out1);
        let conv2d68_out1 = self.conv2d68.forward(relu67_out1);
        let relu68_out1 = burn::tensor::activation::relu(conv2d68_out1);
        let conv2d69_out1 = self.conv2d69.forward(relu68_out1);
        let relu69_out1 = burn::tensor::activation::relu(conv2d69_out1);
        let averagepool2d7_out1 = self.averagepool2d7.forward(concat7_out1);
        let conv2d70_out1 = self.conv2d70.forward(averagepool2d7_out1);
        let relu70_out1 = burn::tensor::activation::relu(conv2d70_out1);
        let concat8_out1 = burn::tensor::Tensor::cat(
            [relu61_out1, relu64_out1, relu69_out1, relu70_out1].into(),
            1,
        );
        let conv2d71_out1 = self.conv2d71.forward(concat8_out1.clone());
        let relu71_out1 = burn::tensor::activation::relu(conv2d71_out1);
        let conv2d72_out1 = self.conv2d72.forward(relu71_out1);
        let relu72_out1 = burn::tensor::activation::relu(conv2d72_out1);
        let conv2d73_out1 = self.conv2d73.forward(concat8_out1.clone());
        let relu73_out1 = burn::tensor::activation::relu(conv2d73_out1);
        let conv2d74_out1 = self.conv2d74.forward(relu73_out1);
        let relu74_out1 = burn::tensor::activation::relu(conv2d74_out1);
        let conv2d75_out1 = self.conv2d75.forward(relu74_out1);
        let relu75_out1 = burn::tensor::activation::relu(conv2d75_out1);
        let conv2d76_out1 = self.conv2d76.forward(relu75_out1);
        let relu76_out1 = burn::tensor::activation::relu(conv2d76_out1);
        let maxpool2d4_out1 = self.maxpool2d4.forward(concat8_out1);
        let concat9_out1 =
            burn::tensor::Tensor::cat([relu72_out1, relu76_out1, maxpool2d4_out1].into(), 1);
        let conv2d77_out1 = self.conv2d77.forward(concat9_out1.clone());
        let relu77_out1 = burn::tensor::activation::relu(conv2d77_out1);
        let conv2d78_out1 = self.conv2d78.forward(concat9_out1.clone());
        let relu78_out1 = burn::tensor::activation::relu(conv2d78_out1);
        let conv2d79_out1 = self.conv2d79.forward(relu78_out1.clone());
        let relu79_out1 = burn::tensor::activation::relu(conv2d79_out1);
        let conv2d80_out1 = self.conv2d80.forward(relu78_out1);
        let relu80_out1 = burn::tensor::activation::relu(conv2d80_out1);
        let conv2d81_out1 = self.conv2d81.forward(concat9_out1.clone());
        let relu81_out1 = burn::tensor::activation::relu(conv2d81_out1);
        let conv2d82_out1 = self.conv2d82.forward(relu81_out1);
        let relu82_out1 = burn::tensor::activation::relu(conv2d82_out1);
        let conv2d83_out1 = self.conv2d83.forward(relu82_out1.clone());
        let relu83_out1 = burn::tensor::activation::relu(conv2d83_out1);
        let conv2d84_out1 = self.conv2d84.forward(relu82_out1);
        let relu84_out1 = burn::tensor::activation::relu(conv2d84_out1);
        let averagepool2d8_out1 = self.averagepool2d8.forward(concat9_out1);
        let conv2d85_out1 = self.conv2d85.forward(averagepool2d8_out1);
        let relu85_out1 = burn::tensor::activation::relu(conv2d85_out1);
        let concat10_out1 = burn::tensor::Tensor::cat(
            [
                relu77_out1,
                relu79_out1,
                relu80_out1,
                relu83_out1,
                relu84_out1,
                relu85_out1,
            ]
            .into(),
            1,
        );
        let conv2d86_out1 = self.conv2d86.forward(concat10_out1.clone());
        let relu86_out1 = burn::tensor::activation::relu(conv2d86_out1);
        let conv2d87_out1 = self.conv2d87.forward(concat10_out1.clone());
        let relu87_out1 = burn::tensor::activation::relu(conv2d87_out1);
        let conv2d88_out1 = self.conv2d88.forward(relu87_out1.clone());
        let relu88_out1 = burn::tensor::activation::relu(conv2d88_out1);
        let conv2d89_out1 = self.conv2d89.forward(relu87_out1);
        let relu89_out1 = burn::tensor::activation::relu(conv2d89_out1);
        let conv2d90_out1 = self.conv2d90.forward(concat10_out1.clone());
        let relu90_out1 = burn::tensor::activation::relu(conv2d90_out1);
        let conv2d91_out1 = self.conv2d91.forward(relu90_out1);
        let relu91_out1 = burn::tensor::activation::relu(conv2d91_out1);
        let conv2d92_out1 = self.conv2d92.forward(relu91_out1.clone());
        let relu92_out1 = burn::tensor::activation::relu(conv2d92_out1);
        let conv2d93_out1 = self.conv2d93.forward(relu91_out1);
        let relu93_out1 = burn::tensor::activation::relu(conv2d93_out1);
        let averagepool2d9_out1 = self.averagepool2d9.forward(concat10_out1);
        let conv2d94_out1 = self.conv2d94.forward(averagepool2d9_out1);
        let relu94_out1 = burn::tensor::activation::relu(conv2d94_out1);
        let concat11_out1 = burn::tensor::Tensor::cat(
            [
                relu86_out1,
                relu88_out1,
                relu89_out1,
                relu92_out1,
                relu93_out1,
                relu94_out1,
            ]
            .into(),
            1,
        );
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(concat11_out1);
        let flatten1_out1 = globalaveragepool1_out1.flatten(1, 3);
        let gemm1_out1 = self.gemm1.forward(flatten1_out1);
        gemm1_out1
    }
}
