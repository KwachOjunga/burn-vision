// Generated from ONNX "../../models/onnx_dir/efficientnet_b0.onnx" by burn-import
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
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
    globalaveragepool1: AdaptiveAvgPool2d,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    globalaveragepool2: AdaptiveAvgPool2d,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    globalaveragepool3: AdaptiveAvgPool2d,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    globalaveragepool4: AdaptiveAvgPool2d,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    globalaveragepool5: AdaptiveAvgPool2d,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    globalaveragepool6: AdaptiveAvgPool2d,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    globalaveragepool7: AdaptiveAvgPool2d,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    globalaveragepool8: AdaptiveAvgPool2d,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    globalaveragepool9: AdaptiveAvgPool2d,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    globalaveragepool10: AdaptiveAvgPool2d,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    globalaveragepool11: AdaptiveAvgPool2d,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    globalaveragepool12: AdaptiveAvgPool2d,
    conv2d58: Conv2d<B>,
    conv2d59: Conv2d<B>,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    conv2d62: Conv2d<B>,
    globalaveragepool13: AdaptiveAvgPool2d,
    conv2d63: Conv2d<B>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    globalaveragepool14: AdaptiveAvgPool2d,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    conv2d72: Conv2d<B>,
    globalaveragepool15: AdaptiveAvgPool2d,
    conv2d73: Conv2d<B>,
    conv2d74: Conv2d<B>,
    conv2d75: Conv2d<B>,
    conv2d76: Conv2d<B>,
    conv2d77: Conv2d<B>,
    globalaveragepool16: AdaptiveAvgPool2d,
    conv2d78: Conv2d<B>,
    conv2d79: Conv2d<B>,
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    globalaveragepool17: AdaptiveAvgPool2d,
    gemm1: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("../../models/onnx_dir/efficientnet_b0", &Default::default())
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
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d3 = Conv2dConfig::new([32, 8], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([8, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([16, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(true)
            .init(device);
        let globalaveragepool2 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d8 = Conv2dConfig::new([96, 4], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([4, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([96, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([24, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([144, 144], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(144)
            .with_bias(true)
            .init(device);
        let globalaveragepool3 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d13 = Conv2dConfig::new([144, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([6, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([144, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([24, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([144, 144], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(144)
            .with_bias(true)
            .init(device);
        let globalaveragepool4 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d18 = Conv2dConfig::new([144, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([6, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([144, 40], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([40, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([240, 240], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(true)
            .init(device);
        let globalaveragepool5 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d23 = Conv2dConfig::new([240, 10], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([10, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([240, 40], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([40, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([240, 240], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(true)
            .init(device);
        let globalaveragepool6 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d28 = Conv2dConfig::new([240, 10], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([10, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([240, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([80, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([480, 480], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(true)
            .init(device);
        let globalaveragepool7 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d33 = Conv2dConfig::new([480, 20], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([20, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([480, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([80, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([480, 480], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(true)
            .init(device);
        let globalaveragepool8 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d38 = Conv2dConfig::new([480, 20], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([20, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([480, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([80, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([480, 480], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(true)
            .init(device);
        let globalaveragepool9 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d43 = Conv2dConfig::new([480, 20], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([20, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([480, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([112, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([672, 672], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(672)
            .with_bias(true)
            .init(device);
        let globalaveragepool10 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d48 = Conv2dConfig::new([672, 28], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([28, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([672, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([112, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([672, 672], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(672)
            .with_bias(true)
            .init(device);
        let globalaveragepool11 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d53 = Conv2dConfig::new([672, 28], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([28, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([672, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([112, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([672, 672], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(672)
            .with_bias(true)
            .init(device);
        let globalaveragepool12 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d58 = Conv2dConfig::new([672, 28], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([28, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([672, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([1152, 1152], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let globalaveragepool13 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d63 = Conv2dConfig::new([1152, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([48, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([1152, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([1152, 1152], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let globalaveragepool14 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d68 = Conv2dConfig::new([1152, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([48, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([1152, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([1152, 1152], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let globalaveragepool15 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d73 = Conv2dConfig::new([1152, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d74 = Conv2dConfig::new([48, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d75 = Conv2dConfig::new([1152, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([1152, 1152], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let globalaveragepool16 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv2d78 = Conv2dConfig::new([1152, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([48, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d80 = Conv2dConfig::new([1152, 320], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([320, 1280], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let globalaveragepool17 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let gemm1 = LinearConfig::new(1280, 1000).with_bias(true).init(device);
        Self {
            conv2d1,
            conv2d2,
            globalaveragepool1,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            globalaveragepool2,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            globalaveragepool3,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            globalaveragepool4,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            globalaveragepool5,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            globalaveragepool6,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            globalaveragepool7,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            globalaveragepool8,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            globalaveragepool9,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            globalaveragepool10,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            globalaveragepool11,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            globalaveragepool12,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            globalaveragepool13,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            globalaveragepool14,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            globalaveragepool15,
            conv2d73,
            conv2d74,
            conv2d75,
            conv2d76,
            conv2d77,
            globalaveragepool16,
            conv2d78,
            conv2d79,
            conv2d80,
            conv2d81,
            globalaveragepool17,
            gemm1,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d1_out1.clone());
        let mul1_out1 = conv2d1_out1.mul(sigmoid1_out1);
        let conv2d2_out1 = self.conv2d2.forward(mul1_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv2d2_out1.clone());
        let mul2_out1 = conv2d2_out1.mul(sigmoid2_out1);
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(mul2_out1.clone());
        let conv2d3_out1 = self.conv2d3.forward(globalaveragepool1_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv2d3_out1.clone());
        let mul3_out1 = conv2d3_out1.mul(sigmoid3_out1);
        let conv2d4_out1 = self.conv2d4.forward(mul3_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv2d4_out1);
        let mul4_out1 = sigmoid4_out1.mul(mul2_out1);
        let conv2d5_out1 = self.conv2d5.forward(mul4_out1);
        let conv2d6_out1 = self.conv2d6.forward(conv2d5_out1);
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(conv2d6_out1.clone());
        let mul5_out1 = conv2d6_out1.mul(sigmoid5_out1);
        let conv2d7_out1 = self.conv2d7.forward(mul5_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(conv2d7_out1.clone());
        let mul6_out1 = conv2d7_out1.mul(sigmoid6_out1);
        let globalaveragepool2_out1 = self.globalaveragepool2.forward(mul6_out1.clone());
        let conv2d8_out1 = self.conv2d8.forward(globalaveragepool2_out1);
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(conv2d8_out1.clone());
        let mul7_out1 = conv2d8_out1.mul(sigmoid7_out1);
        let conv2d9_out1 = self.conv2d9.forward(mul7_out1);
        let sigmoid8_out1 = burn::tensor::activation::sigmoid(conv2d9_out1);
        let mul8_out1 = sigmoid8_out1.mul(mul6_out1);
        let conv2d10_out1 = self.conv2d10.forward(mul8_out1);
        let conv2d11_out1 = self.conv2d11.forward(conv2d10_out1.clone());
        let sigmoid9_out1 = burn::tensor::activation::sigmoid(conv2d11_out1.clone());
        let mul9_out1 = conv2d11_out1.mul(sigmoid9_out1);
        let conv2d12_out1 = self.conv2d12.forward(mul9_out1);
        let sigmoid10_out1 = burn::tensor::activation::sigmoid(conv2d12_out1.clone());
        let mul10_out1 = conv2d12_out1.mul(sigmoid10_out1);
        let globalaveragepool3_out1 = self.globalaveragepool3.forward(mul10_out1.clone());
        let conv2d13_out1 = self.conv2d13.forward(globalaveragepool3_out1);
        let sigmoid11_out1 = burn::tensor::activation::sigmoid(conv2d13_out1.clone());
        let mul11_out1 = conv2d13_out1.mul(sigmoid11_out1);
        let conv2d14_out1 = self.conv2d14.forward(mul11_out1);
        let sigmoid12_out1 = burn::tensor::activation::sigmoid(conv2d14_out1);
        let mul12_out1 = sigmoid12_out1.mul(mul10_out1);
        let conv2d15_out1 = self.conv2d15.forward(mul12_out1);
        let add1_out1 = conv2d15_out1.add(conv2d10_out1);
        let conv2d16_out1 = self.conv2d16.forward(add1_out1);
        let sigmoid13_out1 = burn::tensor::activation::sigmoid(conv2d16_out1.clone());
        let mul13_out1 = conv2d16_out1.mul(sigmoid13_out1);
        let conv2d17_out1 = self.conv2d17.forward(mul13_out1);
        let sigmoid14_out1 = burn::tensor::activation::sigmoid(conv2d17_out1.clone());
        let mul14_out1 = conv2d17_out1.mul(sigmoid14_out1);
        let globalaveragepool4_out1 = self.globalaveragepool4.forward(mul14_out1.clone());
        let conv2d18_out1 = self.conv2d18.forward(globalaveragepool4_out1);
        let sigmoid15_out1 = burn::tensor::activation::sigmoid(conv2d18_out1.clone());
        let mul15_out1 = conv2d18_out1.mul(sigmoid15_out1);
        let conv2d19_out1 = self.conv2d19.forward(mul15_out1);
        let sigmoid16_out1 = burn::tensor::activation::sigmoid(conv2d19_out1);
        let mul16_out1 = sigmoid16_out1.mul(mul14_out1);
        let conv2d20_out1 = self.conv2d20.forward(mul16_out1);
        let conv2d21_out1 = self.conv2d21.forward(conv2d20_out1.clone());
        let sigmoid17_out1 = burn::tensor::activation::sigmoid(conv2d21_out1.clone());
        let mul17_out1 = conv2d21_out1.mul(sigmoid17_out1);
        let conv2d22_out1 = self.conv2d22.forward(mul17_out1);
        let sigmoid18_out1 = burn::tensor::activation::sigmoid(conv2d22_out1.clone());
        let mul18_out1 = conv2d22_out1.mul(sigmoid18_out1);
        let globalaveragepool5_out1 = self.globalaveragepool5.forward(mul18_out1.clone());
        let conv2d23_out1 = self.conv2d23.forward(globalaveragepool5_out1);
        let sigmoid19_out1 = burn::tensor::activation::sigmoid(conv2d23_out1.clone());
        let mul19_out1 = conv2d23_out1.mul(sigmoid19_out1);
        let conv2d24_out1 = self.conv2d24.forward(mul19_out1);
        let sigmoid20_out1 = burn::tensor::activation::sigmoid(conv2d24_out1);
        let mul20_out1 = sigmoid20_out1.mul(mul18_out1);
        let conv2d25_out1 = self.conv2d25.forward(mul20_out1);
        let add2_out1 = conv2d25_out1.add(conv2d20_out1);
        let conv2d26_out1 = self.conv2d26.forward(add2_out1);
        let sigmoid21_out1 = burn::tensor::activation::sigmoid(conv2d26_out1.clone());
        let mul21_out1 = conv2d26_out1.mul(sigmoid21_out1);
        let conv2d27_out1 = self.conv2d27.forward(mul21_out1);
        let sigmoid22_out1 = burn::tensor::activation::sigmoid(conv2d27_out1.clone());
        let mul22_out1 = conv2d27_out1.mul(sigmoid22_out1);
        let globalaveragepool6_out1 = self.globalaveragepool6.forward(mul22_out1.clone());
        let conv2d28_out1 = self.conv2d28.forward(globalaveragepool6_out1);
        let sigmoid23_out1 = burn::tensor::activation::sigmoid(conv2d28_out1.clone());
        let mul23_out1 = conv2d28_out1.mul(sigmoid23_out1);
        let conv2d29_out1 = self.conv2d29.forward(mul23_out1);
        let sigmoid24_out1 = burn::tensor::activation::sigmoid(conv2d29_out1);
        let mul24_out1 = sigmoid24_out1.mul(mul22_out1);
        let conv2d30_out1 = self.conv2d30.forward(mul24_out1);
        let conv2d31_out1 = self.conv2d31.forward(conv2d30_out1.clone());
        let sigmoid25_out1 = burn::tensor::activation::sigmoid(conv2d31_out1.clone());
        let mul25_out1 = conv2d31_out1.mul(sigmoid25_out1);
        let conv2d32_out1 = self.conv2d32.forward(mul25_out1);
        let sigmoid26_out1 = burn::tensor::activation::sigmoid(conv2d32_out1.clone());
        let mul26_out1 = conv2d32_out1.mul(sigmoid26_out1);
        let globalaveragepool7_out1 = self.globalaveragepool7.forward(mul26_out1.clone());
        let conv2d33_out1 = self.conv2d33.forward(globalaveragepool7_out1);
        let sigmoid27_out1 = burn::tensor::activation::sigmoid(conv2d33_out1.clone());
        let mul27_out1 = conv2d33_out1.mul(sigmoid27_out1);
        let conv2d34_out1 = self.conv2d34.forward(mul27_out1);
        let sigmoid28_out1 = burn::tensor::activation::sigmoid(conv2d34_out1);
        let mul28_out1 = sigmoid28_out1.mul(mul26_out1);
        let conv2d35_out1 = self.conv2d35.forward(mul28_out1);
        let add3_out1 = conv2d35_out1.add(conv2d30_out1);
        let conv2d36_out1 = self.conv2d36.forward(add3_out1.clone());
        let sigmoid29_out1 = burn::tensor::activation::sigmoid(conv2d36_out1.clone());
        let mul29_out1 = conv2d36_out1.mul(sigmoid29_out1);
        let conv2d37_out1 = self.conv2d37.forward(mul29_out1);
        let sigmoid30_out1 = burn::tensor::activation::sigmoid(conv2d37_out1.clone());
        let mul30_out1 = conv2d37_out1.mul(sigmoid30_out1);
        let globalaveragepool8_out1 = self.globalaveragepool8.forward(mul30_out1.clone());
        let conv2d38_out1 = self.conv2d38.forward(globalaveragepool8_out1);
        let sigmoid31_out1 = burn::tensor::activation::sigmoid(conv2d38_out1.clone());
        let mul31_out1 = conv2d38_out1.mul(sigmoid31_out1);
        let conv2d39_out1 = self.conv2d39.forward(mul31_out1);
        let sigmoid32_out1 = burn::tensor::activation::sigmoid(conv2d39_out1);
        let mul32_out1 = sigmoid32_out1.mul(mul30_out1);
        let conv2d40_out1 = self.conv2d40.forward(mul32_out1);
        let add4_out1 = conv2d40_out1.add(add3_out1);
        let conv2d41_out1 = self.conv2d41.forward(add4_out1);
        let sigmoid33_out1 = burn::tensor::activation::sigmoid(conv2d41_out1.clone());
        let mul33_out1 = conv2d41_out1.mul(sigmoid33_out1);
        let conv2d42_out1 = self.conv2d42.forward(mul33_out1);
        let sigmoid34_out1 = burn::tensor::activation::sigmoid(conv2d42_out1.clone());
        let mul34_out1 = conv2d42_out1.mul(sigmoid34_out1);
        let globalaveragepool9_out1 = self.globalaveragepool9.forward(mul34_out1.clone());
        let conv2d43_out1 = self.conv2d43.forward(globalaveragepool9_out1);
        let sigmoid35_out1 = burn::tensor::activation::sigmoid(conv2d43_out1.clone());
        let mul35_out1 = conv2d43_out1.mul(sigmoid35_out1);
        let conv2d44_out1 = self.conv2d44.forward(mul35_out1);
        let sigmoid36_out1 = burn::tensor::activation::sigmoid(conv2d44_out1);
        let mul36_out1 = sigmoid36_out1.mul(mul34_out1);
        let conv2d45_out1 = self.conv2d45.forward(mul36_out1);
        let conv2d46_out1 = self.conv2d46.forward(conv2d45_out1.clone());
        let sigmoid37_out1 = burn::tensor::activation::sigmoid(conv2d46_out1.clone());
        let mul37_out1 = conv2d46_out1.mul(sigmoid37_out1);
        let conv2d47_out1 = self.conv2d47.forward(mul37_out1);
        let sigmoid38_out1 = burn::tensor::activation::sigmoid(conv2d47_out1.clone());
        let mul38_out1 = conv2d47_out1.mul(sigmoid38_out1);
        let globalaveragepool10_out1 = self.globalaveragepool10.forward(mul38_out1.clone());
        let conv2d48_out1 = self.conv2d48.forward(globalaveragepool10_out1);
        let sigmoid39_out1 = burn::tensor::activation::sigmoid(conv2d48_out1.clone());
        let mul39_out1 = conv2d48_out1.mul(sigmoid39_out1);
        let conv2d49_out1 = self.conv2d49.forward(mul39_out1);
        let sigmoid40_out1 = burn::tensor::activation::sigmoid(conv2d49_out1);
        let mul40_out1 = sigmoid40_out1.mul(mul38_out1);
        let conv2d50_out1 = self.conv2d50.forward(mul40_out1);
        let add5_out1 = conv2d50_out1.add(conv2d45_out1);
        let conv2d51_out1 = self.conv2d51.forward(add5_out1.clone());
        let sigmoid41_out1 = burn::tensor::activation::sigmoid(conv2d51_out1.clone());
        let mul41_out1 = conv2d51_out1.mul(sigmoid41_out1);
        let conv2d52_out1 = self.conv2d52.forward(mul41_out1);
        let sigmoid42_out1 = burn::tensor::activation::sigmoid(conv2d52_out1.clone());
        let mul42_out1 = conv2d52_out1.mul(sigmoid42_out1);
        let globalaveragepool11_out1 = self.globalaveragepool11.forward(mul42_out1.clone());
        let conv2d53_out1 = self.conv2d53.forward(globalaveragepool11_out1);
        let sigmoid43_out1 = burn::tensor::activation::sigmoid(conv2d53_out1.clone());
        let mul43_out1 = conv2d53_out1.mul(sigmoid43_out1);
        let conv2d54_out1 = self.conv2d54.forward(mul43_out1);
        let sigmoid44_out1 = burn::tensor::activation::sigmoid(conv2d54_out1);
        let mul44_out1 = sigmoid44_out1.mul(mul42_out1);
        let conv2d55_out1 = self.conv2d55.forward(mul44_out1);
        let add6_out1 = conv2d55_out1.add(add5_out1);
        let conv2d56_out1 = self.conv2d56.forward(add6_out1);
        let sigmoid45_out1 = burn::tensor::activation::sigmoid(conv2d56_out1.clone());
        let mul45_out1 = conv2d56_out1.mul(sigmoid45_out1);
        let conv2d57_out1 = self.conv2d57.forward(mul45_out1);
        let sigmoid46_out1 = burn::tensor::activation::sigmoid(conv2d57_out1.clone());
        let mul46_out1 = conv2d57_out1.mul(sigmoid46_out1);
        let globalaveragepool12_out1 = self.globalaveragepool12.forward(mul46_out1.clone());
        let conv2d58_out1 = self.conv2d58.forward(globalaveragepool12_out1);
        let sigmoid47_out1 = burn::tensor::activation::sigmoid(conv2d58_out1.clone());
        let mul47_out1 = conv2d58_out1.mul(sigmoid47_out1);
        let conv2d59_out1 = self.conv2d59.forward(mul47_out1);
        let sigmoid48_out1 = burn::tensor::activation::sigmoid(conv2d59_out1);
        let mul48_out1 = sigmoid48_out1.mul(mul46_out1);
        let conv2d60_out1 = self.conv2d60.forward(mul48_out1);
        let conv2d61_out1 = self.conv2d61.forward(conv2d60_out1.clone());
        let sigmoid49_out1 = burn::tensor::activation::sigmoid(conv2d61_out1.clone());
        let mul49_out1 = conv2d61_out1.mul(sigmoid49_out1);
        let conv2d62_out1 = self.conv2d62.forward(mul49_out1);
        let sigmoid50_out1 = burn::tensor::activation::sigmoid(conv2d62_out1.clone());
        let mul50_out1 = conv2d62_out1.mul(sigmoid50_out1);
        let globalaveragepool13_out1 = self.globalaveragepool13.forward(mul50_out1.clone());
        let conv2d63_out1 = self.conv2d63.forward(globalaveragepool13_out1);
        let sigmoid51_out1 = burn::tensor::activation::sigmoid(conv2d63_out1.clone());
        let mul51_out1 = conv2d63_out1.mul(sigmoid51_out1);
        let conv2d64_out1 = self.conv2d64.forward(mul51_out1);
        let sigmoid52_out1 = burn::tensor::activation::sigmoid(conv2d64_out1);
        let mul52_out1 = sigmoid52_out1.mul(mul50_out1);
        let conv2d65_out1 = self.conv2d65.forward(mul52_out1);
        let add7_out1 = conv2d65_out1.add(conv2d60_out1);
        let conv2d66_out1 = self.conv2d66.forward(add7_out1.clone());
        let sigmoid53_out1 = burn::tensor::activation::sigmoid(conv2d66_out1.clone());
        let mul53_out1 = conv2d66_out1.mul(sigmoid53_out1);
        let conv2d67_out1 = self.conv2d67.forward(mul53_out1);
        let sigmoid54_out1 = burn::tensor::activation::sigmoid(conv2d67_out1.clone());
        let mul54_out1 = conv2d67_out1.mul(sigmoid54_out1);
        let globalaveragepool14_out1 = self.globalaveragepool14.forward(mul54_out1.clone());
        let conv2d68_out1 = self.conv2d68.forward(globalaveragepool14_out1);
        let sigmoid55_out1 = burn::tensor::activation::sigmoid(conv2d68_out1.clone());
        let mul55_out1 = conv2d68_out1.mul(sigmoid55_out1);
        let conv2d69_out1 = self.conv2d69.forward(mul55_out1);
        let sigmoid56_out1 = burn::tensor::activation::sigmoid(conv2d69_out1);
        let mul56_out1 = sigmoid56_out1.mul(mul54_out1);
        let conv2d70_out1 = self.conv2d70.forward(mul56_out1);
        let add8_out1 = conv2d70_out1.add(add7_out1);
        let conv2d71_out1 = self.conv2d71.forward(add8_out1.clone());
        let sigmoid57_out1 = burn::tensor::activation::sigmoid(conv2d71_out1.clone());
        let mul57_out1 = conv2d71_out1.mul(sigmoid57_out1);
        let conv2d72_out1 = self.conv2d72.forward(mul57_out1);
        let sigmoid58_out1 = burn::tensor::activation::sigmoid(conv2d72_out1.clone());
        let mul58_out1 = conv2d72_out1.mul(sigmoid58_out1);
        let globalaveragepool15_out1 = self.globalaveragepool15.forward(mul58_out1.clone());
        let conv2d73_out1 = self.conv2d73.forward(globalaveragepool15_out1);
        let sigmoid59_out1 = burn::tensor::activation::sigmoid(conv2d73_out1.clone());
        let mul59_out1 = conv2d73_out1.mul(sigmoid59_out1);
        let conv2d74_out1 = self.conv2d74.forward(mul59_out1);
        let sigmoid60_out1 = burn::tensor::activation::sigmoid(conv2d74_out1);
        let mul60_out1 = sigmoid60_out1.mul(mul58_out1);
        let conv2d75_out1 = self.conv2d75.forward(mul60_out1);
        let add9_out1 = conv2d75_out1.add(add8_out1);
        let conv2d76_out1 = self.conv2d76.forward(add9_out1);
        let sigmoid61_out1 = burn::tensor::activation::sigmoid(conv2d76_out1.clone());
        let mul61_out1 = conv2d76_out1.mul(sigmoid61_out1);
        let conv2d77_out1 = self.conv2d77.forward(mul61_out1);
        let sigmoid62_out1 = burn::tensor::activation::sigmoid(conv2d77_out1.clone());
        let mul62_out1 = conv2d77_out1.mul(sigmoid62_out1);
        let globalaveragepool16_out1 = self.globalaveragepool16.forward(mul62_out1.clone());
        let conv2d78_out1 = self.conv2d78.forward(globalaveragepool16_out1);
        let sigmoid63_out1 = burn::tensor::activation::sigmoid(conv2d78_out1.clone());
        let mul63_out1 = conv2d78_out1.mul(sigmoid63_out1);
        let conv2d79_out1 = self.conv2d79.forward(mul63_out1);
        let sigmoid64_out1 = burn::tensor::activation::sigmoid(conv2d79_out1);
        let mul64_out1 = sigmoid64_out1.mul(mul62_out1);
        let conv2d80_out1 = self.conv2d80.forward(mul64_out1);
        let conv2d81_out1 = self.conv2d81.forward(conv2d80_out1);
        let sigmoid65_out1 = burn::tensor::activation::sigmoid(conv2d81_out1.clone());
        let mul65_out1 = conv2d81_out1.mul(sigmoid65_out1);
        let globalaveragepool17_out1 = self.globalaveragepool17.forward(mul65_out1);
        let flatten1_out1 = globalaveragepool17_out1.flatten(1, 3);
        let gemm1_out1 = self.gemm1.forward(flatten1_out1);
        gemm1_out1
    }
}
