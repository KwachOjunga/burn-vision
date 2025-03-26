// Generated from ONNX "../../models/onnx_dir/regnet_x_16gf.onnx" by burn-import
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
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
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
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    globalaveragepool1: AdaptiveAvgPool2d,
    gemm1: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("../../models/onnx_dir/regnet_x_16gf", &Default::default())
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
        let conv2d2 = Conv2dConfig::new([32, 256], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(2)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(2)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([512, 896], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([512, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(7)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([896, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([896, 2048], [1, 1])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([896, 2048], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([2048, 2048], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([2048, 2048], [1, 1])
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
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
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
            conv2d70,
            conv2d71,
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
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1.clone());
        let conv2d3_out1 = self.conv2d3.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu3_out1);
        let add1_out1 = conv2d2_out1.add(conv2d5_out1);
        let relu4_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu4_out1.clone());
        let relu5_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1);
        let add2_out1 = relu4_out1.add(conv2d8_out1);
        let relu7_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu7_out1.clone());
        let conv2d10_out1 = self.conv2d10.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu9_out1);
        let add3_out1 = conv2d9_out1.add(conv2d12_out1);
        let relu10_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu10_out1.clone());
        let relu11_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu12_out1);
        let add4_out1 = relu10_out1.add(conv2d15_out1);
        let relu13_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu13_out1.clone());
        let relu14_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu14_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu15_out1);
        let add5_out1 = relu13_out1.add(conv2d18_out1);
        let relu16_out1 = burn::tensor::activation::relu(add5_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu16_out1.clone());
        let relu17_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu17_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu18_out1);
        let add6_out1 = relu16_out1.add(conv2d21_out1);
        let relu19_out1 = burn::tensor::activation::relu(add6_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu19_out1.clone());
        let relu20_out1 = burn::tensor::activation::relu(conv2d22_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu20_out1);
        let relu21_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu21_out1);
        let add7_out1 = relu19_out1.add(conv2d24_out1);
        let relu22_out1 = burn::tensor::activation::relu(add7_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu22_out1.clone());
        let relu23_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu23_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu24_out1);
        let add8_out1 = relu22_out1.add(conv2d27_out1);
        let relu25_out1 = burn::tensor::activation::relu(add8_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu25_out1.clone());
        let conv2d29_out1 = self.conv2d29.forward(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu26_out1);
        let relu27_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu27_out1);
        let add9_out1 = conv2d28_out1.add(conv2d31_out1);
        let relu28_out1 = burn::tensor::activation::relu(add9_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu28_out1.clone());
        let relu29_out1 = burn::tensor::activation::relu(conv2d32_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu29_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu30_out1);
        let add10_out1 = relu28_out1.add(conv2d34_out1);
        let relu31_out1 = burn::tensor::activation::relu(add10_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu31_out1.clone());
        let relu32_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu32_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu33_out1);
        let add11_out1 = relu31_out1.add(conv2d37_out1);
        let relu34_out1 = burn::tensor::activation::relu(add11_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu34_out1.clone());
        let relu35_out1 = burn::tensor::activation::relu(conv2d38_out1);
        let conv2d39_out1 = self.conv2d39.forward(relu35_out1);
        let relu36_out1 = burn::tensor::activation::relu(conv2d39_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu36_out1);
        let add12_out1 = relu34_out1.add(conv2d40_out1);
        let relu37_out1 = burn::tensor::activation::relu(add12_out1);
        let conv2d41_out1 = self.conv2d41.forward(relu37_out1.clone());
        let relu38_out1 = burn::tensor::activation::relu(conv2d41_out1);
        let conv2d42_out1 = self.conv2d42.forward(relu38_out1);
        let relu39_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu39_out1);
        let add13_out1 = relu37_out1.add(conv2d43_out1);
        let relu40_out1 = burn::tensor::activation::relu(add13_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu40_out1.clone());
        let relu41_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = self.conv2d45.forward(relu41_out1);
        let relu42_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let conv2d46_out1 = self.conv2d46.forward(relu42_out1);
        let add14_out1 = relu40_out1.add(conv2d46_out1);
        let relu43_out1 = burn::tensor::activation::relu(add14_out1);
        let conv2d47_out1 = self.conv2d47.forward(relu43_out1.clone());
        let relu44_out1 = burn::tensor::activation::relu(conv2d47_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(relu45_out1);
        let add15_out1 = relu43_out1.add(conv2d49_out1);
        let relu46_out1 = burn::tensor::activation::relu(add15_out1);
        let conv2d50_out1 = self.conv2d50.forward(relu46_out1.clone());
        let relu47_out1 = burn::tensor::activation::relu(conv2d50_out1);
        let conv2d51_out1 = self.conv2d51.forward(relu47_out1);
        let relu48_out1 = burn::tensor::activation::relu(conv2d51_out1);
        let conv2d52_out1 = self.conv2d52.forward(relu48_out1);
        let add16_out1 = relu46_out1.add(conv2d52_out1);
        let relu49_out1 = burn::tensor::activation::relu(add16_out1);
        let conv2d53_out1 = self.conv2d53.forward(relu49_out1.clone());
        let relu50_out1 = burn::tensor::activation::relu(conv2d53_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu50_out1);
        let relu51_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let conv2d55_out1 = self.conv2d55.forward(relu51_out1);
        let add17_out1 = relu49_out1.add(conv2d55_out1);
        let relu52_out1 = burn::tensor::activation::relu(add17_out1);
        let conv2d56_out1 = self.conv2d56.forward(relu52_out1.clone());
        let relu53_out1 = burn::tensor::activation::relu(conv2d56_out1);
        let conv2d57_out1 = self.conv2d57.forward(relu53_out1);
        let relu54_out1 = burn::tensor::activation::relu(conv2d57_out1);
        let conv2d58_out1 = self.conv2d58.forward(relu54_out1);
        let add18_out1 = relu52_out1.add(conv2d58_out1);
        let relu55_out1 = burn::tensor::activation::relu(add18_out1);
        let conv2d59_out1 = self.conv2d59.forward(relu55_out1.clone());
        let relu56_out1 = burn::tensor::activation::relu(conv2d59_out1);
        let conv2d60_out1 = self.conv2d60.forward(relu56_out1);
        let relu57_out1 = burn::tensor::activation::relu(conv2d60_out1);
        let conv2d61_out1 = self.conv2d61.forward(relu57_out1);
        let add19_out1 = relu55_out1.add(conv2d61_out1);
        let relu58_out1 = burn::tensor::activation::relu(add19_out1);
        let conv2d62_out1 = self.conv2d62.forward(relu58_out1.clone());
        let relu59_out1 = burn::tensor::activation::relu(conv2d62_out1);
        let conv2d63_out1 = self.conv2d63.forward(relu59_out1);
        let relu60_out1 = burn::tensor::activation::relu(conv2d63_out1);
        let conv2d64_out1 = self.conv2d64.forward(relu60_out1);
        let add20_out1 = relu58_out1.add(conv2d64_out1);
        let relu61_out1 = burn::tensor::activation::relu(add20_out1);
        let conv2d65_out1 = self.conv2d65.forward(relu61_out1.clone());
        let relu62_out1 = burn::tensor::activation::relu(conv2d65_out1);
        let conv2d66_out1 = self.conv2d66.forward(relu62_out1);
        let relu63_out1 = burn::tensor::activation::relu(conv2d66_out1);
        let conv2d67_out1 = self.conv2d67.forward(relu63_out1);
        let add21_out1 = relu61_out1.add(conv2d67_out1);
        let relu64_out1 = burn::tensor::activation::relu(add21_out1);
        let conv2d68_out1 = self.conv2d68.forward(relu64_out1.clone());
        let conv2d69_out1 = self.conv2d69.forward(relu64_out1);
        let relu65_out1 = burn::tensor::activation::relu(conv2d69_out1);
        let conv2d70_out1 = self.conv2d70.forward(relu65_out1);
        let relu66_out1 = burn::tensor::activation::relu(conv2d70_out1);
        let conv2d71_out1 = self.conv2d71.forward(relu66_out1);
        let add22_out1 = conv2d68_out1.add(conv2d71_out1);
        let relu67_out1 = burn::tensor::activation::relu(add22_out1);
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(relu67_out1);
        let flatten1_out1 = globalaveragepool1_out1.flatten(1, 3);
        let gemm1_out1 = self.gemm1.forward(flatten1_out1);
        gemm1_out1
    }
}
