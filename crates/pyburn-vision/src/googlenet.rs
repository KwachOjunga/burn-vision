// Generated from ONNX "../../models/onnx_dir/googlenet.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
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
pub struct Googlenet<B: Backend> {
    conv2d1: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    maxpool2d4: MaxPool2d,
    conv2d15: Conv2d<B>,
    maxpool2d5: MaxPool2d,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    maxpool2d6: MaxPool2d,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    maxpool2d7: MaxPool2d,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    maxpool2d8: MaxPool2d,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    maxpool2d9: MaxPool2d,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    maxpool2d10: MaxPool2d,
    conv2d45: Conv2d<B>,
    maxpool2d11: MaxPool2d,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    maxpool2d12: MaxPool2d,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    conv2d56: Conv2d<B>,
    maxpool2d13: MaxPool2d,
    conv2d57: Conv2d<B>,
    globalaveragepool1: AdaptiveAvgPool2d,
    gemm1: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


// impl<B: Backend> Default for Model<B> {
//     fn default() -> Self {
//         Self::from_file("../../models/onnx_dir/googlenet", &Default::default())
//     }
// }

// impl<B: Backend> Model<B> {
//     pub fn from_file(file: &str, device: &B::Device) -> Self {
//         let record = burn::record::PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
//             .load(file.into(), device)
//             .expect("Record file to exist.");
//         Self::new(device).load_record(record)
//     }
// }

impl<B: Backend> Googlenet<B> {
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
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d2 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([64, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d4 = Conv2dConfig::new([192, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([192, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([96, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([192, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([16, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d9 = Conv2dConfig::new([192, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([128, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([32, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d4 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d15 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d5 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d16 = Conv2dConfig::new([480, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([480, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([96, 208], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([480, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([16, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d6 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d21 = Conv2dConfig::new([480, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([512, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([512, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([112, 224], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([512, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([24, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d7 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d27 = Conv2dConfig::new([512, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([512, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([24, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d8 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d33 = Conv2dConfig::new([512, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([512, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([512, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([144, 288], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([512, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d9 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d39 = Conv2dConfig::new([512, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([528, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([528, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([160, 320], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([528, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([32, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d10 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d45 = Conv2dConfig::new([528, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d11 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d46 = Conv2dConfig::new([832, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([832, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([160, 320], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([832, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([32, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d12 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d51 = Conv2dConfig::new([832, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([832, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([832, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([192, 384], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([832, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([48, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d13 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d57 = Conv2dConfig::new([832, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let gemm1 = LinearConfig::new(1024, 1000).with_bias(true).init(device);
        Self {
            conv2d1,
            maxpool2d1,
            conv2d2,
            conv2d3,
            maxpool2d2,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            maxpool2d3,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            maxpool2d4,
            conv2d15,
            maxpool2d5,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            maxpool2d6,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            maxpool2d7,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            maxpool2d8,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            maxpool2d9,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            maxpool2d10,
            conv2d45,
            maxpool2d11,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            maxpool2d12,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            maxpool2d13,
            conv2d57,
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
        let conv2d2_out1 = self.conv2d2.forward(maxpool2d1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu3_out1);
        let conv2d4_out1 = self.conv2d4.forward(maxpool2d2_out1.clone());
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(maxpool2d2_out1.clone());
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(maxpool2d2_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(maxpool2d2_out1);
        let conv2d9_out1 = self.conv2d9.forward(maxpool2d3_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let concat1_out1 =
            burn::tensor::Tensor::cat([relu4_out1, relu6_out1, relu8_out1, relu9_out1].into(), 1);
        let conv2d10_out1 = self.conv2d10.forward(concat1_out1.clone());
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(concat1_out1.clone());
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = self.conv2d13.forward(concat1_out1.clone());
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu13_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let maxpool2d4_out1 = self.maxpool2d4.forward(concat1_out1);
        let conv2d15_out1 = self.conv2d15.forward(maxpool2d4_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [relu10_out1, relu12_out1, relu14_out1, relu15_out1].into(),
            1,
        );
        let maxpool2d5_out1 = self.maxpool2d5.forward(concat2_out1);
        let conv2d16_out1 = self.conv2d16.forward(maxpool2d5_out1.clone());
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(maxpool2d5_out1.clone());
        let relu17_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let conv2d19_out1 = self.conv2d19.forward(maxpool2d5_out1.clone());
        let relu19_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu19_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let maxpool2d6_out1 = self.maxpool2d6.forward(maxpool2d5_out1);
        let conv2d21_out1 = self.conv2d21.forward(maxpool2d6_out1);
        let relu21_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [relu16_out1, relu18_out1, relu20_out1, relu21_out1].into(),
            1,
        );
        let conv2d22_out1 = self.conv2d22.forward(concat3_out1.clone());
        let relu22_out1 = burn::tensor::activation::relu(conv2d22_out1);
        let conv2d23_out1 = self.conv2d23.forward(concat3_out1.clone());
        let relu23_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu23_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(concat3_out1.clone());
        let relu25_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let maxpool2d7_out1 = self.maxpool2d7.forward(concat3_out1);
        let conv2d27_out1 = self.conv2d27.forward(maxpool2d7_out1);
        let relu27_out1 = burn::tensor::activation::relu(conv2d27_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [relu22_out1, relu24_out1, relu26_out1, relu27_out1].into(),
            1,
        );
        let conv2d28_out1 = self.conv2d28.forward(concat4_out1.clone());
        let relu28_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = self.conv2d29.forward(concat4_out1.clone());
        let relu29_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu29_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = self.conv2d31.forward(concat4_out1.clone());
        let relu31_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu31_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv2d32_out1);
        let maxpool2d8_out1 = self.maxpool2d8.forward(concat4_out1);
        let conv2d33_out1 = self.conv2d33.forward(maxpool2d8_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [relu28_out1, relu30_out1, relu32_out1, relu33_out1].into(),
            1,
        );
        let conv2d34_out1 = self.conv2d34.forward(concat5_out1.clone());
        let relu34_out1 = burn::tensor::activation::relu(conv2d34_out1);
        let conv2d35_out1 = self.conv2d35.forward(concat5_out1.clone());
        let relu35_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu35_out1);
        let relu36_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let conv2d37_out1 = self.conv2d37.forward(concat5_out1.clone());
        let relu37_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu37_out1);
        let relu38_out1 = burn::tensor::activation::relu(conv2d38_out1);
        let maxpool2d9_out1 = self.maxpool2d9.forward(concat5_out1);
        let conv2d39_out1 = self.conv2d39.forward(maxpool2d9_out1);
        let relu39_out1 = burn::tensor::activation::relu(conv2d39_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [relu34_out1, relu36_out1, relu38_out1, relu39_out1].into(),
            1,
        );
        let conv2d40_out1 = self.conv2d40.forward(concat6_out1.clone());
        let relu40_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let conv2d41_out1 = self.conv2d41.forward(concat6_out1.clone());
        let relu41_out1 = burn::tensor::activation::relu(conv2d41_out1);
        let conv2d42_out1 = self.conv2d42.forward(relu41_out1);
        let relu42_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = self.conv2d43.forward(concat6_out1.clone());
        let relu43_out1 = burn::tensor::activation::relu(conv2d43_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu43_out1);
        let relu44_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let maxpool2d10_out1 = self.maxpool2d10.forward(concat6_out1);
        let conv2d45_out1 = self.conv2d45.forward(maxpool2d10_out1);
        let relu45_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [relu40_out1, relu42_out1, relu44_out1, relu45_out1].into(),
            1,
        );
        let maxpool2d11_out1 = self.maxpool2d11.forward(concat7_out1);
        let conv2d46_out1 = self.conv2d46.forward(maxpool2d11_out1.clone());
        let relu46_out1 = burn::tensor::activation::relu(conv2d46_out1);
        let conv2d47_out1 = self.conv2d47.forward(maxpool2d11_out1.clone());
        let relu47_out1 = burn::tensor::activation::relu(conv2d47_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu47_out1);
        let relu48_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(maxpool2d11_out1.clone());
        let relu49_out1 = burn::tensor::activation::relu(conv2d49_out1);
        let conv2d50_out1 = self.conv2d50.forward(relu49_out1);
        let relu50_out1 = burn::tensor::activation::relu(conv2d50_out1);
        let maxpool2d12_out1 = self.maxpool2d12.forward(maxpool2d11_out1);
        let conv2d51_out1 = self.conv2d51.forward(maxpool2d12_out1);
        let relu51_out1 = burn::tensor::activation::relu(conv2d51_out1);
        let concat8_out1 = burn::tensor::Tensor::cat(
            [relu46_out1, relu48_out1, relu50_out1, relu51_out1].into(),
            1,
        );
        let conv2d52_out1 = self.conv2d52.forward(concat8_out1.clone());
        let relu52_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let conv2d53_out1 = self.conv2d53.forward(concat8_out1.clone());
        let relu53_out1 = burn::tensor::activation::relu(conv2d53_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu53_out1);
        let relu54_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let conv2d55_out1 = self.conv2d55.forward(concat8_out1.clone());
        let relu55_out1 = burn::tensor::activation::relu(conv2d55_out1);
        let conv2d56_out1 = self.conv2d56.forward(relu55_out1);
        let relu56_out1 = burn::tensor::activation::relu(conv2d56_out1);
        let maxpool2d13_out1 = self.maxpool2d13.forward(concat8_out1);
        let conv2d57_out1 = self.conv2d57.forward(maxpool2d13_out1);
        let relu57_out1 = burn::tensor::activation::relu(conv2d57_out1);
        let concat9_out1 = burn::tensor::Tensor::cat(
            [relu52_out1, relu54_out1, relu56_out1, relu57_out1].into(),
            1,
        );
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(concat9_out1);
        let flatten1_out1 = globalaveragepool1_out1.flatten(1, 3);
        let gemm1_out1 = self.gemm1.forward(flatten1_out1);
        gemm1_out1
    }
}
