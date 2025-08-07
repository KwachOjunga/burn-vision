use burnvision::alexnet::AlexNetConfig;
use burnvision::burn::backend::Wgpu;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = AlexNetConfig::new(3000).init::<MyBackend>("relu", &device);
    println!("{:?}", model)
}
