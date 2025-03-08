use burn_vision::alexnet::AlexNetConfig;
use burn_vision::burn::backend::Wgpu;

fn main(){
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = AlexNetConfig::new(3000,).init::<MyBackend>(&device);
    println!("{:?}",model)
}
