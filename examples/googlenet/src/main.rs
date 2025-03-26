use burn_vision::burn::backend::Wgpu;
use burn_vision::googlenet::Model;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = burn_vision::burn::backend::wgpu::WgpuDevice::default();
    let model: Model<MyBackend> = Model::new(&device);
    println!("{:#?}", model);
}
