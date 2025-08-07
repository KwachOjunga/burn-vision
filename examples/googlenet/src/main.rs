use burnvision::burn::backend::Wgpu;
use burnvision::googlenet::Model;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = burnvision::burn::backend::wgpu::WgpuDevice::default();
    let model: Model<MyBackend> = Model::new(&device);
    println!("{:#?}", model);
}
