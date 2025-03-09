instances := 'alexnet'
work_dir := "$(pwd)"

alias b := build

build:
    cargo build --release


clean:
    cargo clean

# run instances in the example dir
run name: build
    cargo run --release -p {{name}}

# IF RUNNING examples FOR THE FIRST TIME
# by default torchvison installs torch automatically
install:
    pip install uv
    uv pip install torchvision

# convert a model to a format that can be handled by burn
convert model:
   python3 ./examples/py_Files/convert_to_pt.py {{model}} {{work_dir}}
