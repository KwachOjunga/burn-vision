instances := 'alexnet'

alias b := build

build:
    cargo build --release


clean:
    cargo clean

# run instances in the example dir
run name: build
    cargo run --release -p {{name}}
