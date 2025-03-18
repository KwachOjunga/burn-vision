# Introduction

This portion aims to describe a python interface for pyburn-vision
henceforth described as p_vision.

p_vision's interface aims to be a simple and straightforward interface,
eliminating any unneccessary complexity that may be introduced as a
consequence of the underlying implementation in Rust.
That being said, it may be difficult to comletely hide the intricacies
involved granted that burn's implementation offers a certain degree of
flexibility for it's users.

## Design decisions.

Given that burn offers an alternative to use the `wgpu` backend,
itt is preferable to have it as the deault backend given it's
superiority in performance in comparison to CPU.

### Typical design implementations and comparisons

This section offers various implmentations of models in p_vision as
they may be implemented from scratch using keras and pytorch_lightning.

`keras`

```python

from keras import Sequential, Input, Dense
model=Sequential()
model.add(Input(shape=(16,)))
model.add(Dense(8))
len(model.weights) # This yields the models weights provided its
                   # definition began with the Input layer definition.
```

A similar implementation on the rust side would be

```rust

/// The layers ought to be printable by default 
/// the layers also oought to be trainable by default


struct Sequential<T: trait_specific_to_layers> {}

impl<T> Sequential {
  pub fn add<T> (_layer: T) -> impl T {
    // check currebt structure of sequential if it has any layers and
    // add one if it is.
  }
}

impl LayerTrait for Sequential {}

struct Dense(_num: u8);

impl LayerTrait for Dense {}

struct Input {
  shape: (u8,u8);
}

impl LayerTrait for Input {}

```

Thus when exposing the model the following method o python
importation method remains the same, while description underneath
it is different

NOTE: I think it is way easier to have the model have a default backend
first before offering the ability to switch backends.

I.e

```p_vision

  model = Sequential(backend="Wgpu")
  model.add(Input(shape=(3,3)))
  model.add(Dense(8))
```

or

```p_vision

model = Sequential()
model.add()
.
.
.
model.backend = "Wgpu"
```
