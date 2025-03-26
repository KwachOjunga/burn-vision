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



### Note


The models extracted by onnx2... are rather verbose and expose only the backend
as its object capable of being altered arbitrarily.
Now, for the package to be considered as even remotely flexible certain facrors have
to be met.
This proves true provided my assumptions of what it means to have a deep learning
architecture stand.

(a) a model's architecture retains its functional ability and utility
  and is not bound by particularities of its internal implementation.
(b) What defines the deep learning architecture is fundamentally the organization
  of neural network layers.


To have a model whose backend is configurable, input passed on from python's side may
be in form of a string or python enum that can be matched to yield the set backend on
the python's side.

```
  from typing import Enum
  from p_vision import googlenet
  
  backend = Enum {
    WGPUDEVICE,
    FUSIONJITBACKEND,
    PYTORCH,
    NDARRAY,
  }
  # i am uncertain of whether or not this is the correct implementation of a python enum

  def train_model(device : backend):
      mybackend = backend.WGPUDEVUCE
      model = googlenet(mybackend)
      return model.train(train_data, # and dataloading configurations)
      
```


On the backend, you just have to f8gue out how to ensure the parameter passed from python
will be received before anything else can be done.

It is very likely that py03's types allow for that differentiation in the functions exposed in
to the frontend.
