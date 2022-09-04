---
### REMEMBER TO CHANGE ###
title: "Jax for Torch users"
date: 2021-12-20T11:30:03+00:00
tags: ["AI", "Jax", "Pytorch"]
description: "An introduction to Jax for Pytorch users and some tricks and patterns I've learned and used."
author: "Stone Tao"
# author: ["Me", "You"] # multiple authors
### REMEMBER TO CHANGE ###

# weight: 1
# aliases: ["/first"]

showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: false

canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/StoneT2000/blog/tree/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# Jax for Pytorch Users

I've been a pytorch user for years. It's simple, pythonic, and my code velocity has never been higher. While it was slower than tensorflow usually, I found this tradeoff to be more than necessary in order to do research at a good pace.

I only recently started to heavily use jax for my research and projects and it's a very different way of programming.

Transitioning to using Jax was difficult, but blog posts like [this one](https://sjmielke.com/jax-purify.htm) and repositories like [jaxrl](https://github.com/ikostrikov/jaxrl/) helped me a lot and I consolidate what I've learned and some tricks/patterns I've picked up through the process.

tldr; I talk about the core features and caveats of Jax, before diving into how to use Jax a bit more "Object Oriented" like Pytorch.

## The main course

In Jax, you just-in-time (jit) compile your functions to leverage [XLA](https://www.tensorflow.org/xla) and massively improve performance on GPU/TPUs. It further comes with incredibly easy and fast vectorization, enabling you to cleanly define your functions without batch dimensions (less shape wrangling!) while maintaining the benefits of batched operations. Finally, another core feature is easy and flexible differentiation.

But in order to leverage the power of Jax, **you must work functionally.** All jitted functions must be pure functions, meaning that they return the same outputs given the same inputs. As a corrollary, Jax safely assumes jitted functions work with inputs and outputs that do not change shape (and will complain if you use mismatching shapes).

There is a large list of things to be aware of when using Jax detailed in their documentation [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) with example code, but some of the most important ones are to deal with random number generation (RNG) and control flow and I paraphrase some of the documentation as well as provide some nifty ways to use them. GPU memory allocation is also a caveat that seems seldomly mentioned in the docs and I mention it a little later.

### RNG
Since everything in Jax is pure, to have any kind of randomness you need to pass in some kind of RNG key. In Pytorch, to generate random numbers we normally do this

```python
torch.manual_seed(0)
def do_something_random():
  return torch.randint(high=10, size=(1, )) # generate a random integer 0 to 9
do_something_random() # 4
do_something_random() # 9
do_something_random() # 3
```

But in Jax, we do this
```python
@jax.jit # jit compile the function, although not necessary
def do_something_random(key: PRNGKey):
  return jax.random.randint(key, shape=(1,), minval=0, maxval=10)
key = jax.random.PRNGKey(0)
do_something_random(key) # 2
do_something_random(key) # 2
do_something_random(key) # 2
```

The difference here is that in Pytorch and normal python, `do_something_random` has a side effect of changing the underlying Pytorch RNG with every call whereas in Jax it's pure. As a result Jax always returns the same value whereas Pytorch will return something random. Note that I jit compile the function here although since we are only generating random numbers it's not necessary. It's worth noting that a PRNGKey is simply an array of numbers, and so the jit-compiler can happily take PRNGKeys as inputs.

Now that we have made randomness "pure" how do we make it actually random? We split the `PRNGKey` into several new keys.

```python
@jax.jit
def do_something_random(key: PRNGKey):
  return jax.random.randint(key, shape=(1,), minval=0, maxval=10)
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(subkey)
do_something_random(subkey) # 6
key, subkey = jax.random.split(subkey)
do_something_random(subkey) # 3
key, subkey = jax.random.split(subkey)
do_something_random(subkey) # 0
```

More importantly, each subkey is a complete "fork" from the old key. Anything you do with that subkey will not impact any other key. A benefit of this feature is that you can seed entire subroutines by simply generate a subkey and passing that to the subroutine while you propagate the original key through the main code. A more succinct version of the above code would be to do `key, *subkeys = jax.random.split(subkey, 4)` instead

```python
@jax.jit
def do_something_random(key: PRNGKey):
  return jax.random.randint(key, shape=(1,), minval=0, maxval=10)
key = jax.random.PRNGKey(0)
# generate 4 keys, 1 is key, the remaining 3 are stored in subkeys
key, *subkeys = jax.random.split(subkey, 4)
do_something_random(subkeys[0]) # 6
do_something_random(subkeys[1]) # 3
do_something_random(subkeys[2]) # 0
```
### Control flow

In jit compiled functions, you can't do anything that would make it impossible for the compiler to determine the shape of the inputs and outputs. Shape here can represent not just array shapes, but also dictionaries and tuples of which individual elements are different array shapes.

The following functions would be problematic

```python
@jax.jit
def compute_sum(n: int):
  r = 0
  for _ in range(n):
    r += 1
  return r

@jax.jit
def absolute(pick: int):
  if pick < 0:
    return 
```


### Memory

One of the biggest ones is Jax preallocates your entire GPU memory. If you are someone working with a single GPU device usually, this can be annoying since you can't simultaneously run several scripts that use the GPU now. The solution is to set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and for more options see https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html

## Common errors

### Forgetting to jit



### In-place updates

Outside of jitted functions, unlike numpy arrays and torch tensors you can't perform in-place updates to data. In particular, Jax is very strict with this and the only time it will do an in-place update is when you are in a jitted function and the updated array was either initialized in the jitted function or was an inputted jax array.



- Add buffer donation

## Models / nn.Module

If you use Pytorch you will be very accustomed to writing your own models using the `nn.Module` class. Simply define the layers, initialize weights however you want, and then define a forward pass function.

Jax itself doesn't provide a neural network library, instead you can use [flax](https://github.com/google/flax) or [haiku](https://github.com/deepmind/dm-haiku). I'll present examples with flax although you can generally do the same with haiku as well.

Recall that for Jax to be fast, we must jit-compile functions. But a requirement of this is that the function is pure. A neural network function is effectively just its forward pass, parameterized by its own model parameters. 

In Pytorch you may be accustomed to writing modules as so
```python
from typing import Sequence
import torch
import torch.nn as nn
class MLP(nn.Module):
  def __init__(self, features: Sequence[int]):
    super().__init__()
    layers = []
    for j in range(len(features) - 2):
      layers += [nn.Linear(features[j], features[j + 1]), nn.ReLU()]
    layers += [nn.Linear(features[-2], features[-1])]
    self.mlp = nn.Sequential(*layers)
  def forward(self, x):
    return self.mlp(x)
model = MLP([10, 12, 8, 4])
batch = torch.ones((32, 10))
model(batch) # output is shape (32, 4)
```

Both the model parameters (`model.parameters()`) and forward pass (`model.forward`) are bundled together which is nice. Rarely do you need to have separation between these two components. On the other hand, with flax it looks like the following
```python
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(jax.random.PRNGKey(0), batch)
output = model.apply(variables, batch)
```
Now you have to keep around both the model parameters (`variables`) and the forward pass (`model.apply`) which can easily be confusing and make room for mistakes. This design in itself is in many ways better than pytorch since it enables more flexibility in defining functions and moving around parameters whenever you want. Moreover, you can still bundle the parameters and forward pass together like pytorch and I highly recommend using the following `Model` class approach to writing and maintaining neural networks in Jax.




## Gradients, Optimization

- discuss how to get gradients, 2nd order derivatives, kth order... (and compare with torch)
- discuss optax vs torch.optim

## Jax Ecosystem

To do anything more complicated with Jax, you need libraries. But isn't Jax *the* library? While Pytorch bundles almost everything you need into the `torch` package (including CUDA even), Jax maintains an ecosystem of mostly standalone packages for different features, with the main ones being [optax](https://github.com/deepmind/optax) for optimization, [haiku](https://github.com/deepmind/dm-haiku) for neural nets, and [chex](https://github.com/deepmind/chex) for testing. Pytorch is more of a monorepo, but even they have problems with maintainence and eventually made a separate `torch-vision` package, whereas the Jax ecosystem is a multirepo. In my opinion this offers an invaluable amount of flexibility to users and especially researchers for starting new projects. Choose what packages you need, and if there's a problem, you likely don't need to trace through 1000 API calls to figure out what's wrong.

For more information checkout [Deepmind's blog post](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research) on their Jax ecosystem.

<!-- I personally use [flax](https://github.com/google/flax) instead of haiku and  -->
<!-- For RL people, I highly recommend using [distrax](https://github.com/deepmind/distrax) as well for probability distributions.  -->

