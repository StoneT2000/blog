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

Jax 

I only recently started to heavily use jax for my research and projects and it's a very different way of programming

## The main tid-bits

In Jax, you just-in-time (jit) compile your functions to leverage [XLA](https://www.tensorflow.org/xla) and massively improve performance on GPU/TPUs. It further comes with incredibly easy and fast vectorization, enabling you to cleanly define your functions without batch dimensions (less shape wrangling!) while maintaining the benefits of batched operations. Finally, another core feature is easy and flexible differentiation.

But in order to leverage the power of Jax, **you must work functionally.** All jitted functions must be pure functions, meaning that they return the same outputs given the same inputs. As a corrollary, Jax safely assumes jitted functions work with inputs and outputs that do not change shape (and will complain if you use mismatching shapes).

- add smth about memory allocation
XLA_PYTHON_CLIENT_PREALLOCATE=false
https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html

## Common errors

### Forgetting to jit



### In-place updates

Outside of jitted functions, unlike numpy arrays and torch tensors you can't perform in-place updates to data. In particular, Jax is very strict with this and the only time it will do an in-place update is when you are in a jitted function and the updated array was either initialized in the jitted function or was an inputted jax array.



- Add buffer donation

## Models / nn.Module

## Gradients, Optimization

- discuss how to get gradients, 2nd order derivatives, kth order... (and compare with torch)
- discuss optax vs torch.optim

## Jax Ecosystem

To do anything more complicated with Jax, you need libraries. But isn't Jax *the* library? While Pytorch bundles almost everything you need into the `torch` package (including CUDA even), Jax maintains an ecosystem of mostly standalone packages for different features, with the main ones being [optax](https://github.com/deepmind/optax) for optimization, [haiku](https://github.com/deepmind/dm-haiku) for neural nets, and [chex](https://github.com/deepmind/chex) for testing. Pytorch is more of a monorepo, but even they have problems with maintainence and eventually made a separate `torch-vision` package, whereas the Jax ecosystem is a multirepo. In my opinion this offers an invaluable amount of flexibility to users and especially researchers for starting new projects. Choose what packages you need, and if there's a problem, you likely don't need to trace through 1000 API calls to figure out what's wrong.

For more information checkout [Deepmind's blog post](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research) on their Jax ecosystem.

<!-- I personally use [flax](https://github.com/google/flax) instead of haiku and  -->
<!-- For RL people, I highly recommend using [distrax](https://github.com/deepmind/distrax) as well for probability distributions.  -->

