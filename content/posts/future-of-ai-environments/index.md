---
### REMEMBER TO CHANGE ###
title: "The Future of AI Environments"
date: 2023-07-12T11:30:03+00:00
tags: ["AI", "Robotics", "Games", "Research"]
description: "I have a lot of frustration in figuring out what the state of the art is for a particular environment and what environments should be tested on and why certain methods do well on one complicated looking environment but poorly on supposedly easy looking environments. To address that, I talk about the various characteristics of robotics environments and construct a robotics environments tierlist off of that."
author: "Stone Tao"
# author: ["Me", "You"] # multiple authors
### REMEMBER TO CHANGE ###

# weight: 1
# aliases: ["/first"]


showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
math: true

canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
editPost:
    URL: "https://github.com/StoneT2000/blog/tree/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

In the world of embodied AI research, which includes applications of reinforcement learning (RL), imitation learning, open-ended learning etc. it's often dominated by the notion of an environment.

A common vision in the world of embodied AI. Many believe a path to achieve human-like intelligence (or perhaps beyond) is to follow the same path humans take to learn. 

1. Pre-existing DNA and matter
2. Life long learning


One's DNA can be analogous



---

So we have a general analgoy to human learning, can we replicate this in training AI agents? There are a number of things we can mimic, but a number of what I call "practicalities" that we must understand before pushing for replicating human learning.

I will discuss primarily around the interaction component of life long learning. There are many other exciting directions to consider from self-supervised learning approaches, representation learning...


## Practicality

As we observe with LLMs, things tend to work better when we scale up. The sort of 3 core scaling modes are data, model size, and training time. 

The most proven scaling mode is data, more specifically data quantity and diversity. But scaling RL on embodied agents proves extremely costly and difficult for most applications (robotics, games)


When looking at purely online RL, where an agent has no access to data from elsewhere, it's data source is the environment it interacts in. Can we scale this type of data? This is heavily dependent on the problem you to talk to and I'll discuss my experience in both games and robotics

### Games


### Robotics

The ultimate goal of robotics is real world deployment of a highly capable robot that can

1. Perform a wide range of tasks
2. Learn new tasks just like a physical human. 

We are currently still working on 1 and the most impressive results so far like 
Google's [RT-2](https://robotics-transformer2.github.io/) still are major steps away from a capable robot. Right now RT-2 can understand language and what it is looking at, but its robotics capabilities remain limited to simple pick and place (and of predominantly simple objects). RT-2 remains incapable of e.g. setting up an entire dinner table (Long horizon task, fine/precise manipulation of silverware), or . In my opinion, RT-2 shows a strong proof of concept of how to solve the high-level planning problem, going from a human-understand interface of language to a sequence of tasks to complete.

To my understanding the major unsolved problems of 1 (perform a wide range of tasks) primarily resides on manipulation, creating a robot hand/gripper that can autonomously grasp and control any object like a human. Right now in simulation, training an agent via RL to pick up a cube, or plug a charger, isn't trivial without using dense rewards or a good amount of pre-collected data.


#### Robot Simulation

If you have done any robot learning or robotics, you most likely have heard of [Mujoco](), [Brax], [Isaac Sim](), and perhaps less likely my own lab's [SAPIEN/ManiSkill2]() (namely because Mujoco, Brax, and Isaac are from industry, SAPIEN remains really the only top simulator made by academic labs).

RL is a fascinating prospect but you need a ton of data. Isaac Sim and Brax present a fresh approach to this by enabling simulation of 1000s of instances of an environment on a single GPU. Typically parallelization is achieved by parallelizing across the CPU...

While Isaac Sim and Brax are exciting and seem like a no brainer solution, GPU parallelized physics simulation for robotics brings some problems. After length discussion with my friend [Fanbo]() (Maintainer of SAPIEN), GPU parallelized code brings parallelization but cannot make accurate simulations. The same problem plagues approaches that attempt to speed up CPU based simulation, with these approaches often decreasing the simulation fidelity and resulting in obviously non sim2real transferrable videos like this:

[Insert D4RL video]



-----

Ok, *\<deep breath\>*

In my time doing robotic learning research I've been frustrated. Frequently. By how there are both 100s of different environments in robotics, but very little consolidation amongst them in terms of 

- environment interface ([Deepmind Environment interface](https://github.com/deepmind/dm_env), [Gym/Gymnasium](https://github.com/Farama-Foundation/Gymnasium))
- choice of control frequencies, horizons, and reporting styles ([Deepmind Control Suite](https://github.com/deepmind/dm_control) uses 1000 steps while [ManiSkill 2](https://github.com/haosulab/ManiSkill2) uses 200 steps, and Deepmind typically reports metrics in seconds as seen in these papers: [Haarnoja et al., (2023)](https://arxiv.org/abs/2304.13653), [Adaptive Agents Team, (2023)](https://arxiv.org/abs/2301.07608))
- physics engines ()

The primary question I often have is, which environment do I test on? And what kind of performance should I expect when using algorithm from paper A vs an algorithm from paper B when applied to a new environment?

To properly structure this discussion, I will make a few assumptions. When comparing environments I will compare them in terms of well a standard online RL algorithm can perform, e.g. [PPO]() or [SAC]().


Lets go to basics. What is different between any two environments? An AI environment is typically defined as a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process), consisting of 4 core elements, the state space $S$, the action space $A$, the dynamics function $P$, and the reward function $R$.

It's already well known that a high dimension state or action space will cause any machine learning algorithm to cripple under the might of the curse of dimensionality.

## Defining Characteristics

To understand 


Transition skills ()
Task Horizon
Dynamics involved
Randomizations

## References

[1] 

[2] 
