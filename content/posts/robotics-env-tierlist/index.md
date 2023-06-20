---
### REMEMBER TO CHANGE ###
title: "Robot Environments Tierlist"
date: 2023-05-16T11:30:03+00:00
tags: ["AI", "Robotics"]
description: "I have a lot of frustration in figuring out what the state of the art is for a particular environment and what environments should be tested on and why certain methods do well on one complicated looking environment but poorly on supposedly easy looking environments. To address that, I talk about the various characteristics of robotics environments and construct a robotics environments tierlist off of that."
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
