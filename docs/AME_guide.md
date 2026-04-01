# Attention-based Map Encoding: a Practical Tuning Guide

Author: Chong Zhang chong.zhang@ai.ethz.ch

Note: this guide contains instructions on reimplementing [AME-1](https://www.science.org/stoken/author-tokens/ST-2851/full) and [AME-2](https://arxiv.org/abs/2601.08485), two works on achieving generalized legged locomotion. It is based on not only my own engineering experience, but also many communications from other researchers and engineers. We plan to release archived codes after AME-2.5 release, which is expected to happen in late 2026.

### Citations

```bibtex
@article{he2025attention,
  title={Attention-based map encoding for learning generalized legged locomotion},
  author={He, Junzhe and Zhang, Chong and Jenelten, Fabian and Grandia, Ruben and B{\"a}cher, Moritz and Hutter, Marco},
  journal={Science Robotics},
  volume={10},
  number={105},
  pages={eadv3604},
  year={2025},
  publisher={American Association for the Advancement of Science}
}

@article{zhang2026ame,
  title={AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding},
  author={Zhang, Chong and Klemm, Victor and Yang, Fan and Hutter, Marco},
  journal={arXiv preprint arXiv:2601.08485},
  year={2026}
}
```
------


## AME-1
AME-1 is the first work of the AME series, published on *Science Robotics*. The free-access link is provided above in hyperlink. At its core, it uses attention to select footholds, replacing the model-based or heuristic planning methods with an end-to-end module, while achieving generalization to unseen terrains. It uses the velocity-tracking formulation, and solves scalability during multi-terrain training.

Our video results: https://www.youtube.com/watch?v=GUgwB6WxcFo  
Independent reimplementation from Shanghai Innovation Institute on Unitree G1: https://www.bilibili.com/video/BV1aQqLBWEcQ

### Before you try it
A well-established sim-to-real pipeline is expected. AME-1 does not solve sim-to-real transfer, or motion shaping. It can slightly improve exploration on sparse terrains, but do not give high hope to this.

### Terrain engineering
Having the sparse terrains like stepping stones is a necessity. We build the stepping stones with curriculum in the paper [Learning Agile Locomotion on Risky Terrains](https://arxiv.org/abs/2311.10484), which is further used in works like [BeamDojo](https://arxiv.org/abs/2502.10363), [Walking with Terrain Reconstruction](https://arxiv.org/abs/2409.15692), [MARG](https://arxiv.org/abs/2509.20036). Without the stepping stones, the attention is an overkill and can easily fall into bad local minima that can perfectly solve dense terrains (like stairs).

In the paper, there is a second-stage finetuning. The terrain part is not something that should be rigidly followed -- it just refines the attention pattern, and one can add whatever terrains they want without having a curriculum. The randomization part helps robustness. Adding drift randomization of maps also enforces precise footholds without reward shaping, and compensates for real-world mapping drifts and delays.

### Symmetry augmentation
This can be verified first with MLP before using AME, and can also be added after implementing AME. It helps improve the motion style a lot. For terrain map observation symmetry, imagine the robot is walking on a mirrored terrain, so only mirror the z values and keep old x & y values.

### AME-1 network implementation
We use heightmap cell coordinates in the base-yaw frame as "positional embedding" of the cell, and use CNNs that keep the shape as "tokenizer" to extract local features. So each cell of the heightmap becomes a token with local features and position information.

We use 64-dim 16-head MHA. We found that it doesn't work with fewer heads. We share the encoder for actor and ciritc -- intended to save memory, and we later found this helps speed up convergence, given that the critic loss has better supervision signals.

### Learning parameters
We find it is good to have 4096 environments plus symmetry augmentation, with 24 steps per env to trigger policy updates. Then the minibatch things are just set to make the thing fit into the GPU. This can be tuned based on available GPU capacity.

### Training humanoids
Motion shaping can be important, especially for arm and ankle movements. Tracking velocity on the torso is recommended over pelvis, as the waist can do active stabilization.

### Training smaller quadrupeds
To improve the stability for small quadrupeds (like Go2), making the terrains easier and denser can help. Due to the size limitation, it might be hard for these dogs to traverse big gaps and stones.

### Training quadrupeds with Go1/Spot-like joint configuration
These robots can generate stronger bursts of motion than ANYmal because of their leg configuration, but they are also more prone to exploiting shank contact. A practical recommendation is to keep the terrain sparse enough that shank collisions do not assist traversal, increase the associated penalties, and, when needed, slightly adjust the colliders to make shank exploitation useless for traversal.

-----

## AME-2 Policy Learning

AME-2 is a systematic "upgrade" of AME-1 to make the controller more agile, scale to more terrains, achieve robot-agnostic rewards, and solve perception uncertainties with neural mapping. We separate the issues of training a teacher policy and building the neural mapping pipeline, and provide a step-by-step roadmap. Upon having these two, training the student is also trivial.

A principle: after each step, verify the thing still works.

### Step 1: using the goal reaching formulation
This is quite straightforward to do in IsaacLab -- Just use the exisitng goal command here, and use the flat patch sampling function to help sample valid goals. It would still be worth implementing a new command based on that to customize command interface and resample rules.

At the same time, implement the task rewards -- easy for an AI agent today! Then you can verify an MLP goal reaching policy on flat/rough terrains.

### Step 2: rewards, termination, curriculum, in an easy way
Implement those rewards, termination, curriculum setups should be easy with an agent now. But for safety, you can use large task rewards and relaxed termination conditions, otherwise you cannot tell if the learning part is off or the parameters are bad during debugging.

My own case: I directly used acceleration termination conditions from legged gym to Isaaclab, and found the policy learning is much slower. It turns out that, in Isaaclab, the accelerations can be much larger, so the env just keeps terminating. Do not let this slow down the development.

### Step 3: actor critic implementation
Now it is the time to implement the policy and the critic! For the policy, it is similar to AME-1, except that we have 32 heads (96 dim in total), and a global feature. For the critic, we use MoE with 16 MLP experts fused via softmax. We also did symmetry augmentation only for the critic losses because optimizing the critic is computationally cheap.

### Step 4: add terrains
With the AME-2 policy, multi-terrain learning should be stable. That said, start from easy terrains that the robot can 100% pass, but also keep stepping stones.

### Step 5: add domain randomization
Now we can challenge the learning with domain randomization. Ideally, this should only slow down the learning a bit.

### Step 6: refining
After verifying the whole stack can train well, the reward weights, termination conditions, terrain parameters can all be tuned to the paper's version. That said, due to potential differences between robots and simulators, this can be better done in a cautious way: do not add everything all at once.

### Extra step: motion tuning
For humanoids, I found it is still necessary to add two rewards: 1) upper-body joint deviation penalty for arm pose, and 2) ankle orientation penalty. The effects can be seen in: [post1](https://x.com/ChongZitaZhang/status/2032841021067804998), [post 2](https://x.com/ChongZitaZhang/status/2028474390334054739), [post 3](https://x.com/ChongZitaZhang/status/2031728176494190856).
Intuitively, I also make a terrain-aware weight here (0.1x weight for terrains with big elevation diffs).

------
## AME-2 Neural Mapping

### tbd: I have not transferred this into IsaacLab. Besides, the old implementation ignores self mesh (with filter on real robot), but now I am think about training with that.