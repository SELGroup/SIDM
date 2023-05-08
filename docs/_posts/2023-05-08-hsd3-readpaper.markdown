
---
layout: post
title: "read hsd3"
date: 2023年5月8日22点38分
categories: paper HRL skill
---
## concept
### Concepts in Hierarchical Skill Learning

Imagine a bipedal robot learning to navigate through a complex environment filled with obstacles. The robot needs to learn various skills, such as walking, turning, and climbing, to efficiently explore and complete tasks in the environment.

#### Skill Policy and Shared Policy

Skill policies control specific aspects of the robot's behavior. Shared policies represent multiple skills or behaviors with a single policy. In this example, a skill policy might control the movement of the left leg, while the shared policy could represent both the left and right leg movements, as well as more complex behaviors like turning or climbing.

#### Hierarchy of Skills

Skills are organized into a hierarchy based on their complexity. In this case, simpler skills like moving individual legs form the lower levels of the hierarchy, while more complex skills like turning and climbing are at higher levels.

#### High-Level Control Signal

The high-level control signal is provided by a higher-level policy that guides the robot's overall behavior. In our example, the high-level control signal might instruct the robot to move towards a specific target or avoid an obstacle, which then influences the low-level skill policies, such as leg movements and turning.

#### Robot Configuration

The robot's configuration refers to its current state, including positions, orientations, and joint angles. In this scenario, the robot's configuration might include the positions of its legs, torso, and other body parts.

The paper presents a hierarchical skill learning framework that allows the agent to learn skills of varying complexity without prior knowledge of the task. This framework helps the agent effectively balance generality and specificity, leading to better performance in diverse, sparse-reward tasks for bipedal robots.

## method

### personal (GPT4's) notes

The authors propose an approach that first acquires low-level policies capable of carrying out a useful set of skills in an unsupervised manner, meaning without reward signals from the main tasks of interest. These skills are then used in a hierarchical reinforcement learning setting.

The pre-training environment consists only of the robot, while in downstream tasks, additional objects may be present. These objects could be fixed (e.g., obstacles) or only indirectly controllable (such as a ball).

#### Low-Level Skills and Goal Spaces

The usefulness of a skill policy comes from providing additional structure for effective exploration. To provide benefits across a wide range of tasks, skills need to support both fast, directed exploration (e.g., locomotion) and precise movements (e.g., lift the left foot while bending down). The authors propose short-horizon, goal-directed low-level policies trained to achieve target configurations of robot-level state features, such as the position or orientation of its torso or relative limb positions.

This approach allows defining a hierarchy of skills by controlling single features and their combinations, resulting in varying amounts of control exposed to high-level policies. Each skill is trained to reach goals in a goal space defined over a set of features, yielding policies without task-specific rewards. These policies rely solely on state features related to the specific robot and the prescribed hierarchy of goal spaces.

    A feature, in the context of this work, refers to a specific property or characteristic of the robot's state. State features can include the position, orientation, or relative limb positions of the robot. These features help describe the current situation or configuration of the robot in the environment.

    In this approach, low-level policies are trained to achieve target configurations of robot-level state features, such as reaching a certain position or orientation. The authors define a hierarchy of skills by controlling single features and their combinations, which results in different levels of control exposed to high-level policies.

    A goal space is defined over a set of features. In this case, the goal space represents a collection of possible goals that can be achieved by manipulating these specific features. For instance, a goal space could be defined over the position and orientation features, meaning that the goals within this space would be related to reaching different positions and orientations.

    These low-level policies (skills) do not rely on task-specific rewards; instead, they are trained based on the state features of the robot and the prescribed hierarchy of goal spaces. By focusing on state features and goal spaces, the approach enables the robot to learn a variety of skills that can be useful across a wide range of tasks. The high-level policy then selects and combines these low-level skills to accomplish the specific tasks it is given.


#### High-Level Policy and Task-Specific Features

On downstream tasks, high-level policies operate with a combinatorial action space. State spaces are enriched with task-specific features containing information regarding additional objects in the downstream task. This enriched state space is only available to the high-level policy. The high-level policy is modeled hierarchically with two policies that prescribe a goal space and a goal, respectively.

	1. "Prescribe" here means that the high-level policy determines or selects the goal space and goal to be used by the low-level policies. It chooses the appropriate goal space and goal based on the current state and the task requirements.
	
	2. Action space refers to the set of all possible actions that an agent (in this case, the robot) can take in a given environment. In the context of this work, "operate with a combinatorial action space" means that the high-level policy chooses a combination of low-level skills (goal space) and goals from the available options. This allows the high-level policy to create diverse and complex actions by combining the skills and goals it has access to.
	
	3. The high-level policy can be seen as a decision-making mechanism that directs the low-level policies (skills) to achieve specific tasks. It is modeled hierarchically, with two components: one that selects a goal space (which skill to use) and another that selects a specific goal within that goal space. By doing so, the high-level policy can dynamically switch between different skills and goals as needed for the given task, while the low-level policies focus on executing the chosen skill to reach the desired goal. The high-level policy's primary function is to adapt and combine the low-level skills in a way that allows the robot to effectively perform various tasks.

#### Temporal Abstraction and HSD-3

With a task- and state-dependent policy, it is possible to not only select the required set of skills for a given environment but also switch between them within an episode. In this three-level hierarchy, the higher-level policy explores the hierarchy of goal spaces and dynamically trades off between generality and specificity. Temporal abstraction is obtained by selecting new high-level actions at regular intervals but with a reduced frequency compared to low-level actions. The resulting learning algorithm is called HSD-3, emphasizing the three different levels of control obtained after the hierarchical skill discovery phase.

    "Switch between them within an episode" refers to the ability of the high-level policy to change the selected skill and goal as needed during a single episode of the task being performed. An episode typically refers to a single run or attempt at solving a task, starting from an initial state and ending when the task is completed or a termination condition is met. By switching between different skills and goals during an episode, the high-level policy can adapt to changing requirements of the task and environment.
        
    "Temporal abstraction" is a concept in hierarchical reinforcement learning where decisions can be made at different time scales. In the context of this work, it means that the high-level policy selects new actions (which skill to use and which goal to pursue) at a slower frequency compared to the low-level policies. This allows the high-level policy to make long-term decisions while the low-level policies focus on the shorter-term execution of those decisions. By having different levels of control operating at different time scales, the system can achieve more efficient learning and better performance in complex tasks.

### formulization
In the original text, the following mathematical symbols are used:

1. $\left\{\pi_F^{\text {lo }}: \mathcal{S}^{\mathrm{p}} \times \mathcal{G}^F \rightarrow A\right\}_{F \in \mathcal{F}}$: This notation represents a set of low-level goal-directed skill policies ($\pi_F^{\text {lo }}$) for each feature set $F$ in the collection of feature sets $\mathcal{F}$. Each policy takes as input the proprioceptive observations ($\mathcal{S}^{\mathrm{p}}$) and a goal ($\mathcal{G}^F$) from the goal space associated with the specific feature set $F$. The policy outputs an action ($A$) to be executed by the robot.

2. $\pi^{\mathrm{f}}: S \rightarrow \mathcal{F}$: This notation defines the high-level policy that takes the current state ($S$) as input and selects a goal space from the collection of goal spaces ($\mathcal{F}$). The output is a specific feature set $F$ that the policy determines to be the most appropriate for the current state.

3. $\pi^{\mathrm{g}}: S \times \mathcal{F} \rightarrow \mathcal{G}^F$: This notation defines the second part of the high-level policy that takes the current state ($S$) and the selected feature set $F$ as input and outputs a specific goal ($g$) from the goal space associated with that feature set ($\mathcal{G}^F$).

These mathematical symbols help to define the structure of the high-level and low-level policies and their relationships in a concise manner. They indicate the inputs and outputs of each policy and demonstrate how the policies interact in the hierarchical reinforcement learning framework.