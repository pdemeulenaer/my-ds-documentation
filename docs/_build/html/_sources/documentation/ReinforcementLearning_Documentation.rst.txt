=====================================
 Reinforcement Learning
=====================================

.. topic:: Introduction

    Blabla

Lecture to Deep Reinforcement Learning: https://simoninithomas.github.io/Deep_reinforcement_learning_Course/

https://www.youtube.com/watch?v=gCJyVX98KJ4&list=PLQLZ37V8CnUTdIoJJdvmmFoQJntZ9dp5Q

https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419



RL core idea: The idea behind Reinforcement Learning is that an agent will learn from the environment by interacting with it and receiving rewards for performing actions.

Ex: 
- the child and fire: close = fun and warm (positive reward, +1). Too close = painful and dangerous (negative reward, -1)
- the dog and the master: the dog does not "know" the master's language at start. He learns as he interacts with the master. If he does an action that coincides with the master's will, then positive reward, otherwise no negative (or no).
- any kind of trial and error process, like learning to ride a bike, is a RL process.

This is a major difference between RL and Supervised (or unsupervised) ML: in SL you feed the data, train on it and then apply on new, unknown data. But in RL we have no knowledge at start, we gather knowledge as we interact with the environment. 

This makes the RL an ONLINE process, we learn as we play, in contrast with SL where we train once and use afterwards (and if we want to update the knowledge of our SL model, then we need to retrain it...). For example, after each episode of tic-tac-toe game, an agent gets better at it. 



Main RL architectures to solve RL problems:

- Q-learning, 
- Deep Q-learning, 
- Policy Gradients,
- Actor Critic, 
- PPO.


Definitions
------------------

- Agent

- State

- Environment

- Action

- Reward

[insert image RL_principle]


Episodic or Continuing tasks: episodic is typically a game with a clear end (game over), while an continuous task does not have a terminal state, like for example the stock trading; the agent keeps running, improving, over and over. 

Central idea: maximize the expected cumulative rewards!

[insert Maximize_expected_cumulative_reward]

Value Function
-------------------

The value function is a function that tells us the maximum expected future reward the agent will get at each state.

The value of each state is the total amount of the reward an agent can expect to accumulate over the future, starting at that state.

[insert RL_Value]

Example:

[insert Maze_example_Value]

In this maze example, at each step we will take the biggest value: -7, then -6, then -5 (and so on) to attain the goal.

Other example:

The agent will use the value function to select which state to choose at each step. The agent takes the state with the biggest value.

https://devblogs.nvidia.com/deep-learning-nutshell-reinforcement-learning/

[Insert image Value_Function_game]

This framework means that each given state’s value is influenced by all the values of the states which are easily accessible from that state. Figure 1 shows an example world where each square is a state and S and G are the start and goal states, respectively, T squares are traps, and black squares are states which cannot be entered (big boulders blocking the way). The goal is to move from state S to state G. If we imagine this to be our racetrack, then over time we create a “ramp” from the start state to the goal state, so that we just have to go along the direction which is steepest in order to find the goal. This is the main goal of the value function: to provide an estimate of the value of each state, so that we can make step-by-step decisions which lead to a reward.


Discounting Factor
----------------------

Even though the value function handles competing rewards, it also depends on a parameter called the discounting factor that defines its exact behavior by limiting how much an agent is affected by rewards which lie in the far future. This is done by multiplying the partial reward by the discounting factor whenever a state change is made. So with a discounting factor of 0.5, the original reward will be one eighth of its value after only 3 state changes, causing the agent to pursue rewards in close proximity rather than rewards which are further away. Thus the discounting factor is a weight that controls whether the value function favors prudent or greedy actions.

For example, a race car might get high reward by going fast, so a very greedy decision with discount factor close to zero might always go as fast as possible. But a more prudent agent knows that going as fast as possible is a bad idea when there is a sharp turn ahead, while slowing down at the right moment may yield a better lap time later, and thus a higher reward. To have this foresight, the agent needs to receive the reward for going faster later, so a discounting factor closer to 1 may be better.

The value function’s discounting factor ensures that rewards that are far away diminish proportionally to the distance or steps it takes to reach them. This factor is usually set to a value which values long-term rewards highly, but not too high (for example the reward decays by 5% per state transition).

The algorithm learns the specific partial reward value of each state of the value function by first initializing all states to zero or their specific reward value, and then searching all states and all their possible next states and evaluating the reward it would get in the next states. If it gets a (partial) reward in the next state, it increases the reward of the current state. This process repeats until the partial reward within each state doesn’t change any more, which means that it has taken into account all possible directions to find all possible rewards from all possible states. You can see the process of this value iteration algorithm in Figure 1.

[insert Mouse_cheese_cat_game]

Example: Let say your agent is this small mouse and your opponent is the cat. Your goal is to eat the maximum amount of cheese before being eaten by the cat. As we can see in the diagram, it’s more probable to eat the cheese near us than the cheese close to the cat (the closer we are to the cat, the more dangerous it is). As a consequence, the reward near the cat, even if it is bigger (more cheese), will be discounted. We’re not really sure we’ll be able to eat it.

To discount the rewards, we proceed like this:

We define a discount rate called gamma. It must be between 0 and 1.

- The larger the gamma, the smaller the discount. This means the learning agent cares more about the long term reward.
- On the other hand, the smaller the gamma, the bigger the discount. This means our agent cares more about the short term reward (the nearest cheese).

[insert Discount_definition]

To be simple, each reward will be discounted by gamma to the exponent of the time step. As the time step increases, the cat gets closer to us, so the future reward is less and less probable to happen.

Hence now will try to maximize the discounted cumulative rewards. 


How to learn? Monte-Carlo vs Temporal Difference Learning
-------------------------------------------------------------------

- Collecting the rewards at the end of the episode and then calculating the maximum expected future reward: Monte Carlo Approach
- Estimate the rewards at each step: Temporal Difference Learning

Monte Carlo: (ex: maze, tic-tac-toe game,...)

When the episode ends (the agent reaches a “terminal state”), the agent looks at the total cumulative reward to see how well it did. In Monte Carlo approach, rewards are only received at the end of the game.

Then, we start a new game with the added knowledge. The agent makes better decisions with each iteration.

[insert Monte_Carlo_learning]

If we take the mouse maze environment:

- We always start at the same starting point.
- We terminate the episode if the cat eats us or if we move > 20 steps.
- At the end of the episode, we have a list of State, Actions, Rewards, and New States.
- The agent will sum the total rewards Gt (to see how well it did).
- It will then update V(st) based on the formula above.
- Then start a new game with this new knowledge.

By running more and more episodes, the agent will learn to play better and better.

Temporal Difference Learning : learning at each time step

TD Learning, on the other hand, will not wait until the end of the episode to update the maximum expected future reward estimation: it will update its value estimation V for the non-terminal states St occurring at that experience.
This method is called TD(0) or one step TD (update the value function after any individual step).

[insert Temporal_difference_learning]

TD methods only wait until the next time step to update the value estimates. At time t+1 they immediately form a TD target using the observed reward Rt+1 and the current estimate V(St+1).

TD target is an estimation: in fact you update the previous estimate V(St) by updating it towards a one-step target.


Exploration/Exploitation trade off
----------------------------------------

- Exploration is finding more information about the environment.
- Exploitation is exploiting known information to maximize the reward.

RL approaches
--------------------

- Value based
- Policy based


The Policy Function
-----------------------

The policy function represents a strategy that, given the value function, selects the action believed to yield the highest (long-term) reward. Often there is no clear winner among the possible next actions. For example, the agent might have the choice to enter one of four possible next states A,B,C, and D with reward values A=10, B=10, C=5 and D=5. So both A and B would be good immediate choices, but a long way down the road action A might actually have been better than B, or action C might even have been the best choice. It is worth exploring these options during training, but at the same time the immediate reward values should guide the choice. So how can we find a balance between exploiting high reward values and exploring less rewarding paths which might return high rewards in the long term?

Note that the policy and value functions depend on each other. Given a certain value function, different policies may result in different choices, and given a certain policy, the agent may value actions differently. Given the policy “just win the game” for a chess game, the value function will assign high value to the moves in the game which have high probability of winning (sacrificing chess pieces in order to win safely would be valued highly). But given the policy “win with a large lead or not at all”, then the policy function will just learn to select the moves which maximize the score in the particular game (never sacrifice chess pieces). These are just two examples of many. If we want to have a specific outcome we can use both the policy and value functions to guide the agent to learn good strategies to achieve that outcome. This makes reinforcement learning versatile and powerful.

We train the policy function by (1) initializing it randomly—for example, let each state be chosen with probability proportional to its reward—and initialize the value function with the rewards; that is, set the reward of all states to zero where no direct reward is defined (for example the racetrack goal has a reward of 10, off-track states have a penalty of -2, and all states on the racetrack itself have zero reward). Then (2) train the value function until convergence (see Figure 1), and (3) increase the probability of the action (moving from A to B) for a given state (state A) which most increases the reward (going from A to C might have a low or even negative reward value, like sacrificing a chess piece, but it might be in line with the policy of winning the game). Finally, (4) repeat from step (1) until the policy no longer changes.

In policy-based RL, we want to directly optimize the policy function π(s) without using a value function.

The policy is what defines the agent behavior at a given time.


In policy-based RL, we want to directly optimize the policy function π(s) without using a value function. The policy is what defines the agent behavior at a given time.

[insert RL_policy]

We learn a policy function. This lets us map each state to the best corresponding action. We have two types of policy:

- Deterministic: a policy at a given state will always return the same action.
- Stochastic: output a distribution probability over actions.

[Stochastic_policy]

The Q-function
--------------------

We’ve seen that the policy and value functions are highly interdependent: our policy is determined mostly by what we value, and what we value determines our actions. So maybe we could combine the value and policy functions? We can, and the combination is called the Q-function.

The Q-function takes both the current state (like the value function) and the next action (like the policy function) and returns the partial reward for the state-action pair. For more complex use cases the Q-function may also take more states to predict the next state. For example if the direction of movement is important one needs at least 2 states to predict the next state, since it is often not possible to infer precise direction from a single state (e.g. a still image). We can also just pass input states to the Q-function to get a partial reward value for each possible action. From this we can (for example) randomly choose our next action with probability proportional to the partial reward (exploration), or just take the highest-valued action (exploitation).

However, the main point of the Q-function is actually different. Consider an autonomous car: there are so many “states” that it is impossible to create a value function for them all; it would take too long to compute all partial rewards for every possible speed and position on all the roads that exist in on Earth. Instead, the Q-function:
- (1) looks at all possible next states that lie one step ahead and 
- (2) looks at the best possible action from the current state to that next state. 
So for each next state the Q-function has a look-ahead of one step (not all possible steps until termination, like the value function). These look-aheads are represented as state-action pairs. For example, state A might have 4 possible actions, so that we have the action pairs A->A, A->B, A->C, A->D. With four actions for every state and a 10×10 grid of states we can represent the entire Q-function as four 10×10 matrices, or one 10x10x4 tensor. See Figure 3 for a Q-function which represents the solution to a grid world problem (a 2D world where you can move to neighboring states) where the goal is located in the bottom right corner.

[insert image Q_function_grid_problem]

Important: the behaviour is INDEPENDENT from the starting position!


Q-Learning
--------------

To train the Q-function we initialize all Q-values of all state-action pairs to zero and initialize the states with their given rewards. Because the agent does not know how to get to the rewards (an agent can only see the Q-value of the next states, which will all be zero) the agent may go through many states until it finds a reward. Thus we often terminate training the Q-function after a certain length (for example, 100 actions) or after a certain state has been reached (one lap on a racetrack). This ensures we don’t get stuck in the process of learning good actions for one state since it’s possible to be stuck in a hopeless state where you can make as many iterations as you like and never receive any noticeable rewards.

[Insert Q_function_learning image]

Learning the Q-function proceeds from end (the reward) to start (the first state). Figure 4 depicts the grid world example from Figure 1 in terms of Q-learning. Assume the goal is to reach the goal state G in the smallest number of steps. Initially the agent makes random moves until it (accidentally) reaches either a trap or the goal state. Because the traps are closer to the start state the agent is most likely to hit a trap first, but this pattern is broken once the agent stumbles upon the goal state. From that iteration on the states right before the goal state (one step look-ahead) have a partial reward and since the reward is now closer to the start the agent is more likely to hit such a rewarding state. In this way a chain of partial rewards builds up more quickly the more often the agent reaches a partial reward state so that the chain of partial rewards from goal to start state is quickly completed (see Figure 4).


