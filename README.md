# simple_rl
Just some entry-level projects in reinforcement learning using open-ai gym.

This code was developed completely independently without reference to tutorials. Reward-shaping was avoided since this makes the model task specific. In future developments, this learner should be able to play many atari games, using only the final reward at the end of the session.

The default settings allow for the learner 'win' (balance the pole for 5 seconds) occasionally after about 10 minutes of training.

To get the dependencies,

`pip install requirements.txt`

To run the default learner,

`python cartpole.py`
