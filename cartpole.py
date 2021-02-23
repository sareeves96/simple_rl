import tensorflow as tf
import gym
import numpy as np


# discount rewards so that actions close to the end of the game have a larger weight
# a larger gamma makes early actions matter more
def discount(rw, gamma=0.9):
    # weight individual rewards with an exponential decay function
    # since the magnitude should be largest close to the end of the session, apply the weights in reverse order
    weights = np.array([gamma**(rw.shape[0]-i-1) for i in range(rw.shape[0])])
    discounted = tf.convert_to_tensor(weights, dtype=tf.float32) * rw

    return discounted


class Model(object):

    def __init__(self,
                 activation='elu',
                 layers=1,
                 size=4,
                 gamma=0.95,
                 batch_size=20,
                 optimizer=tf.keras.optimizers.Adam(0.05),
                 stop=500
                 ):

        # specify the model architecture
        # a simple model suffices for such a low dimension, low complexity case
        self.model = tf.keras.Sequential(
              [tf.keras.layers.Input(shape=env.observation_space.shape)]
              + [tf.keras.layers.Dense(size, activation=activation) for _ in range(layers)]
              + [tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax, use_bias=False)]
        )
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.iteration = 0
        self.stop = stop
        self.display = False

    def run(self):
        all_obs = []
        all_ac = []
        rw = 0
        observation = env.reset()

        while True:
            if self.display:
                env.render()
            # get the distribution over action space from the model
            action_dist = self.model.predict(tf.convert_to_tensor(tf.expand_dims(observation, 0)))[0]
            # sample from the action space: compromise between exploration and exploitation
            action = int(np.random.choice(np.arange(len(action_dist)), p=action_dist))
            # get the information about the state of the system following the action
            observation, reward, done, info = env.step(action)
            all_obs.append(observation)
            all_ac.append(action)
            rw += reward
            # done specifies that the session is over, usually due to a win or loss
            if done:
                break
        env.close()
        self.iteration += 1

        # return the observations and rewards (the reward will be discounted later)
        return all_obs, all_ac, rw

    def compute_loss(self, obs, ac, rw):
        # reproduce the action probabilities that were predicted on this step so that we can calculate the gradient
        # this is the second time the action probabilities are calculated and it may be possible to recode this better
        y_pred = self.model(obs)
        y_true = ac

        # increase the probability of actions that led to rewards, decrease the converse
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # element-wise multiplication of discounted reward
        rw_weighted_loss = tf.math.multiply(loss, rw)
        return rw_weighted_loss

    def batch_train(self):
        obs_list, ac_list, rw_list = [], [], []

        for _ in range(self.batch_size):
            obs, ac, rw = self.run()
            obs_list.append(obs)
            ac_list.append(ac)
            rw_list.append(rw)

        mean_rw = np.mean(rw_list)
        std_rw = np.std(rw_list)
        print(f'\nMean reward after {self.iteration} sessions: ', mean_rw, '  stdev ', std_rw, '\n')
        if mean_rw >= self.stop:
            print('Model optimized!')
            return 0

        # normalize rewards, punishing worst performing session decisions and rewarding best performing ones
        rw_norm = [(r - mean_rw) / std_rw for r in rw_list]
        # expand each session reward so that each frame's action is initially given the same total reward
        rw_tensors = [tf.ones(shape=len(obs_list[i])) * rw for i, rw in enumerate(rw_norm)]
        # discount rewards so that actions closer to the end of the session get more weight
        rw_discount = [discount(rw) for rw in rw_tensors]
        # list of observations
        obs_tensors = [tf.convert_to_tensor(obs, dtype=tf.float32) for obs in obs_list]
        # list of actions
        ac_tensors = [tf.one_hot(ac, depth=env.action_space.n) for ac in ac_list]

        gradients = []
        # for each session of observations and discounted rewards
        for obs, act, rw in zip(obs_tensors, ac_tensors, rw_discount):
            # compute the gradient of the selected actions with respect to the observations of the environment
            with tf.GradientTape() as tape:
                loss = self.compute_loss(obs, act, rw)
            g = tape.gradient(loss, self.model.trainable_variables)
            # collect all the gradients, instead of applying them at each step which would give inaccurate rewards
            gradients.append(g)
        avg_gradients = []

        # each k represents a kernel's gradients
        # can't do this as a tensor because shapes change
        for k in range(len(gradients[0])):
            # get all of the gradients associated to one kernel
            t = tf.convert_to_tensor([grad[k] for grad in gradients])
            # average gradient across sessions
            t = tf.reduce_mean(t, axis=0)
            avg_gradients.append(t)

        # apply the gradients to their respective model parameters
        self.optimizer.apply_gradients(zip(avg_gradients, self.model.trainable_variables))
        print('SSE of weights: ', [tf.reduce_mean(f**2).numpy() for f in self.model.weights])
        return 1


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    model = Model()
    losses = 1
    while bool(losses):
        losses = model.batch_train()
    model.display = True
    while True:
        model.run()
