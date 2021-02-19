import tensorflow as tf
import gym
import numpy as np
import argparse

from scipy.stats import zscore as z_transform

# comment out to use gpu
# tf.config.set_visible_devices([], 'GPU')


# discount rewards so that actions close to the end of the game have a larger weight
# a larger gamma makes early actions matter more
def discount(rw, gamma=0.9):
    weights = np.array([gamma**(rw.shape[0]-i-1) for i in range(rw.shape[0])])
    discounted = tf.convert_to_tensor(weights, dtype=tf.float32) * rw
    return discounted


class Model(object):

    def __init__(self,
                 activation='sigmoid',
                 layers=1,
                 size=4,
                 gamma=0.90,
                 render=False,
                 batch_size=10,
                 optimizer=tf.keras.optimizers.Adam(5e-2)
                 ):

        # specify the model architecture
        # a simple model suffices for such a low dimension, low complexity case
        self.model = tf.keras.Sequential(
              [tf.keras.layers.Input(shape=env.observation_space.shape)]
            + [tf.keras.layers.Dense(size, activation=activation) for _ in range(layers)]
            + [tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)]
        )
        self.gamma = gamma
        self.render = render
        self.batch_size = batch_size
        self.optimizer = optimizer

    def run(self):
        all_obs = []
        all_ac = []
        rw = 0
        observation = env.reset()

        while True:
            if self.render:
                env.render()
            # get the distribution over action space from the model
            action_dist = self.model.predict(tf.convert_to_tensor(tf.expand_dims(observation, 0)))[0]
            # sample from the action space
            action = int(np.random.choice(np.arange(len(action_dist)), p=action_dist))
            # get the information about the state of the system following the action
            observation, reward, done, info = env.step(action)
            all_obs.append(observation)
            all_ac.append(action)
            rw += reward
            # done specifies that the session is over, usually due to a win or loss
            if done:
                break

        # return the observations and rewards (the reward will be discounted later)
        return all_obs, all_ac, rw

    def compute_loss(self, obs, ac, rw):
        # reproduce the action probabilities that were predicted on this step so that we can calculate the gradient
        # this is the second time the action probabilities are calculated and it may be possible to recode this better
        y_pred = self.model(obs)

        # this is the chosen action, which depends on the probability in the prediction
        # the less likely this chosen action is, the higher the loss
        # since rewards are z-transformed, the sign will be negative for relatively poor runs in a session
        # increase the probability of actions that led to rewards, decrease those that led to negative rewards
        y_true = ac

        out = tf.math.multiply(tf.keras.losses.binary_crossentropy(y_true, y_pred), rw)

        return out

    def batch_train(self):
        obs_list = []
        ac_list = []
        rw_list = []

        for _ in range(self.batch_size):
            obs, ac, rw = self.run()
            obs_list.append(obs)
            ac_list.append(ac)
            rw_list.append(rw)

        if np.array_equal(np.array(rw_list), np.ones(len(rw_list)) * 500):
            print('Congratulations! The learner has achieved the best possible performance.')
            print('Saving weights to "success"')
            self.model.save_weights('success')
            return 0

        print('Mean reward: ', np.mean(rw_list))
        # normalize rewards
        rw_norm = z_transform(np.array(rw_list))
        # convert to tensors to discount
        rw_tensors = [tf.ones(shape=len(obs_list[i])) * rw for i, rw in enumerate(rw_norm)]
        # list of discounted rewards in session
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
                loss = tf.convert_to_tensor(loss, dtype=tf.float32)
            g = tape.gradient(loss, self.model.trainable_variables)
            # collect all the gradients, instead of applying them at each step which would give inaccurate rewards
            gradients.append(g)
        avg_gradients = []

        # each k represents a kernel's gradients
        # can't do this as a tensor because shapes change
        for k in range(len(gradients[0])):
            # get all of the gradients associated to one kernel
            t = tf.convert_to_tensor([grad[k] for grad in gradients])
            t = tf.reduce_mean(t, axis=0)
            avg_gradients.append(t)

        # apply the gradients to their respective variables
        # can't do this until all gradients have been calculated
        self.optimizer.apply_gradients(zip(avg_gradients, self.model.trainable_variables))
        # print(self.model.weights)

        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true',
                        help='whether to show the model, which slows down training')
    args = parser.parse_args()

    # training is much faster if we don't show the simulation. But thats all the fun!
    env = gym.make('Acrobot-v1')
    model = Model(render=args.show)
    losses = 1
    while bool(losses):
        losses = model.batch_train()
        model.model.save_weights('current')
    model.render = True
    while True:
        model.run()
