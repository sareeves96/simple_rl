import tensorflow as tf
import gym
import numpy as np
import argparse

from scipy.stats import zscore as z_transform


# discount rewards so that actions close to the end of the game have a larger weight
# a larger gamma makes early actions matter more
def discount(rw, gamma=0.9):
    weights = np.array([gamma**(rw.shape[0]-i-1) for i in range(rw.shape[0])])
    discounted = tf.convert_to_tensor(weights, dtype=tf.float32) * rw
    return discounted


class Model(object):

    def __init__(self,
                 activation='elu',
                 layers=1,
                 size=4,
                 gamma=0.95,
                 render=False,
                 batch_size=20,
                 optimizer=tf.keras.optimizers.Adam(1e-2)):
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
            action_dist = self.model.predict(tf.expand_dims(tf.convert_to_tensor(observation, tf.float32), axis=0))[0]
            # sample from the action space
            action = int(np.random.choice(np.arange(2), p=action_dist))
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

    # strengthen the outputs based on the size of the reward
    # this reinforces all decisions, but the unit norm constraint compensates for the growing weights
    # also, rewards are batch-normalized, such that the worst performance gets a negative reward and the best a
    # positive one
    def compute_loss(self, obs, ac, rw):
        y_pred = self.model(obs)
        # whether the action was likely or unlikely, we want to improve the chance of it occuring again
        y_true = ac
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) * rw

    def batch_train(self):
        obs_list = []
        ac_list = []
        rw_list = []
        l = 0

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
        rw_discount = [discount(rw) for rw in rw_tensors]
        obs_tensors = [tf.convert_to_tensor(obs, dtype=tf.float32) for obs in obs_list]
        ac_tensors = [tf.one_hot(ac, depth=env.action_space.n) for ac in ac_list]

        gradients = []

        # for each set of observations and discounted rewards
        for obs, ac, rw in zip(obs_tensors, ac_tensors, rw_discount):
            # and for each frame in the simulation, which has one associated reward
            for i in range(tf.shape(obs)[1]):
                obs_tens = tf.expand_dims(tf.convert_to_tensor(obs.numpy()[i, :], tf.float32), axis=0)
                # calculate the gradient with respect to that one instance
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(obs_tens, ac, rw[i])
                    loss = tf.convert_to_tensor(loss, dtype=tf.float32)
                    l += tf.reduce_sum(loss)
                g = tape.gradient(loss, self.model.trainable_variables)
                # collect all the gradients, instead of applying them at each step which would give inaccurate rewards
                gradients.append(g)
            avg_gradients = []

        # each k represents a kernel's gradients
        for k in range(len(gradients[0])):
            # get all of the gradients associated to one kernel
            t = tf.convert_to_tensor([grad[k] for grad in gradients])
            t = tf.reduce_mean(t, axis=0)
            avg_gradients.append(t)

        assert not np.isnan(avg_gradients[0][0][0])
        # apply the gradients to their respective variables
        self.optimizer.apply_gradients(zip(avg_gradients, self.model.trainable_variables))

        return l


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true',
                        help='whether to show the model, which slows down training')
    args = parser.parse_args()

    # training is much faster if we don't show the simulation. But thats all the fun!
    env = gym.make('CartPole-v1')
    model = Model(render=args.show)
    losses = 1
    while bool(losses):
        losses = model.batch_train()
        model.model.save_weights('current')

