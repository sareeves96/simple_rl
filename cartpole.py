import tensorflow as tf
import gym
import numpy as np
from scipy.stats import zscore as z_transform
from IPython.display import clear_output, display

env = gym.make('CartPole-v1')


def run(model, render=False):
    
    all_obs = []
    all_ac = []
    rw = 0
    observation = env.reset()
    
    while True:
        if render:
            env.render()
        # get the distribution over action space from the model
        action_dist = model.predict(
            tf.expand_dims(tf.convert_to_tensor(observation, tf.float32), axis=0)
        )[0]
        
        # choose the highest value to simplify things
        # alternatively, could sample from action space:
        # action = int(np.random.choice(np.arange(2), p=action_dist))
        action = tf.argmax(action_dist).numpy()
        # get the information about the state of the system following the action
        observation, reward, done, info = env.step(action)
        all_obs.append(observation)
        all_ac.append(action)
        rw += reward
        # done specifies that the session is over, usually due to a win or loss
        if done:
            break
            
    # return the observations and rewards (the reward will be discounted later)
    return all_obs, rw


# discount rewards so that actions close to the end of the game have a larger weight
# a larger gamma makes early actions matter more
def discount(rw, gamma=0.9):
    weights = np.array([gamma**(rw.shape[0]-i-1) for i in range(rw.shape[0])])
    discounted = tf.convert_to_tensor(weights, dtype=tf.float32) * rw
    return discounted


# strengthen the outputs based on the size of the reward
# this reinforces all decisions, but the unit norm constraint compensates for the growing weights
# also, rewards are batch-normalized, such that the worst performance gets a negative reward and the best a positive one
def compute_loss(model, obs, rw):
    y_pred = model(obs)
    y_true = tf.round(y_pred)
    #return tf.reduce_sum((y_pred - y_true) ** 2) * rw
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) * rw


def batch_train(model, optimizer, batch_size, show=False):
    
    obs_list = []
    rw_list = []
    l = 0
    
    for _ in range(batch_size):
        obs, rw = run(model, show)
        obs_list.append(obs)
        rw_list.append(rw)
    
    print('rewards: ', rw_list)
    # normalize rewards
    rw_norm = z_transform(np.array(rw_list))
    # convert to tensors to discount
    rw_tensors = [tf.ones(shape=len(obs_list[i])) * rw for i, rw in enumerate(rw_norm)]
    rw_discount = [discount(rw) for rw in rw_tensors]
    obs_tensors = [tf.convert_to_tensor(obs, dtype=tf.float32) for obs in obs_list]
    
    gradients = []
    
    # for each set of observations and discounted rewards
    for obs, rw in zip(obs_tensors, rw_discount):
        # and for each frame in the simulation, which has one associated reward
        for i in range(tf.shape(obs)[1]):
            obs_tens = tf.expand_dims(tf.convert_to_tensor(obs.numpy()[i, :], tf.float32), axis=0)
            # calculate the gradient with respect to that one instance
            with tf.GradientTape() as tape:
                loss = compute_loss(model, obs_tens, rw[i])
                loss = tf.convert_to_tensor(loss, dtype=tf.float32)
            g = tape.gradient(loss, model.trainable_variables)
            # collect all the gradients, instead of applying them at each step which would give inaccurate rewards
            gradients.append(g)
        avg_gradients = []
    
    # each k represents a kernel's gradients
    for k in range(len(gradients[0])):
        # get all of the gradients associated to one kernel
        t = tf.convert_to_tensor([grad[k] for grad in gradients])
        t = tf.reduce_sum(t, axis=0)
        avg_gradients.append(t)
    
    assert not np.isnan(avg_gradients[0][0][0])
    # apply the gradients to their respective variables
    optimizer.apply_gradients(zip(avg_gradients, model.trainable_variables))
    
    return l

if __name__=='__main__':
	# specify the model architecture
	# a simple model suffices for such a low dimension, low complexity case
	model = tf.keras.Sequential([
    			tf.keras.layers.Input(shape=(env.observation_space.shape)),
    			tf.keras.layers.Dense(64), 
   			tf.keras.layers.Dense(64),
        		tf.keras.layers.Dense(env.action_space.n, 
                              		      activation=tf.nn.softmax)
	])


	# training is much faster if we don't show the simulation. But thats all the fun!
	optimizer = tf.keras.optimizers.Adam(1e-3)
	while True:
    		clear_output(wait=True)
    		batch_train(model, optimizer, 5, show=True)

