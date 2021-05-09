import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import sys
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# from queue import Queue
from copy import deepcopy

class Queue():
    def __init__(self):
        self.data = []

    def get(self):
        return self.data.pop(0)

    def put(self, item):
        self.data.append(item)

    def empty(self):
        return len(self.data) == 0

def monte_carlo_estimates_fn(rewards, gamma=0.99, dtype=tf.float32):
    discounted_rewards = [0] * len(rewards)
    # discounted_rewards = tf.zeros_like(rewards, dtype=dtype)
    running_add = tf.constant([0.], dtype=dtype)
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards = tf.stack(discounted_rewards)
    return discounted_rewards

@tf.function
def loss_V_fn(monte_carlo_estimates: tf.Tensor, value: tf.Tensor):
    return 0.5 * tf.math.reduce_mean((monte_carlo_estimates - value) ** 2)

def build_V_model():
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation=None, kernel_initializer='he_uniform'))
    model.compile(loss=loss_V_fn, optimizer=Adam(learning_rate=0.001))
    return model

# def train_V(Vmodel: tf.keras.Model, states: tf.Tensor, mce: tf.Tensor, batch_size=16):
#     # d = tf.data.Dataset.from_tensor_slices((states, mce)).batch(batch_size)
#     # d = d.repeat()
#     # Vmodel.fit(d, batch_size=batch_size, verbose=0, epochs=100, steps_per_epoch=16)
#     Vmodel.fit(states, mce, batch_size=batch_size, verbose=0)

def loss_pi_fn(pi_poss: tf.Tensor, monte_carlo_estimates: tf.Tensor, 
            action: tf.Tensor,
            value_of_state: tf.Tensor, 
            beta = 0.05, action_space = 2):
    act = tf.one_hot(tf.squeeze(tf.cast(action, dtype=tf.int64)), depth=action_space)
    l = tf.clip_by_value(tf.math.exp((monte_carlo_estimates - value_of_state) / beta), -50., 500.)
    return -1 * tf.reduce_mean(tf.math.log(
                tf.math.reduce_sum(pi_poss * act, axis=1, keepdims=True)) * l)

def build_pi_model():
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))
    return model

def train_pi_per_step(pi_model: tf.keras.Model, V_model: tf.keras.Model, 
             states, actions, mce, beta=0.05, opt=None):
    value = V_model.predict(states)
    with tf.GradientTape() as tape:
        opt.minimize(lambda: loss_pi_fn(pi_model(states), mce, actions, 
                     value, beta=beta), pi_model.trainable_variables)
        # print(f"pi loss: {loss_pi_fn(pi_model(states), mce, actions, states, beta=beta).numpy()}")

def train_pi(pi_model: tf.keras.Model, V_model: tf.keras.Model, 
             states, actions, mce, beta=0.05, iteration = 10):
    opt = Adam(0.01)
    # opt = Adam(0.01, clipvalue=20.0)
    for _ in range(iteration):
        train_pi_per_step(pi_model, V_model, states, actions, mce, 
                          beta=beta, opt=opt)

def train(pi_model: tf.keras.Model, V_model: tf.keras.Model, D: Queue,
          gamma = 0.99, beta = 0.05):
    states, actions, mce = D.get()
    D.put((states, actions, mce))
    # mce = monte_carlo_estimates_fn(rewards, gamma=gamma)
    # mce, _ = tf.linalg.normalize(mce)
    V_model.fit(states, mce, batch_size=16, verbose=0)
    train_pi(pi_model, V_model, states, actions, mce, beta=beta)

def add_trajectory(D:Queue, state, action, reward, poss_act=None, dtype=tf.float32):
    if not D.empty():
        states, actions, rewards = D.get()
        states = tf.concat([states, tf.constant(state, dtype=dtype)], axis = 0)
        actions = tf.concat([actions, tf.constant([[action]], dtype=dtype)], axis = 0)
        rewards = tf.concat([rewards, tf.constant([[reward]], dtype=dtype)], axis = 0)
    else:
        states = tf.constant(state, dtype=dtype)
        actions = tf.constant([[action]], dtype=dtype)
        rewards = tf.constant([[reward]], dtype=dtype)
    
    D.put((states, actions, rewards))

def allocate_rewards(tmp: Queue, D: Queue, gamma = 0.99):
    states, actions, rewards = tmp.get()
    mce = monte_carlo_estimates_fn(rewards, gamma=gamma)
    # mce, _ = tf.linalg.normalize(mce)
    # D.put((states, actions, mce))
    if not D.empty():
        # states, actions, rewards = D.get()
        total_states, total_actions, total_mce = D.get()
        total_states = tf.concat([total_states, states], axis = 0)
        total_actions = tf.concat([total_actions, actions], axis = 0)
        total_mce = tf.concat([total_mce, mce], axis = 0)
    else:
        total_states = states
        total_actions = actions
        total_mce = mce

    D.put((total_states, total_actions, total_mce))


def record(replay_storage, D: Queue):
    if D.empty():
        replay_storage.append(D.get())

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    K = 5000
    beta = 0.05
    gamma = 0.9

    D = Queue()
    tmpQue = Queue()

    Pi = build_pi_model()
    V = build_V_model()

    replay_storage = []

    for k in range(K):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
        while not done:
            trajectories = Pi(state).numpy().flatten()
            action = np.random.choice(2, 1, p=trajectories)[0]
            next_state, reward, done, info = env.step(action)
            add_trajectory(tmpQue, state, action, reward)
            state = np.reshape(next_state, [1, 4])
            score += reward
            if done:
                # record(replay_storage, D)
                allocate_rewards(tmpQue, D, gamma = gamma)
                train(Pi, V, D, gamma=gamma, beta=beta)
        if k % 100 == 0:
            ts = 0
            for e in range(20):
                done = False
                score = 0
                state = env.reset()
                state = np.reshape(state, [1, 4])
                while not done:
                    trajectories = Pi(state).numpy().flatten()
                    action = np.argmax(trajectories)
                    next_state, reward, done, info = env.step(action)
                    env.render()
                    state = np.reshape(next_state, [1, 4])
                    score += reward
                ts += score
            print(f"[testing] the score is {ts / 20}")

