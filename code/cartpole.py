import gym
import pyvirtualdisplay
import numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(1)
tf.compat.v1.set_random_seed(3)

# env = gym.make("CartPole-v1")
# obs = env.reset()
# print(obs)

# action = 1
# obs, reward, done, info = env.step(action)

# n_inputs = 4 # == env.observation_space.shape[0]
# model = keras.models.Sequential([
#     keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
#     keras.layers.Dense(1, activation="sigmoid"),
# ])

# def play_one_step(env, obs, model, loss_fn):
#     with tf.GradientTape() as tape:
#         left_proba = model(obs[np.newaxis])
#         action = (tf.random.uniform([1, 1]) > left_proba)
#         y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
#         loss = tf.reduce_mean(loss_fn(y_target, left_proba))
#     grads = tape.gradient(loss, model.trainable_variables)
#     obs, reward, done, info = env.step(int(action[0, 0].numpy()))
#     return obs, reward, done, grads

# def play_single_episodes(env, n_max_steps, model, loss_fn):
#     current_rewards = 0
#     obs = env.reset()
#     for step in range(n_max_steps):
#         obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
#         current_rewards += reward
#         if done:
#             break
#     return current_rewards

# def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
#     all_rewards = []
#     all_grads = []
#     for episode in range(n_episodes):
#         current_rewards = []
#         current_grads = []
#         obs = env.reset()
#         for step in range(n_max_steps):
#             obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
#             current_rewards.append(reward)
#             current_grads.append(grads)
#             if done:
#                 break
#         all_rewards.append(current_rewards)
#         all_grads.append(current_grads)
#     return all_rewards, all_grads

# def discount_rewards(rewards, discount_factor):
#     discounted = np.array(rewards)
#     for step in range(len(rewards) - 2, -1, -1):
#         discounted[step] += discounted[step + 1] * discount_factor
#     return discounted

# def discount_and_normalize_rewards(all_rewards, discount_factor):
#     all_discounted_rewards = [discount_rewards(rewards, discount_factor)
#         for rewards in all_rewards]
#     flat_rewards = np.concatenate(all_discounted_rewards)
#     reward_mean = flat_rewards.mean()
#     reward_std = flat_rewards.std()
#     return [(discounted_rewards - reward_mean) / reward_std
#         for discounted_rewards in all_discounted_rewards]

# n_iterations = 150
# n_episodes_per_update = 10
# n_max_steps = 200
# discount_factor = 0.95

# optimizer = keras.optimizers.Adam(lr=0.01)
# loss_fn = keras.losses.binary_crossentropy

# for iteration in range(n_iterations):
#     all_rewards, all_grads = play_multiple_episodes(
#         env, n_episodes_per_update, n_max_steps, model, loss_fn)
#     all_final_rewards = discount_and_normalize_rewards(all_rewards,
#         discount_factor)
#     all_mean_grads = []
#     for var_index in range(len(model.trainable_variables)):
#         mean_grads = tf.reduce_mean(
#             [final_reward * all_grads[episode_index][step][var_index]
#             for episode_index, final_rewards in enumerate(all_final_rewards)
#                 for step, final_reward in enumerate(final_rewards)], axis=0)
#         all_mean_grads.append(mean_grads)
#     optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
#     print(f"Iteration number: {iteration}")

# totals = []
# for episode in range(300):
#     episode_rewards = 0
#     obs = env.reset()
#     for step in range(200):
#         # action = basic_policy(obs)
#         left_proba = model(obs[np.newaxis])
#         action = (tf.random.uniform([1, 1]) > left_proba)
#         # obs, reward, done, info = env.step(action)
#         obs, reward, done, info = env.step(int(action[0, 0].numpy()))
#         episode_rewards += reward
#         if done:
#             break
#     totals.append(episode_rewards)

# print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))


# transition_probabilities = [ # shape=[s, a, s']
#     [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
#     [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
#     [None, [0.8, 0.1, 0.1], None]]

# rewards = [ # shape=[s, a, s']
#     [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
#     [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
#     [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]

# possible_actions = [[0, 1, 2], [0, 2], [1]]

# Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
# for state, actions in enumerate(possible_actions):
#     print(actions)
#     Q_values[state, actions] = 0.0 # for all possible actions

# gamma = 0.90 # the discount factor
# for iteration in range(50):
#     Q_prev = Q_values.copy()
#     for s in range(3):
#         for a in possible_actions[s]:
#             Q_values[s, a] = np.sum([
#                     transition_probabilities[s][a][sp]
#                     * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
#                 for sp in range(3)])

# print("======= ",Q_values)


# def step(state, action):
#     probas = transition_probabilities[state][action]
#     next_state = np.random.choice([0, 1, 2], p=probas)
#     reward = rewards[state][action][next_state]
#     return next_state, reward

# def exploration_policy(state):
#     return np.random.choice(possible_actions[state])

# alpha0 = 0.05 # initial learning rate
# decay = 0.005 # learning rate decay
# gamma = 0.90 # discount factor
# state = 0 # initial state

# for iteration in range(10000):
#     action = exploration_policy(state)
#     next_state, reward = step(state, action)
#     next_value = np.max(Q_values[next_state])
#     alpha = alpha0 / (1 + iteration * decay)
#     Q_values[state, action] *= 1 - alpha
#     Q_values[state, action] += alpha * (reward + gamma * next_value)
#     state = next_state

print("==================================================================")
#Implementing Deep Q-Learning

env = gym.make("CartPole-v0")
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n
model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

from collections import deque

replay_buffer = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]    
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

batch_size = 2
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                    (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
        training_step(batch_size)

print("============================================================")
