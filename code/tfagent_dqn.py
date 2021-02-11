#Implementing TF-Agent DQN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_agents.environments import suite_gym
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.wrappers import TimeLimit

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_seed(42)
np.random.seed(42)

env_name = "Breakout-v0"
env = suite_gym.load(env-name)
env.seed(42)
env = TimeLimit(env, duration=150)

env.reset()
tf_env = TFPyEnvironment(env)

# Creating the Deep Q-Network
preprocessing_layer = keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32) / 255.)

conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

# Creating the DQN Agent
train_step = tf.Variable(0)
update_period = 1 # train the model every 4 steps

optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
    epsilon=0.00001, centered=True)

# learning_rate = 1e-3
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)


epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε

agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000, # <=> 32,000 ALE frames
    td_errors_loss_fn=keras.losses.Huber(reduction="none"),
    gamma=0.99, # discount factor
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

# Creating the Replay Buffer and the Corresponding Observer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=300000)

replay_buffer_observer = replay_buffer.add_batch

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

# Creating Training Metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
    ]

logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)


# Creating the Collect Driver
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration



from tf_agents.policies.random_tf_policy import RandomTFPolicy
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
    tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(200)],
    num_steps=200) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()


# Creating the Dataset
trajectories, buffer_info = replay_buffer.get_next(
    sample_batch_size=2, num_steps=3)

from tf_agents.trajectories.trajectory import to_transition
time_steps, action_steps, next_time_steps = to_transition(trajectories)


dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

# Creating the Training Loop
from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 100 == 0:
            print("Inside\n")
            log_metrics(train_metrics)

print("\n\n ++++++++++++++++++++++++++++++++++\n\n")

train_agent(1000)

