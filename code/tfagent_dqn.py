#Implementing TF-Agent DQN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import logging
import numpy as np
import pathlib
import tensorflow as tf
import tf_agents
import time

from tensorflow import keras
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import suite_gym
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import function, Checkpointer

def main(_):
    env_name = "Breakout-v0"
    num_parallel_environments = 2
    max_steps_per_episode = 150
    opt_adam_learning_rate = 1e-3
    replay_buffer_capacity = 50000
    collect_steps_per_iteration = 1 * num_parallel_environments
    train_batch_size = 4

    path = pathlib.Path(__file__)
    parent_dir = path.parent.resolve()
    checkpoint_dir = parent_dir / "checkpoint"
    checkpoint_name = path.stem + time.strftime("_%Y%m%d_%H%M%S")

    # Create parallel environment for training
    tf_env = TFPyEnvironment(
                ParallelPyEnvironment([
                        lambda: suite_gym.load(
                            env_name,
                            env_wrappers=[
                                lambda env: TimeLimit(env, duration=max_steps_per_episode)
                            ]
                        )
                    ]*num_parallel_environments
                )
            )

    # Create evaluation environment
    eval_py_env = suite_gym.load(env_name)
    eval_env = TFPyEnvironment(eval_py_env)


    tf_env.seed([42]*tf_env.batch_size)
    tf_env.reset()

    # Creating the Deep Q-Network
    preprocessing_layer = keras.layers.Lambda(
            lambda obs: tf.cast(obs, np.float32) / 255.
        )

    conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params=[512]

    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params)

    # Creating the DQN Agent
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=opt_adam_learning_rate)

    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0, # initial ε
        decay_steps=250000,
        end_learning_rate=0.01) # final ε

    train_step = tf.Variable(0)
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
        max_length=replay_buffer_capacity)

    # Create observer
    # Observer: Replay buffer observer
    replay_buffer_observer = replay_buffer.add_batch

    # Observer: Show progress
    class ShowProgress:
        def __init__(self, total):
            self.counter = 0
            self.total = total
        def __call__(self, trajectory):
            self.counter += tf.reduce_sum(tf.cast(trajectory.is_boundary(), tf.int64))
            if self.counter % 100 == 0:
                print("\rTotal steps:{}/{}".format(self.counter, self.total), end="")

    # Observer: Training Metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=tf_env.batch_size),
        ]

    logging.getLogger().setLevel(logging.INFO)

    # Creating the Collect Driver
    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=collect_steps_per_iteration)

    # Initialize dataset
    initial_collect_policy = RandomTFPolicy(
                                tf_env.time_step_spec(),
                                tf_env.action_spec())
    init_driver = DynamicStepDriver(
                    tf_env,
                    initial_collect_policy,
                    observers=[replay_buffer_observer, ShowProgress(200)],
                    num_steps=200)
    final_time_step, final_policy_state = init_driver.run()

    # Creating the Dataset
    # trajectories, buffer_info = replay_buffer.get_next(
    #     sample_batch_size=1, num_steps=2)

    # from tf_agents.trajectories.trajectory import to_transition
    # time_steps, action_steps, next_time_steps = to_transition(trajectories)

    dataset = replay_buffer.as_dataset(
        sample_batch_size=train_batch_size,
        num_steps=2,
        num_parallel_calls=3).prefetch(3)

    # Optimize by wrapping some of the code in a graph using TF function.
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

    # Create evaluation loop
    def eval_agent(eval_env, policy, num_episodes):
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)

    # Creating the Training Loop
    def train_agent(n_iterations):
        time_step = None
        policy_state = agent.collect_policy.get_initial_state(num_parallel_environments)
        iterator = iter(dataset)
  
        for iteration in range(n_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            
            # Sample a batch of data from the buffer and update the agent's network.
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)

            # Metrics
            print(f"\rTraining iteration: {agent.train_step_counter.numpy()}, Loss:{train_loss.loss.numpy():.5f}", end="")
            if iteration % 100 == 0:
                print("\nTraining Metrics:")
                log_metrics(train_metrics)
                print("\n")

        print("\n")

    # Checkpoint
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    

    print("\n\n++++++++++++++++++++++++++++++++++\n\n")

    train_agent(300)



if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tf.random.set_seed(42)
    np.random.seed(42)
    
    tf_agents.system.multiprocessing.handle_main(main)
