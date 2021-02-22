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
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import function, Checkpointer

# Observer: Show progress
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary()[0]:
            self.counter += 1 
        if self.counter % 10 == 0:
            print(f"\rBuffer replay init steps:{self.counter}/{self.total}", end="")

# Training Loop
def train_agent(n_iterations, tf_env, agent, dataset, collect_driver, global_step, train_metrics, train_checkpointer, train_checkpoint_interval, train_summary_writer, summary_interval):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)

    timed_at_step = global_step.numpy()
    time_acc = 0
    train_summary_writer.set_as_default()
    for iteration in range(n_iterations):
        # Start timer
        start_time = time.time()

        # Collect a few steps using collect_policy and save to the replay buffer.
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        
        # Sample a batch of data from the buffer and update the agent's network.
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)

        # Stop timer
        time_acc += time.time() - start_time
        
        # Metrics
        print(f"\rTraining iteration: {agent.train_step_counter.numpy()}, Loss:{train_loss.loss.numpy():.5f}", end="")
        if iteration % 100 == 0:
            print("\nTraining Metrics:")
            metric_utils.log_metrics(train_metrics)
            print("\n")

        with tf.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
            steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
            tf.summary.scalar(name='global_steps_per_sec', data=steps_per_sec, step=global_step)
            timed_at_step = global_step.numpy()
            time_acc = 0

        if global_step.numpy() % train_checkpoint_interval == 0:
            train_checkpointer.save(global_step=global_step.numpy())

    print("\n")

def main(_):
    env_name = "Breakout-v0"
    num_parallel_environments = 2
    max_steps_per_episode = 250
    opt_adam_learning_rate = 1e-3
    replay_buffer_capacity = 50000
    collect_steps_per_iteration = 1 * num_parallel_environments
    train_batch_size = 4
    summary_interval=33
    train_checkpoint_interval=100

    path = pathlib.Path(__file__)
    parent_dir = path.parent.resolve()
    saved_file_name = path.stem + time.strftime("_%Y%m%d_%H%M%S")
    train_checkpoint_dir = str(parent_dir / "train_checkpoint")
    train_summary_dir = str(parent_dir / "train_summary")
    # eval_summary_dir = str(parent_dir / "eval_summary")

    train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
    # train_summary_writer.set_as_default()

    # eval_summary_writer = tf.summary.create_file_writer(eval_summary_dir)
    # eval_metrics = [
    #     tf_metrics.AverageReturnMetric(),
    #     tf_metrics.AverageEpisodeLengthMetric()
    # ]

    global_step = tf.compat.v1.train.get_or_create_global_step()

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
    tf_env.seed([42]*tf_env.batch_size)
    tf_env.reset()

    # Create evaluation environment
    eval_py_env = suite_gym.load(
                    env_name, 
                    env_wrappers=[
                        lambda env: TimeLimit(env, duration=max_steps_per_episode)
                    ])
    eval_tf_env = TFPyEnvironment(eval_py_env)
    eval_tf_env.reset()

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

    agent = DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=10,
        td_errors_loss_fn=keras.losses.Huber(reduction="none"),
        gamma=0.99, # discount factor
        train_step_counter=global_step,
        epsilon_greedy=lambda: epsilon_fn(global_step))
    agent.initialize()

    # Creating the Replay Buffer and the Corresponding Observer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    # Create observer
    # Observer: Replay buffer observer
    replay_buffer_observer = replay_buffer.add_batch

    # Observer: Training Metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=tf_env.batch_size),
        ]

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
    print("\n")

    # Creating the Dataset
    # trajectories, buffer_info = replay_buffer.get_next(
    #     sample_batch_size=1, num_steps=2)

    # from tf_agents.trajectories.trajectory import to_transition
    # time_steps, action_steps, next_time_steps = to_transition(trajectories)

    # Creating the Dataset
    dataset = replay_buffer.as_dataset(
        sample_batch_size=train_batch_size,
        num_steps=2,
        num_parallel_calls=3).prefetch(3)

    # Optimize by wrapping some of the code in a graph using TF function.
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

    # print("Beginning initial metric ===================")
    # results = metric_utils.eager_compute(
    #     eval_metrics,
    #     eval_tf_env,
    #     agent.policy,
    #     num_episodes=10,
    #     train_step=global_step,
    #     summary_writer=eval_summary_writer,
    #     summary_prefix='Metrics',
    # )
    # metric_utils.log_metrics(eval_metrics)
    # print("Finished initial metric ===================")

    # Create checkpoint
    train_checkpointer = Checkpointer(
        ckpt_dir=train_checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        # replay_buffer=replay_buffer,
        global_step=global_step,
        # train_metrics = train_metrics,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics')
        )

    # Restore checkpoint
    train_checkpointer.initialize_or_restore()

    # Create evaluation loop
    def eval_agent(eval_env, policy, num_episodes):
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)

    print("\n\n++++++++++++++++++++++++++++++++++\n\n")

    # Train agent
    train_agent(300, tf_env, agent, dataset, collect_driver, global_step, train_metrics, train_checkpointer, train_checkpoint_interval, train_summary_writer, summary_interval)
    
    # # Policy Saver
    # tf_policy_saver = PolicySaver(agent.policy)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tf.random.set_seed(42)
    np.random.seed(42)
    
    tf_agents.system.multiprocessing.handle_main(main)


# Instructions to run the code
# ```bash
# $ cd /path/to/SMARTS/ultra_tf
# $ tensorboard --logdir . --port 2223
# $ python3.7 tfagent_dqn.py
# ```