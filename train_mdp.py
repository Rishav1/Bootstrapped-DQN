import plotly.offline as offline
import plotly.graph_objs as go

import argparse
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tempfile
import time
from mdp_environment import ActionGenerator, StateGenerator, MDPModel

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
from model import model, dueling_model, bootstrap_model, simple_bootstrap_model

def plotly_plot(ep_rewards, filename):
    layout = go.Layout(
        title='Mean episodic score for multidimentional MDP',
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=dict(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='Episodes',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            title='(10 episode) Mean score',
            zeroline=False
        ),
    )
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    mean_data = []
    max_data = []
    min_data = []
    
    
    data = []
    ep_rewards_mean = []

    for head, rewards in ep_rewards.items():
        rewards_mean = running_mean(rewards, 100)
        data.append(
            go.Scatter(
                y=rewards_mean,
                x=list(range(len(rewards_mean))),
                mode='lines',
                name='head {}'.format(head + 1),
            )
        )
        ep_rewards_mean.append(rewards_mean)
    
    for tup in zip(*ep_rewards_mean):
        mean_data.append(sum(tup)/len(tup))
        max_data.append(max(tup))
        min_data.append(min(tup))
    
    max_data = np.flip(max_data, 0)
    episodes = list(range(len(mean_data)))
    episodes_reversed = list(range(len(mean_data) - 1, -1, -1))
    
    data.append(
        go.Scatter(
            x=episodes,
            y=mean_data,
            line=dict(color='rgb(0,176,246)', width=4, dash='dash'),    
            mode='lines',
            name='mean'
        )
    )

    data.append(
        go.Scatter(
            x=episodes+episodes_reversed,
            y=np.concatenate([min_data, max_data], axis=0),
            fill='tozerox',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='transparent'),
            showlegend=False
        )
    )

     
    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename=filename, auto_open=False, image='png')
    #offline.image.save_as(fig, filename=filename)

def multidim_mdp(num_nodes_per_dim, num_dimension, state_size):
    states = {}
    actions = []
    action_generator = ActionGenerator('name')
    for dim in range(num_dimension):
        actions.append(action_generator.generate_action(name=2*dim))
        actions.append(action_generator.generate_action(name=2*dim+1))

    state_generator = StateGenerator('name', 'value', 'reward')
    for state_ind_value in range(num_nodes_per_dim ** num_dimension):
        state_ind = []
        state_ind_value_copy = state_ind_value
        for i in range(num_dimension):
            state_ind.append(state_ind_value_copy % num_nodes_per_dim)
            state_ind_value_copy = int(state_ind_value_copy / num_nodes_per_dim)

        if state_ind_value == 0:
            reward = 1/ num_nodes_per_dim
        elif state_ind_value == num_nodes_per_dim ** num_dimension - 1:
            reward = 1
        else:
            reward = 0
        states[tuple(state_ind)] = state_generator.generate_state(name=tuple(state_ind),
                                                                  value=np.random.random_sample(state_size),
                                                                  reward=reward)

    multidim_model = MDPModel('Multidim')
    multidim_model.add_states(states.values())
    multidim_model.add_actions(actions)


    for state_ind, state in states.items():
        for dim in range(num_dimension):
            if state_ind[dim] != 0:
                next_state_ind = list(state_ind)
                next_state_ind[dim] -= 1
                multidim_model.add_transition(state, actions[2 * dim], {states[tuple(next_state_ind)] : 1})
            else:
                multidim_model.add_transition(state, actions[2 * dim], {state : 1})

            if state_ind[dim] != num_nodes_per_dim - 1:
                next_state_ind = list(state_ind)
                next_state_ind[dim] += 1
                multidim_model.add_transition(state, actions[2 * dim + 1], {states[tuple(next_state_ind)] : 1})
            else:
                multidim_model.add_transition(state, actions[2 * dim + 1], {state : 1})

    multidim_model.add_init_states({states[(1,) * num_dimension] : 1})
    multidim_model.add_final_states([states[(0,) * num_dimension], states[(num_nodes_per_dim - 1,) * num_dimension]], num_nodes_per_dim * num_dimension + 9)
    multidim_model.finalize()
    return multidim_model


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Cartpole")
    # Environment
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    parser.add_argument("--gpu", type=int, default=1, help="GPU device to use(0 for none)")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(4e5), help="total number of steps to run the environment for")
    parser.add_argument("--epsilon-schedule", type=int, default=5, help="epsilon shedule parameter")
    parser.add_argument("--learning-schedule", type=float, default=1.6, help="learning shedule parameter")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=1, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="number of iterations between every target network update")
    parser.add_argument("--heads", type=int, default=5, help="number of heads for bootstrap")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "bootstrap", default=True, help="whether or not to use bootstrap model")
    boolean_flag(parser, "swarm", default=False, help="whether or not to use bootstrap model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # mdp Parameters
    parser.add_argument("--mdp-arity", type=int, default=10, help="nodes per dim of MDP")
    parser.add_argument("--mdp-dimension", type=int, default=1, help="dimentions of MDP")
    parser.add_argument("--mdp-state-size", type=int, default=5, help="representational dimension of MDP states")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./logs", help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=5000, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


def maybe_save_model(savedir, container, state, rewards, steps):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    # relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    # if container is not None:
    #     container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    relatively_safe_pickle_dump(rewards, os.path.join(savedir, 'rewards.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'rewards.pkl'), 'rewards.pkl')
    relatively_safe_pickle_dump(steps, os.path.join(savedir, 'steps.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'steps.pkl'), 'steps.pkl')
    plotly_plot(rewards, os.path.join(savedir, 'returns.html'))
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    args = parse_args()
    if args.gpu == 0:
        args.device = "/cpu:0"
    else:
        args.device = "/gpu:{}".format(args.gpu - 1)
    # Parse savedir and azure container.
    savedir = "{}{}_mdp_{}_{}_{}".format(args.save_dir,"swarm" if args.swarm else "bootstrap", args.mdp_arity, args.mdp_dimension, args.mdp_state_size)
    if args.save_azure_container is not None:
        account_name, account_key, container_name = args.save_azure_container.split(":")
        container = Container(account_name=account_name,
                              account_key=account_key,
                              container_name=container_name,
                              maybe_create=True)
        if savedir is None:
            # Careful! This will not get cleaned up. Docker spoils the developers.
            savedir = tempfile.TemporaryDirectory().name
    else:
        container = None

    env = multidim_mdp(args.mdp_arity, args.mdp_dimension, args.mdp_state_size)

    with U.make_session(120) as sess:
        # Create training graph and replay buffer
        if args.bootstrap :
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: U.Uint8Input((args.mdp_state_size,), name=name),
                q_func=simple_bootstrap_model,
                bootstrap=args.bootstrap,
                num_actions=2 * args.mdp_dimension,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=0.99,
                grad_norm_clipping=10,
                double_q=args.double_q,
                heads=args.heads,
                swarm=args.swarm,
                device=args.device
            )
        else:
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: U.Uint8Input((args.mdp_state_size,), name=name),
                q_func=dueling_model if args.dueling else model,
                num_actions=2 * args.mdp_arity,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=0.99,
                grad_norm_clipping=10,
                double_q=args.double_q,
                device=args.device
            )

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.num_steps / args.epsilon_schedule, 0.1), # (approximate_num_iters / 50, 0.1),
            (args.num_steps / (args.epsilon_schedule * 0.1), 0.01) # (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)
        learning_rate = PiecewiseSchedule([
            (0, 1e-4),
            (args.num_steps / args.learning_schedule, 1e-4),
            (args.num_steps / (args.learning_schedule * 0.5), 5e-5)
        ], outside_value=5e-5)

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0
        ep_rewards = {}

        # Load the model
        state = maybe_load_model(savedir, container)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            # monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.initialize()

        # Main training loop
        head = np.random.randint(args.heads)        #Initial head initialisation
        ep_rewards[head] = [0]
        ep_rewards_cum = []
        episode = 1

        while True:
            num_iters += 1
            # Take action and store transition in the replay buffer.
            if args.bootstrap:
                action = act(obs.value[None], head=head, update_eps=exploration.value(num_iters))[0]
            else:
                action = act(obs.value[None], update_eps=exploration.value(num_iters))[0]
            new_obs = env.transition(next(env.get_actions(action)))
            ep_rewards[head][-1] += new_obs.reward

            terminated = env.is_terminated()
            replay_buffer.add(obs.value, action, new_obs.reward, new_obs.value, terminated)
            obs = new_obs
            if terminated:
                ep_rewards_cum.append(ep_rewards[head][-1])
                episode += 1
                obs = env.initialize()
                head = np.random.randint(args.heads)
                if head in ep_rewards.keys():
                    ep_rewards[head].append(0)
                else:
                    ep_rewards[head] = [0]

            if (num_iters > max(0.5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)
                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, learning_rate.value(num_iters))
                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(num_iters - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), num_iters

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or num_iters > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                }, ep_rewards, num_iters)

            if num_iters > args.num_steps:
                break

            if terminated:
                steps_left = args.num_steps - num_iters
                completion = np.round(num_iters / args.num_steps, 1)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", episode)
                logger.record_tabular("rewards_tot (100 epi mean)", np.mean(ep_rewards_cum[-100:]))
                logger.record_tabular("reward (100 epi mean)", ["{0:.2f}".format(np.mean(rewards[-100:])) for rewards in ep_rewards.values()])
                logger.record_tabular("head for episode", (head+1))
                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter._value) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else None)
                logger.dump_tabular()
                logger.log()
                if fps_estimate:
                    logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
