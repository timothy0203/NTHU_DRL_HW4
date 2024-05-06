import os
import sys
import time
import argparse
import pprint
from importlib import import_module

import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from env_wrappers import DummyVecEnv, SubprocVecEnv, Monitor, L2M2019EnvBaseWrapper, RandomPoseInitEnv, \
                            ActionAugEnv, RewardAugEnv, PoolVTgtEnv, SkipEnv, Obs2VecEnv, NoopResetEnv, L2M2019ClientWrapper
from logger import save_json, load_json

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

parser = argparse.ArgumentParser()
parser.add_argument('--play', action='store_true')
parser.add_argument('--submission', action='store_true')

parser.add_argument('--env', type=str, default='L2M2019', help='Environment id (e.g. HalfCheetah-v2 or L2M2019).')
parser.add_argument('--alg', type=str, default='sac', help='Algorithm name -- module where agent and learn function reside.')
parser.add_argument('--explore', default='DisagreementExploration', type=str, help='Exploration class name within explore module.')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--exp_name', default='exp', type=str)
# training params
# TODO: number of env can larger? 1 -> 12
parser.add_argument('--n_env', default=1, type=int, help='Number of environments in parallel.')
# TODO: need to update n_total_steps
parser.add_argument('--n_total_steps', default=0, type=int, help='Number of training steps on single or vectorized environment.')
parser.add_argument('--max_episode_length', default=1000, type=int, help='Reset episode after reaching max length.')
# logging
parser.add_argument('--load_path', default="./results/exp_L2M2019_2024-05-04_19-03_backup_v2/best_agent.ckpt-121000", type=str)
parser.add_argument('--output_dir', type=str)
# parser.add_argument('--output_dir', default='./results/models', type=str) # TODO: not sure need it or not
parser.add_argument('--save_interval', default=1000, type=int)
parser.add_argument('--log_interval', default=1000, type=int)

best_ep_length = float('-inf')

# --------------------
# Config
# --------------------

def parse_unknown_args(args):
    # construct new parser for the string of unknown args and reparse; each arg is now a list of items
    p2 = argparse.ArgumentParser()
    for arg in args:
        if arg.startswith('--'): p2.add_argument(arg, type=eval, nargs='*')
    # if arg contains only a single value, replace the list with that value
    out = p2.parse_args(args)
    for k, v in out.__dict__.items():
        if len(v) == 1:
            out.__dict__[k] = v[0]
    return out

def get_alg_config(alg, env, extra_args=None):
    alg_args = getattr(import_module('algs.' + alg), 'defaults')(env)
    if extra_args is not None:
        alg_args.update({k: v for k, v in extra_args.items() if k in alg_args})
    return alg_args

def get_env_config(env, extra_args=None):
    env_args = None
    if env == 'L2M2019':
        env_args = {'model': '3D', 'visualize': False, 'integrator_accuracy': 1e-3, 'difficulty': 2, 'stepsize': 0.01}
    if extra_args is not None and env_args is not None:
        env_args.update({k: v for k, v in extra_args.items() if k in env_args})
    return env_args

def print_and_save_config(args, env_args, alg_args, expl_args):
    print('Building environment and agent with the following config:')
    print(' Run config:\n' + pprint.pformat(args.__dict__))
    print(' Env config: ' + args.env + (env_args is not None)*('\n' + pprint.pformat(env_args)))
    print(' Alg config: ' + args.alg + '\n' + pprint.pformat(alg_args))
    print(' Exp config: ' + args.explore + (expl_args is not None)*('\n' + pprint.pformat(expl_args)))
    save_json(args.__dict__, os.path.join(args.output_dir, 'config_run.json'))
    save_json(alg_args, os.path.join(args.output_dir, 'config_alg.json'))
    if env_args: save_json(env_args, os.path.join(args.output_dir, 'config_env.json'))
    if expl_args: save_json(expl_args, os.path.join(args.output_dir, 'config_exp.json'))


# --------------------
# Environment
# --------------------

def make_single_env(env_name, mpi_rank, subrank, seed, env_args, output_dir):
    # env_kwargs serve to initialize L2M2019Env
    #   L2M2019Env default args are: visualize=True, integrator_accuracy=5e-5, difficulty=2, seed=0, report=None
    #   additionally here: env_kwargs include `model` which can be '2D' or '3D'
    #   NOTE -- L2M2019Env uses seed in reseting the velocity target map in VTgtField.reset(seed) in v_tgt_field.py

    if env_name == 'L2M2019':
        env = L2M2019EnvBaseWrapper(**env_args)
        obs = env.reset()
        # print(len(obs))
        # print(obs)
        # input('debug')
        env = RandomPoseInitEnv(env)
        env = ActionAugEnv(env)
        env = PoolVTgtEnv(env, **env_args)
        env = RewardAugEnv(env)
        env = SkipEnv(env)
        env = Obs2VecEnv(env)

        # args.max_episode_length = env.time_limit / env.n_skips
        # args.max_episode_length = env.time_limit / env.n_skips
    else:
        import gym
        env = gym.envs.make(env_name)
        env.seed(seed + subrank if seed is not None else None)

    # apply wrappers
    env = Monitor(env, os.path.join(output_dir, str(mpi_rank) + '.' + str(subrank)))

    return env

def build_env(args, env_args):
    def make_env(subrank):
        return lambda: make_single_env(args.env, args.rank, subrank, args.seed + 10000*args.rank, env_args, args.output_dir)

    if args.n_env > 1:
        return SubprocVecEnv([make_env(i) for i in range(args.n_env)])
    else:
        return DummyVecEnv([make_env(i) for i in range(args.n_env)])



class Agent:

    def __init__(self):
        self.model_path = "./results/exp_L2M2019_2024-05-04_19-03_backup_v2/best_agent.ckpt-121000" # 
        # self.model_path = "./112062530_hw2_data" # 
        print(f"self.model_path: {self.model_path}")
        args = argparse.Namespace(
            play=False,
            submission=False,
            env='L2M2019',
            alg='sac',
            explore='DisagreementExploration',
            seed=1,
            exp_name='exp',
            n_env=1,
            n_total_steps=0,
            max_episode_length=1000,
            rank=0,
            load_path=self.model_path,
            output_dir=os.path.dirname(self.model_path),  # This will be set later based on the provided arguments
            save_interval=1000,
            log_interval=1000
        )
        extra_args = argparse.Namespace()
        extra_args.__dict__.update(load_json(os.path.join(os.path.dirname(self.model_path), 'config_alg.json')))
        env_args = get_env_config(args.env, extra_args.__dict__)
        alg_args = get_alg_config(args.alg, args.env, extra_args.__dict__)
        self.keep_action = None
        self.step = 0
        self.LENGTH0 = 1 # leg length
        env_args = get_env_config(args.env, extra_args.__dict__)
        alg_args = get_alg_config(args.alg, args.env, extra_args.__dict__)
        env = build_env(args, env_args)

        expl_args = getattr(import_module('algs.explore'), 'defaults')(args.explore)
        expl_args.update({k: v for k, v in extra_args.__dict__.items() if k in expl_args})
        exploration = getattr(import_module('algs.explore'), args.explore)
        self.exploration = exploration(env.observation_space.shape, env.action_space.shape, **expl_args)
        
        tf_config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=tf_config)
        learn = getattr(import_module('algs.' + args.alg), 'learn')
        # exploration = None
        agent = learn(env, self.exploration, args.seed, args.n_total_steps, args.max_episode_length, alg_args, args)
        self.agent = agent

    
    def pool_vtgt(self, vtgt_field):
        # Transpose and flip over x coordinate
        vtgt = vtgt_field.swapaxes(1, 2)[:, ::-1, :]

        # Pool v_tgt_field to (3,3)
        pooled_vtgt = vtgt.reshape(2, 11, 11)[:, ::2, ::2].reshape(2, 3, 2, 3, 2).mean((2, 4))

        # Pool each coordinate
        x_vtgt = pooled_vtgt[0].mean(0)  # Pool dx over y coordinate
        y_vtgt = np.abs(pooled_vtgt[1].mean(1))  # Pool dy over x coordinate and return one-hot indicator of the argmin

        # Y turning direction (yaw tgt) = [left, straight, right]
        y_vtgt_onehot = np.zeros_like(y_vtgt)
        y_vtgt_argsort = y_vtgt.argsort()

        # If target is behind, choose second to argmin to force turn
        if y_vtgt[1] < 1 and y_vtgt_argsort[0] == 1:
            y_vtgt_onehot[y_vtgt_argsort[1]] = 1
        else:
            y_vtgt_onehot[y_vtgt_argsort[0]] = 1

        # Distance to vtgt sink
        goal_dist = np.sqrt(x_vtgt[1] ** 2 + y_vtgt[1] ** 2)

        # X speed tgt = [stop, go]
        x_vtgt_onehot = (goal_dist > 0.3)

        # Concatenate x_vtgt_onehot, y_vtgt_onehot, and goal_dist
        pooled_vtgt_field = np.hstack([x_vtgt_onehot, y_vtgt_onehot, goal_dist])

        return pooled_vtgt_field


    def obs2vec(self, obs_dict):
        # Augmented environment from the L2R challenge
        res = []

        # target velocity field (in body frame)
        res += obs_dict['v_tgt_field'].flatten().tolist()

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0] / self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][1] / self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][2] / self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])
        return res

    def preprocess(self, observation):
        # Apply preprocessing steps to observation

        # Pool v_tgt_field
        vtgt_field = observation['v_tgt_field']
        pooled_vtgt_field = self.pool_vtgt(vtgt_field)
        # Replace original v_tgt_field in the observation with pooled_vtgt_field
        observation['v_tgt_field'] = pooled_vtgt_field

        # Convert observation to vector representation
        obs_vec = self.obs2vec(observation)

        return obs_vec


    # checker version
    def act(self, observation):
        # Epsilon-greedy action
        if self.step % 4 == 0:
            observation = self.preprocess(observation)
            action = self.agent.get_actions(observation)
            action = self.exploration.select_best_action(np.atleast_2d(observation), action)
            # next_obs, rew, done, info = env.step(action.flatten())
            # print(f"action.shape: {action.shape}")
            # print(f"action: {action}")
            # print(f"action.flatten().shape: {action.flatten().shape}")
            # print(f"action.flatten(): {action.flatten()}")
            self.keep_action = action.flatten()
        self.step += 1
        return self.keep_action




# --------------------
# Run train and play
# --------------------

""" def main(args, extra_args):
    # env and algorithm config; update defaults with extra_args
    if args.load_path:
        print(f"args.load_path: {args.load_path}")
        extra_args.__dict__.update(load_json(os.path.join(os.path.dirname(args.load_path), 'config_alg.json')))
        if args.explore: extra_args.__dict__.update(load_json(os.path.join(os.path.dirname(args.load_path), 'config_exp.json')))
    env_args = get_env_config(args.env, extra_args.__dict__)
    alg_args = get_alg_config(args.alg, args.env, extra_args.__dict__)
    expl_args = None
    if args.explore:
        expl_args = getattr(import_module('algs.explore'), 'defaults')(args.explore)
        expl_args.update({k: v for k, v in extra_args.__dict__.items() if k in expl_args})

    # mpi config
    args.rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()
    args.world_size = 1 if MPI is None else MPI.COMM_WORLD.Get_size()

    # logging config
    if args.load_path:
        args.output_dir = os.path.dirname(args.load_path)
    if not args.output_dir:  # if not given use results/file_name/time_stamp
        logdir = args.exp_name + '_' + args.env + '_' + time.strftime("%Y-%m-%d_%H-%M")
        args.output_dir = os.path.join('results', logdir)
        if args.rank == 0: os.makedirs(args.output_dir)

    # build environment
    env = build_env(args, env_args)

    # build exploration module and defaults
    exploration = None
    if args.explore:
        exploration = getattr(import_module('algs.explore'), args.explore)
        exploration = exploration(env.observation_space.shape, env.action_space.shape, **expl_args)

    # init session
    tf_config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tf_config)

    # print and save all configs
    if args.rank == 0: print_and_save_config(args, env_args, alg_args, expl_args)

    # build and train agent
    learn = getattr(import_module('algs.' + args.alg), 'learn')
    print("ready to learn")
    agent = learn(env, exploration, args.seed, args.n_total_steps, args.max_episode_length, alg_args, args)
    print("after learn")

    if args.play:
        # if env_args: env_args['visualize'] = True # TODO: dubug try
        env = make_single_env(args.env, args.rank, args.n_env + 100, args.seed, env_args, args.output_dir)
        print("ready to reset in play")
        obs = env.reset()
        print("after reset to reset in play")
        episode_rewards = 0
        episode_steps = 0
        while True:
            # if episode_steps % 5 == 0: i = input('press key to continue ...')
            action = agent.get_actions(obs)  # (n_samples, batch_size, action_dim)
            action = exploration.select_best_action(np.atleast_2d(obs), action)
            next_obs, rew, done, info = env.step(action.flatten())
            r_bonus = exploration.get_exploration_bonus(np.atleast_2d(obs), action, np.atleast_2d(next_obs)).squeeze()
            episode_rewards += rew
            episode_steps += 1
#            print('q value: {:.4f}; reward: {:.2f}; aug_rewards: {:.2f}; bonus: {:.2f}; reward so far: {:.2f}'.format(
#                agent.get_action_value(np.atleast_2d(obs), action).squeeze(), rew, info.get('rewards', 0), r_bonus, episode_rewards))
            obs = next_obs
            env.render()
            if done:
                print('Episode length {}; cumulative reward: {:.2f}'.format(episode_steps, episode_rewards))
                episode_rewards = 0
                episode_steps = 0
                i = input('enter random seed: ')
                obs = env.reset(seed=int(i) if i is not '' else None)

    if args.submission:
        import opensim as osim
        from osim.redis.client import Client

        REMOTE_HOST = os.getenv("AICROWD_EVALUATOR_HOST", "127.0.0.1")
        REMOTE_PORT = os.getenv("AICROWD_EVALUATOR_PORT", 6379)
        client = Client(
            remote_host=REMOTE_HOST,
            remote_port=REMOTE_PORT
        )

        env = L2M2019ClientWrapper(client)
        env = ActionAugEnv(env)
        env = PoolVTgtEnv(env, **env_args)
        env = SkipEnv(env)
        env = Obs2VecEnv(env)

        obs = env.create()

        while True:
            action = agent.get_actions(obs)
            action = exploration.select_best_action(np.atleast_2d(obs), action)
            next_obs, rew, done, _ = env.step(action.flatten())
            obs = next_obs
            if done:
                obs = env.reset()
                if not obs:
                    break
            # env.render()

        env.submit()

    return agent


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    extra_args = parse_unknown_args(unknown)

    main(args, extra_args)
 """