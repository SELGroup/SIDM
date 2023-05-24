import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy
from tqdm import tqdm

import utils
from logger import Logger
from video import VideoRecorder

from curl_sac import RadSacAgent
from torchvision import transforms
import data_augs as rad

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='ball_in_cup')
    parser.add_argument('--task_name', default='catch')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=108, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int) 
    # train
    parser.add_argument('--agent', default='rad_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=100000, type=int)  # 500000
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic (Q_ϕ)
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor (π_ψ)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder (f_θ)
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--load_model', default='', type=str,
        help='File path to trained critic model from which to load encoder')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--test_mode', default=False, action='store_true')
    parser.add_argument('--replicate', default=False, action='store_true')
    # data augs
    parser.add_argument('--data_augs', default='translate', type=str)
    # Markov abstraction
    parser.add_argument('--disable_rad', default=False, action='store_true')
    parser.add_argument('--markov', default=False, action='store_true')
    parser.add_argument('--markov_pretrain_steps', default=1000, type=int,
        help='Number of steps to pretrain the Markov abstraction on the first `init_steps` experiences')
    parser.add_argument('--markov_pretrain_batch_size', default=512, type=int)
    parser.add_argument('--markov_catchup_steps', default=0, type=int,
        help='Number of agent updates to catch up on after pretraining is complete')
    parser.add_argument('--markov_inv_coef', default=1, type=float)
    parser.add_argument('--markov_contr_coef', default=1, type=float)
    parser.add_argument('--markov_smoothness_coef', default=10, type=float)
    parser.add_argument('--markov_smoothness_max_dz', default=0.01, type=float)
    parser.add_argument('--markov_lr', default=1e-3, type=float)
    parser.add_argument('--markov_beta', default=0.9, type=float)
    parser.add_argument('--tag', type=str)
    # SISA abstraction
    # parser.add_argument('--disable_rad', default=False, action='store_true')
    parser.add_argument('--sisa', default=True, action='store_true')
    parser.add_argument('--sisa_pretrain_steps', default=1000, type=int,
        help='Number of steps to pretrain the Markov abstraction on the first `init_steps` experiences')
    parser.add_argument('--sisa_pretrain_batch_size', default=512, type=int)
    parser.add_argument('--sisa_catchup_steps', default=0, type=int,
        help='Number of agent updates to catch up on after pretraining is complete')
    parser.add_argument('--sisa_inv_coef', default=1, type=float)
    parser.add_argument('--sisa_contr_coef', default=1, type=float)
    parser.add_argument('--sisa_smoothness_coef', default=10, type=float)
    parser.add_argument('--sisa_smoothness_max_dz', default=0.01, type=float)
    parser.add_argument('--sisa_lr', default=1e-3, type=float)
    parser.add_argument('--sisa_beta', default=0.9, type=float)
    parser.add_argument('--sisa_update_freq', default=100, type=int)
    parser.add_argument('--sisa_fine_coef', default=0.1, type=float)
    parser.add_argument('--sisa_abs_coef', default=0.1, type=float)
    # parser.add_argument('--tag', type=str)

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()

    markov_params = {
        'enable': args.markov,
        'pretrain_steps': args.markov_pretrain_steps,
        'pretrain_batch_size': args.markov_pretrain_batch_size,
        'catchup_steps': args.markov_catchup_steps,
        'lr': args.markov_lr,
        'inverse_coef': args.markov_inv_coef,
        'contrastive_coef': args.markov_contr_coef,
        'smoothness_coef': args.markov_smoothness_coef,
        'smoothness_max_dz': args.markov_smoothness_max_dz,
        'optim_beta': args.markov_beta,
        'latent_dim': args.encoder_feature_dim,
        'layer_size': args.hidden_dim,
        'log_std_min': args.actor_log_std_min,
        'log_std_max': args.actor_log_std_max,
    }
    args.markov_params = markov_params

    sisa_params = {
        'enable': args.sisa,
        'pretrain_steps': args.sisa_pretrain_steps,
        'pretrain_batch_size': args.sisa_pretrain_batch_size,
        'catchup_steps': args.sisa_catchup_steps,
        'lr': args.sisa_lr,
        'inverse_coef': args.sisa_inv_coef,
        'contrastive_coef': args.sisa_contr_coef,
        'smoothness_coef': args.sisa_smoothness_coef,
        'smoothness_max_dz': args.sisa_smoothness_max_dz,
        'optim_beta': args.sisa_beta,
        'latent_dim': args.encoder_feature_dim,
        'layer_size': args.hidden_dim,
        'log_std_min': args.actor_log_std_min,
        'log_std_max': args.actor_log_std_max,
        'update_freq': args.sisa_update_freq,
        'fine_coef': args.sisa_fine_coef,
        'abs_coef': args.sisa_abs_coef
    }
    args.sisa_params = sisa_params

    if args.replicate:
        # Action repeat
        if args.domain_name in ['finger', 'walker']:
            args.action_repeat = 2
        elif args.domain_name == 'cartpole':
            args.action_repeat = 8
        else:
            args.action_repeat = 4

        # Learning rate
        if args.domain_name == 'cheetah':
            args.critic_lr = 2e-4
            args.actor_lr = 2e-4
            args.encoder_lr = 2e-4

        # Data augmentation
        if args.domain_name == 'walker':
            args.pre_transform_image_size = 100
            args.image_size = 84
            args.data_augs = 'crop'

    if args.disable_rad:
        assert args.data_augs in ['crop', 'translate']
        args.data_augs = 'center_' + args.data_augs

    if args.test_mode:
        print("Test mode enabled; modifying args for speed.")
        args.init_steps = 4
        args.num_train_steps = 8
        args.eval_freq = 4
        args.num_eval_episodes = 2

    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel' and 'crop' in args.data_augs:
                    obs = utils.center_crop_image(obs,args.image_size)
                if args.encoder_type == 'pixel' and 'translate' in args.data_augs:
                    # first crop the center with pre_image_size
                    obs = utils.center_crop_image(obs, args.pre_transform_image_size)
                    # then translate cropped to center
                    obs = utils.center_translate(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs / 255.)
                    else:
                        action = agent.select_action(obs / 255.)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        filename = args.work_dir + '/' + args.domain_name + '--'+ args.task_name + '-' + args.data_augs + '--s' + str(args.seed) + '--eval_scores.npy'
        key = args.domain_name + '-' + args.task_name + '-' + args.data_augs
        try:
            log_data = np.load(filename,allow_pickle=True)
            log_data = log_data.item()
        except:
            log_data = {}

        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]['step'] = step
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward
        log_data[key][step]['max_ep_reward'] = best_ep_reward
        log_data[key][step]['std_ep_reward'] = std_ep_reward
        log_data[key][step]['env_step'] = step * args.action_repeat

        np.save(filename,log_data)
        return log_data[key][step]

    eval_data = run_eval_loop(sample_stochastically=False)
    L.dump(step)
    return eval_data


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'rad_sac':
        return RadSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            markov_params=args.markov_params,
            sisa_params=args.sisa_params
        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)

    pre_transform_image_size = args.pre_transform_image_size if 'crop' in args.data_augs else args.image_size
    pre_image_size = args.pre_transform_image_size # record the pre transform image size for translation

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat
    )

    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type + '-' + args.tag
    args.work_dir = args.work_dir + '/'  + exp_name

    os.makedirs(args.work_dir, exist_ok=True)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on {}'.format(device))

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,pre_transform_image_size,pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        pre_image_size=pre_image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    if args.load_model:
        agent.load_encoder(args.load_model)

    L = Logger(args.work_dir, use_tb=args.save_tb)

    did_pretrain = False

    episode, episode_reward, done = 0, 0, True
    best_avg_reward = -np.inf
    start_time = time.time()

    for step in tqdm(range(args.num_train_steps)):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            eval_data = evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            if args.save_model:
                is_best = (eval_data['mean_ep_reward'] > best_avg_reward)
                if is_best:
                    best_avg_reward = eval_data['mean_ep_reward']
                agent.save(model_dir, step, is_best=is_best)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs / 255.)

        # run training update
        if step >= args.init_steps:
            if not did_pretrain and args.markov and args.markov_pretrain_steps > 0:
                print('Pretraining Markov abstraction...')
                for pretrain_step in tqdm(range(args.markov_pretrain_steps)):
                    pretrain_obs, pretrain_action, _, pretrain_next_obs, _ = replay_buffer.sample_rad(agent.augs_funcs, args.markov_pretrain_batch_size)
                    agent.update_markov_head(pretrain_obs, pretrain_action, pretrain_next_obs, L, pretrain_step)
                if args.markov_catchup_steps > 0:
                    print('Catching up on agent updates...')
                    for catchup_step in tqdm(range(step-args.markov_catchup_steps, step)):
                        agent.update(replay_buffer, L, catchup_step)
                did_pretrain = True

            if not did_pretrain and args.sisa and args.sisa_pretrain_steps > 0:
                print('Pretraining SISA abstraction...')
                for pretrain_step in tqdm(range(args.sisa_pretrain_steps)):
                    pretrain_obs, pretrain_action, _, pretrain_next_obs, _ = replay_buffer.sample_rad(agent.augs_funcs, args.sisa_pretrain_batch_size)
                    agent.pretrain_sisa(pretrain_obs, pretrain_action, pretrain_next_obs, L, pretrain_step)
                if args.sisa_catchup_steps > 0:
                    print('Catching up on agent updates...')
                    for catchup_step in tqdm(range(step-args.sisa_catchup_steps, step)):
                        agent.update(replay_buffer, L, catchup_step)
                did_pretrain = True

            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')

    main()
