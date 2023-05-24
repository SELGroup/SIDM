import glob
import json

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns

WINDOW_SIZE = 5

dfs = []

def rg(seed_first, seed_last):
    return list(range(seed_first, seed_last+1))

def roll_and_tag(data, domain, seed, action_repeat):
    data.reward = data.reward.rolling(WINDOW_SIZE).mean()
    data['domain'] = domain
    data['seed'] = seed
    data['action_repeat'] = action_repeat
    return data

def load_sac(domain, seed, action_repeat, experiment_name):
    filename = 'tmp/sac/' + experiment_name + '/eval.csv'
    data = pd.read_csv(filename, dtype={'step': int, 'episode_reward': float, 'episode': int})
    data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
    data['alg'] = 'SAC (expert)'
    return roll_and_tag(data, domain, seed, action_repeat)

def load_rad(domain, seed, action_repeat, experiment_name):
    filename = 'tmp/rad/' + domain + experiment_name + '/eval.log'
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    data['alg'] = 'RAD'
    return roll_and_tag(data, domain, seed, action_repeat)

def load_curl(domain, seed, action_repeat, experiment_name):
    filename = 'tmp/curl/' + domain.replace('pole','').split('-')[0] + '/' + domain + experiment_name + '/eval.log'
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    data['alg'] = 'CURL'
    return roll_and_tag(data, domain, seed, action_repeat)

def load_zhang(alg, alg_dirname, domain, seed):
    filename = 'tmp/dbc/' + domain.replace('-','_') + '/' + alg_dirname + '/seed_{}'.format(seed) + '/eval.log'
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
    data['alg'] = alg
    return roll_and_tag(data, domain, seed, 2)

def load_ball_in_cup_dbc(seed):
    filename = 'tmp/dbc-7/dbc-ball-original_{}/eval.log'.format(seed)
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
    data['alg'] = 'DBC'
    return roll_and_tag(data, 'ball_in_cup-catch', seed, 2)

def load_markov(run):
    data = pd.read_json(run['filename'], lines=True)
    data['alg'] = run['alg']
    data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
    return roll_and_tag(data, run['domain'], run['seed'], run['action_repeat'])

rad_runs = [
    #      domain        seed action          experiment_name
    #                         repeat
    ('ball_in_cup-catch',   1,   4, '-01-30-im108-b128-s1-pixel-tuning-rad_1'),
    ('ball_in_cup-catch',   2,   4, '-01-30-im108-b128-s2-pixel-tuning-rad_2'),
    ('ball_in_cup-catch',   3,   4, '-01-30-im108-b128-s3-pixel-tuning-rad_3'),
    ('ball_in_cup-catch',   4,   4, '-02-02-im108-b128-s4-pixel-rad-seeds_1'),
    ('ball_in_cup-catch',   5,   4, '-02-02-im108-b128-s5-pixel-rad-seeds_2'),
    ('ball_in_cup-catch',   6,   4, '-02-02-im108-b128-s6-pixel-rad-seeds_3'),
    ('ball_in_cup-catch',   7,   4, '-02-02-im108-b128-s7-pixel-rad-seeds_4'),
    ('ball_in_cup-catch',   8,   4, '-03-09-im108-b128-s8-pixel-rad-seeds_1'),
    ('ball_in_cup-catch',   9,   4, '-03-09-im108-b128-s9-pixel-rad-seeds_2'),
    ('ball_in_cup-catch',   10,  4, '-03-09-im108-b128-s10-pixel-rad-seeds_3'),
    ('cartpole-swingup',    1,   8, '-01-30-im108-b128-s1-pixel-tuning-rad_4'),
    ('cartpole-swingup',    2,   8, '-01-30-im108-b128-s2-pixel-tuning-rad_5'),
    ('cartpole-swingup',    3,   8, '-01-30-im108-b128-s3-pixel-tuning-rad_6'),
    ('cartpole-swingup',    4,   8, '-02-02-im108-b128-s4-pixel-rad-seeds_5'),
    ('cartpole-swingup',    5,   8, '-02-02-im108-b128-s5-pixel-rad-seeds_6'),
    ('cartpole-swingup',    6,   8, '-02-02-im108-b128-s6-pixel-rad-seeds_7'),
    ('cartpole-swingup',    7,   8, '-02-02-im108-b128-s7-pixel-rad-seeds_8'),
    ('cartpole-swingup',    8,   8, '-03-09-im108-b128-s8-pixel-rad-seeds_4'),
    ('cartpole-swingup',    9,   8, '-03-09-im108-b128-s9-pixel-rad-seeds_5'),
    ('cartpole-swingup',    10,  8, '-03-09-im108-b128-s10-pixel-rad-seeds_6'),
    ('cheetah-run',         1,   4, '-01-30-im108-b128-s1-pixel-tuning-rad_7'),
    ('cheetah-run',         2,   4, '-01-30-im108-b128-s2-pixel-tuning-rad_8'),
    ('cheetah-run',         3,   4, '-01-30-im108-b128-s3-pixel-tuning-rad_9'),
    ('cheetah-run',         4,   4, '-02-02-im108-b128-s4-pixel-rad-seeds_9'),
    ('cheetah-run',         5,   4, '-02-02-im108-b128-s5-pixel-rad-seeds_10'),
    ('cheetah-run',         6,   4, '-02-02-im108-b128-s6-pixel-rad-seeds_11'),
    ('cheetah-run',         7,   4, '-02-02-im108-b128-s7-pixel-rad-seeds_12'),
    ('cheetah-run',         8,   4, '-03-10-im108-b128-s8-pixel-rad-seeds_7'),
    ('cheetah-run',         9,   4, '-03-10-im108-b128-s9-pixel-rad-seeds_8'),
    ('cheetah-run',         10,  4, '-03-10-im108-b128-s10-pixel-rad-seeds_9'),
    ('finger-spin',         1,   2, '-01-31-im108-b128-s1-pixel-tuning-finger_1'),
    ('finger-spin',         2,   2, '-01-31-im108-b128-s2-pixel-tuning-finger_2'),
    ('finger-spin',         3,   2, '-01-31-im108-b128-s3-pixel-tuning-finger_3'),
    ('finger-spin',         4,   2, '-02-02-im108-b128-s4-pixel-rad-seeds_13'),
    ('finger-spin',         5,   2, '-02-02-im108-b128-s5-pixel-rad-seeds_14'),
    ('finger-spin',         6,   2, '-02-02-im108-b128-s6-pixel-rad-seeds_15'),
    ('finger-spin',         7,   2, '-02-02-im108-b128-s7-pixel-rad-seeds_16'),
    ('finger-spin',         8,   2, '-03-10-im108-b128-s8-pixel-rad-seeds_10'),
    ('finger-spin',         9,   2, '-03-14-im108-b128-s9-pixel-rad-seeds_11'),
    ('finger-spin',         10,  2, '-03-10-im108-b128-s10-pixel-rad-seeds_12'),
    ('reacher-easy',        1,   4, '-01-31-im108-b128-s1-pixel-tuning-reacher_1'),
    ('reacher-easy',        2,   4, '-01-31-im108-b128-s2-pixel-tuning-reacher_2'),
    ('reacher-easy',        3,   4, '-01-31-im108-b128-s3-pixel-tuning-reacher_3'),
    ('reacher-easy',        4,   4, '-02-02-im108-b128-s4-pixel-rad-seeds_17'),
    ('reacher-easy',        5,   4, '-02-02-im108-b128-s5-pixel-rad-seeds_18'),
    ('reacher-easy',        6,   4, '-02-02-im108-b128-s6-pixel-rad-seeds_19'),
    ('reacher-easy',        7,   4, '-02-02-im108-b128-s7-pixel-rad-seeds_20'),
    ('reacher-easy',        8,   4, '-03-14-im108-b128-s8-pixel-rad-seeds_13'),
    ('reacher-easy',        9,   4, '-03-14-im108-b128-s9-pixel-rad-seeds_14'),
    ('reacher-easy',        10,  4, '-03-14-im108-b128-s10-pixel-rad-seeds_15'),
    ('walker-walk',         1,   2, '-01-31-im84-b128-s1-pixel-tuning-rad_1'),
    ('walker-walk',         2,   2, '-01-31-im84-b128-s2-pixel-tuning-rad_2'),
    ('walker-walk',         3,   2, '-01-31-im84-b128-s3-pixel-tuning-rad_3'),
    ('walker-walk',         4,   2, '-02-02-im84-b128-s4-pixel-rad-seeds_21'),
    ('walker-walk',         5,   2, '-02-02-im84-b128-s5-pixel-rad-seeds_22'),
    ('walker-walk',         6,   2, '-02-02-im84-b128-s6-pixel-rad-seeds_23'),
    ('walker-walk',         7,   2, '-02-02-im84-b128-s7-pixel-rad-seeds_24'),
    ('walker-walk',         8,   2, '-03-11-im84-b128-s8-pixel-rad-seeds_16'),
    ('walker-walk',         9,   2, '-03-12-im84-b128-s9-pixel-rad-seeds_17'),
    ('walker-walk',         10,  2, '-03-12-im84-b128-s10-pixel-rad-seeds_18'),
]

markov_experiments = {
    10: 'exp10_markov_pretrain_bs512',
    12: 'exp12_markov_pretrain_bs512_inv10',
    16: 'exp16_markov_pretrain_bs512_inv1_relu30',
    26: 'exp26_markov_pretrain_bs512_inv30_relu30',
    27: 'exp27_disable_smoothness',
    28: 'exp28_disable_pretrain',
}
markov_domains = [
    # domain-task,        exp,   seeds,  action_repeat
    ('cheetah-run',        16, rg(1, 10), 4),
    ('finger-spin',        10, rg(1, 10), 2),
    ('walker-walk',        10, rg(1, 10), 2),
    ('reacher-easy',       26, rg(1, 10), 4),
    ('ball_in_cup-catch',  26, rg(1, 10), 4),
    ('cartpole-swingup',   10, rg(1, 10), 8),
]
markov_runs = [{
    'alg': 'Markov',
    'action_repeat': action_repeat,
    'domain': domain,
    'seed': seed,
    'filename': list(sorted(glob.glob('tmp/{}/{}-*_{}/eval.log'.format(markov_experiments[exp], domain, seed))))[-1]
} for (domain, exp, seeds, action_repeat) in markov_domains for seed in seeds]

nosmooth_domains = [
    # domain-task,        exp,   seeds,  action_repeat
    ('cheetah-run',        27, rg(1, 6), 4),
    ('finger-spin',        27, rg(1, 6), 2),
    ('walker-walk',        27, rg(1, 6), 2),
    ('reacher-easy',       27, rg(1, 6), 4),
    ('ball_in_cup-catch',  27, rg(1, 6), 4),
    ('cartpole-swingup',   27, rg(1, 6), 8),
]
nosmooth_runs = [{
    'alg': 'No Smoothness',
    'action_repeat': action_repeat,
    'domain': domain,
    'seed': seed,
    'filename': list(sorted(glob.glob('tmp/{}/{}-*_{}/eval.log'.format(markov_experiments[exp], domain, seed))))[-1]
} for (domain, exp, seeds, action_repeat) in nosmooth_domains for seed in seeds]

nopretrain_domains = [
    # domain-task,        exp,   seeds,  action_repeat
    ('cheetah-run',        28, rg(1, 6), 4),
    ('finger-spin',        28, rg(1, 6), 2),
    ('walker-walk',        28, rg(1, 6), 2),
    ('reacher-easy',       28, rg(1, 6), 4),
    ('ball_in_cup-catch',  28, rg(1, 6), 4),
    ('cartpole-swingup',   28, rg(1, 6), 8),
]
nopretrain_runs = [{
    'alg': 'No Pretraining',
    'action_repeat': action_repeat,
    'domain': domain,
    'seed': seed,
    'filename': list(sorted(glob.glob('tmp/{}/{}-*_{}/eval.log'.format(markov_experiments[exp], domain, seed))))[-1]
} for (domain, exp, seeds, action_repeat) in nopretrain_domains for seed in seeds]


for run in rad_runs:
    dfs.append(load_rad(*run))

for run in markov_runs + nosmooth_runs + nopretrain_runs:
    dfs.append(load_markov(run))

data = pd.concat(dfs, axis=0)

#%%
subset = data
subset = subset.query("(domain == 'cheetah-run' and steps <= 500e3) or steps <= 300e3")
subset = subset.query("domain != 'ball_in_cup-catch' or steps <= 100e3")
subset = subset.query("domain != 'cartpole-swingup' or steps <= 100e3")

subset.loc[subset.domain == 'ball_in_cup-catch', 'domain'] = 'Ball-in-cup, Catch'
subset.loc[subset.domain == 'cartpole-swingup', 'domain'] = 'Cartpole, Swingup'
subset.loc[subset.domain == 'cheetah-run', 'domain'] = 'Cheetah, Run'
subset.loc[subset.domain == 'finger-spin', 'domain'] = 'Finger, Spin'
subset.loc[subset.domain == 'reacher-easy', 'domain'] = 'Reacher, Easy'
subset.loc[subset.domain == 'walker-walk', 'domain'] = 'Walker, Walk'

subset = subset.rename(columns={'reward': 'Reward', 'alg': 'Agent', 'domain': 'Task'})
subset = subset.rename_axis(index={'steps':'Steps'})

#%%

p = sns.color_palette('Set1', n_colors=2)
red, _ = p

p = sns.color_palette('Set1', n_colors=10, desat=0.5)
_, blue, green, purple, orange, yellow, brown, pink, gray, sky = p
sky  = sns.color_palette('deep', n_colors=10, desat=0.9)[-1]
teal = sns.color_palette('colorblind', n_colors=3)[-1]

domain_order = [
    'Cartpole, Swingup',
    'Ball-in-cup, Catch',
    'Cheetah, Run',
    'Finger, Spin',
    'Reacher, Easy',
    'Walker, Walk',
]

alg_order = [
    'Markov',
    'No Smoothness',
    'No Pretraining',
    'RAD',
]

color_order = red, orange, sky, purple
style_order = ['Markov', 'No Smoothness', 'RAD', 'No Pretraining']

g = sns.relplot(
    data=subset,
    x='Steps',
    y='Reward',
    hue='Agent',
    hue_order=alg_order,
    style='Agent',
    # style='markov_lr',
    col='Task',
    col_wrap=3,
    col_order=domain_order,
    style_order=alg_order,
    kind='line',
    # units='seed',
    # estimator=None,
    ci=90,
    # height=10,
    palette=color_order,
    legend=False,
    facet_kws={'sharex': False, 'sharey': True},
)

leg = g.axes.flat[3].legend(alg_order, ncol=2)
N = len(leg.legendHandles) // 2
leg.legendHandles[N:] = []
leg.set_draggable(True)
# plt.title(d)
# plt.ylim([0,300])
plt.tight_layout()
plt.subplots_adjust(hspace=0.22)
plt.show()
