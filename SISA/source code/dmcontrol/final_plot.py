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
    filename = 'results/sac/' + experiment_name + '/eval.csv'
    data = pd.read_csv(filename, dtype={'step': int, 'episode_reward': float, 'episode': int})
    data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
    data['alg'] = 'SAC (expert)'
    return roll_and_tag(data, domain, seed, action_repeat)

def load_rad(domain, seed, action_repeat, experiment_name):
    filename = 'results/rad/' + domain + experiment_name + '/eval.log'
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    data['alg'] = 'RAD'
    return roll_and_tag(data, domain, seed, action_repeat)

def load_curl(domain, seed, action_repeat, experiment_name):
    filename = 'results/curl/' + domain.replace('pole','').split('-')[0] + '/' + domain + experiment_name + '/eval.log'
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    data['alg'] = 'CURL'
    return roll_and_tag(data, domain, seed, action_repeat)

def load_zhang(alg, alg_dirname, domain, seed):
    filename = 'results/dbc/' + domain.replace('-','_') + '/' + alg_dirname + '/seed_{}'.format(seed) + '/eval.log'
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
    data['alg'] = alg
    return roll_and_tag(data, domain, seed, 2)

def load_ball_in_cup_dbc(seed):
    filename = 'results/dbc-7/dbc-ball-original_{}/eval.log'.format(seed)
    data = pd.read_json(filename, lines=True)
    data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
    data['alg'] = 'DBC'
    return roll_and_tag(data, 'ball_in_cup-catch', seed, 2)

def load_markov(run):
    data = pd.read_json(run['filename'], lines=True)
    data['alg'] = 'Markov'
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

sac_runs = [
    #      domain        seed  action            experiment_name
    #                          repeat
    ('ball_in_cup-catch',    1,     4,  '1351_sac_ball_in_cup_catch_test_exp_1'),
    ('ball_in_cup-catch',    2,     4,  '1351_sac_ball_in_cup_catch_test_exp_2'),
    ('ball_in_cup-catch',    3,     4,  '1351_sac_ball_in_cup_catch_test_exp_3'),
    ('ball_in_cup-catch',    4,     4,  '1351_sac_ball_in_cup_catch_test_exp_4'),
    ('ball_in_cup-catch',    5,     4,  '1351_sac_ball_in_cup_catch_test_exp_5'),
    ('ball_in_cup-catch',    6,     4,  '1351_sac_ball_in_cup_catch_test_exp_6'),
    ('ball_in_cup-catch',    7,     4,  '1539_sac_ball_in_cup_catch_test_exp_7'),
    ('ball_in_cup-catch',    8,     4,  '2029_sac_ball_in_cup_catch_test_exp_8'),
    ('ball_in_cup-catch',    9,     4,  '2032_sac_ball_in_cup_catch_test_exp_9'),
    ('ball_in_cup-catch',    10,    4,  '2032_sac_ball_in_cup_catch_test_exp_10'),
    ('cartpole-swingup',     1,     8,  '1351_sac_cartpole_swingup_test_exp_1'),
    ('cartpole-swingup',     2,     8,  '1351_sac_cartpole_swingup_test_exp_2'),
    ('cartpole-swingup',     3,     8,  '1351_sac_cartpole_swingup_test_exp_3'),
    ('cartpole-swingup',     4,     8,  '1351_sac_cartpole_swingup_test_exp_4'),
    ('cartpole-swingup',     5,     8,  '1351_sac_cartpole_swingup_test_exp_5'),
    ('cartpole-swingup',     6,     8,  '1351_sac_cartpole_swingup_test_exp_6'),
    ('cartpole-swingup',     7,     8,  '1351_sac_cartpole_swingup_test_exp_7'),
    ('cartpole-swingup',     8,     8,  '1351_sac_cartpole_swingup_test_exp_8'),
    ('cartpole-swingup',     9,     8,  '1351_sac_cartpole_swingup_test_exp_9'),
    ('cartpole-swingup',     10,    8,  '1351_sac_cartpole_swingup_test_exp_10'),
    ('cheetah-run',          1,     4,  '2035_sac_cheetah_run_test_exp_1'),
    ('cheetah-run',          2,     4,  '2038_sac_cheetah_run_test_exp_2'),
    ('cheetah-run',          3,     4,  '2039_sac_cheetah_run_test_exp_3'),
    ('cheetah-run',          4,     4,  '2039_sac_cheetah_run_test_exp_4'),
    ('cheetah-run',          5,     4,  '2043_sac_cheetah_run_test_exp_5'),
    ('cheetah-run',          6,     4,  '2044_sac_cheetah_run_test_exp_6'),
    ('cheetah-run',          7,     4,  '2045_sac_cheetah_run_test_exp_7'),
    ('cheetah-run',          8,     4,  '2048_sac_cheetah_run_test_exp_8'),
    ('cheetah-run',          9,     4,  '2050_sac_cheetah_run_test_exp_9'),
    ('cheetah-run',          10,    4,  '2053_sac_cheetah_run_test_exp_10'),
    ('finger-spin',          1,     2,  '2053_sac_finger_spin_test_exp_1'),
    ('finger-spin',          2,     2,  '2058_sac_finger_spin_test_exp_2'),
    ('finger-spin',          3,     2,  '2103_sac_finger_spin_test_exp_3'),
    ('finger-spin',          4,     2,  '1435_sac_finger_spin_test_exp_4'),
    ('finger-spin',          5,     2,  '1451_sac_finger_spin_test_exp_5'),
    ('finger-spin',          6,     2,  '2246_sac_finger_spin_test_exp_6'),
    ('finger-spin',          7,     2,  '0237_sac_finger_spin_test_exp_7'),
    ('finger-spin',          8,     2,  '0241_sac_finger_spin_test_exp_8'),
    ('finger-spin',          9,     2,  '0247_sac_finger_spin_test_exp_9'),
    ('finger-spin',          10,    2,  '0252_sac_finger_spin_test_exp_10'),
    ('reacher-easy',         1,     4,  '0252_sac_reacher_easy_test_exp_1'),
    ('reacher-easy',         2,     4,  '0258_sac_reacher_easy_test_exp_2'),
    ('reacher-easy',         3,     4,  '0304_sac_reacher_easy_test_exp_3'),
    ('reacher-easy',         4,     4,  '0311_sac_reacher_easy_test_exp_4'),
    ('reacher-easy',         5,     4,  '0311_sac_reacher_easy_test_exp_5'),
    ('reacher-easy',         6,     4,  '0312_sac_reacher_easy_test_exp_6'),
    ('reacher-easy',         7,     4,  '0316_sac_reacher_easy_test_exp_7'),
    ('reacher-easy',         8,     4,  '0317_sac_reacher_easy_test_exp_8'),
    ('reacher-easy',         9,     4,  '0318_sac_reacher_easy_test_exp_9'),
    ('reacher-easy',         10,    4,  '0318_sac_reacher_easy_test_exp_10'),
    ('walker-walk',          1,     2,  '0323_sac_walker_walk_test_exp_1'),
    ('walker-walk',          2,     2,  '0340_sac_walker_walk_test_exp_2'),
    ('walker-walk',          3,     2,  '0443_sac_walker_walk_test_exp_3'),
    ('walker-walk',          4,     2,  '0556_sac_walker_walk_test_exp_4'),
    ('walker-walk',          5,     2,  '0847_sac_walker_walk_test_exp_5'),
    ('walker-walk',          6,     2,  '0848_sac_walker_walk_test_exp_6'),
    ('walker-walk',          7,     2,  '0900_sac_walker_walk_test_exp_7'),
    ('walker-walk',          8,     2,  '0902_sac_walker_walk_test_exp_8'),
    ('walker-walk',          9,     2,  '0911_sac_walker_walk_test_exp_9'),
    ('walker-walk',          10,    2,  '0915_sac_walker_walk_test_exp_10'),
]


curl_runs = [
    #      domain        seed  action      experiment_name
    #                          repeat
    ('ball_in_cup-catch',     1,     4, '-03-21-im84-b128-s1-pixel'),
    ('ball_in_cup-catch',     2,     4, '-03-21-im84-b128-s2-pixel'),
    ('ball_in_cup-catch',     3,     4, '-03-21-im84-b128-s3-pixel'),
    ('ball_in_cup-catch',     4,     4, '-03-21-im84-b128-s4-pixel'),
    ('ball_in_cup-catch',     5,     4, '-03-21-im84-b128-s5-pixel'),
    ('ball_in_cup-catch',     6,     4, '-03-21-im84-b128-s6-pixel'),
    ('ball_in_cup-catch',     7,     4, '-03-21-im84-b128-s7-pixel'),
    ('ball_in_cup-catch',     8,     4, '-03-21-im84-b128-s8-pixel'),
    ('ball_in_cup-catch',     9,     4, '-03-21-im84-b128-s9-pixel'),
    ('ball_in_cup-catch',     10,    4, '-03-21-im84-b128-s10-pixel'),
    ('cartpole-swingup',      1,     8, '-03-20-im84-b128-s1-pixel'),
    ('cartpole-swingup',      2,     8, '-03-20-im84-b128-s2-pixel'),
    ('cartpole-swingup',      3,     8, '-03-20-im84-b128-s3-pixel'),
    ('cartpole-swingup',      4,     8, '-03-20-im84-b128-s4-pixel'),
    ('cartpole-swingup',      5,     8, '-03-20-im84-b128-s5-pixel'),
    ('cartpole-swingup',      6,     8, '-03-20-im84-b128-s6-pixel'),
    ('cartpole-swingup',      7,     8, '-03-20-im84-b128-s7-pixel'),
    ('cartpole-swingup',      8,     8, '-03-21-im84-b128-s8-pixel'),
    ('cartpole-swingup',      9,     8, '-03-21-im84-b128-s9-pixel'),
    ('cartpole-swingup',      10,    8, '-03-27-im84-b128-s10-pixel'),
    ('cheetah-run',           1,     4, '-03-21-im84-b128-s1-pixel'),
    ('cheetah-run',           2,     4, '-03-21-im84-b128-s2-pixel'),
    ('cheetah-run',           3,     4, '-03-21-im84-b128-s3-pixel'),
    ('cheetah-run',           4,     4, '-03-21-im84-b128-s4-pixel'),
    ('cheetah-run',           5,     4, '-03-21-im84-b128-s5-pixel'),
    ('cheetah-run',           6,     4, '-03-21-im84-b128-s6-pixel'),
    ('cheetah-run',           7,     4, '-03-21-im84-b128-s7-pixel'),
    ('cheetah-run',           8,     4, '-03-21-im84-b128-s8-pixel'),
    ('cheetah-run',           9,     4, '-03-21-im84-b128-s9-pixel'),
    ('cheetah-run',           10,    4, '-03-21-im84-b128-s10-pixel'),
    ('finger-spin',           1,     2, '-03-20-im84-b128-s1-pixel'),
    ('finger-spin',           2,     2, '-03-20-im84-b128-s2-pixel'),
    ('finger-spin',           3,     2, '-03-20-im84-b128-s3-pixel'),
    ('finger-spin',           4,     2, '-03-20-im84-b128-s4-pixel'),
    ('finger-spin',           5,     2, '-03-20-im84-b128-s5-pixel'),
    ('finger-spin',           6,     2, '-03-20-im84-b128-s6-pixel'),
    ('finger-spin',           7,     2, '-03-20-im84-b128-s7-pixel'),
    ('finger-spin',           8,     2, '-03-20-im84-b128-s8-pixel'),
    ('finger-spin',           9,     2, '-03-20-im84-b128-s9-pixel'),
    ('finger-spin',           10,    2, '-03-20-im84-b128-s10-pixel'),
    ('reacher-easy',          1,     4, '-03-20-im84-b128-s1-pixel'),
    ('reacher-easy',          2,     4, '-03-20-im84-b128-s2-pixel'),
    ('reacher-easy',          3,     4, '-03-20-im84-b128-s3-pixel'),
    ('reacher-easy',          4,     4, '-03-20-im84-b128-s4-pixel'),
    ('reacher-easy',          5,     4, '-03-20-im84-b128-s5-pixel'),
    ('reacher-easy',          6,     4, '-03-20-im84-b128-s6-pixel'),
    ('reacher-easy',          7,     4, '-03-21-im84-b128-s7-pixel'),
    ('reacher-easy',          8,     4, '-03-21-im84-b128-s8-pixel'),
    ('reacher-easy',          9,     4, '-03-21-im84-b128-s9-pixel'),
    ('reacher-easy',          10,    4, '-03-21-im84-b128-s10-pixel'),
    ('walker-walk',           1,     2, '-03-21-im84-b128-s1-pixel'),
    ('walker-walk',           2,     2, '-03-21-im84-b128-s2-pixel'),
    ('walker-walk',           3,     2, '-03-21-im84-b128-s3-pixel'),
    ('walker-walk',           4,     2, '-03-21-im84-b128-s4-pixel'),
    ('walker-walk',           5,     2, '-03-21-im84-b128-s5-pixel'),
    ('walker-walk',           6,     2, '-03-21-im84-b128-s6-pixel'),
    ('walker-walk',           7,     2, '-03-21-im84-b128-s7-pixel'),
    ('walker-walk',           8,     2, '-03-21-im84-b128-s8-pixel'),
    ('walker-walk',           9,     2, '-03-21-im84-b128-s9-pixel'),
    ('walker-walk',           10,    2, '-03-21-im84-b128-s10-pixel'),
]

zhang_algs = [
    ('DBC', 'bisim_coef0.5_probabilistic_nobg'),
    ('DeepMDP', 'deepmdp_identity_nobg'),
    ('CPC', 'baseline_contrastive_nobg'),
    ('SAC-AE', 'baseline_pixel_nobg'),
]
zhang_domains = ['cartpole-swingup', 'cheetah-run', 'finger-spin', 'reacher-easy', 'walker-walk']


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
# markov_domains = [
#     # domain-task,        exp,   seeds,  action_repeat
#     ('cheetah-run',        28, rg(1, 6), 4),
#     ('finger-spin',        28, rg(1, 6), 2),
#     ('walker-walk',        28, rg(1, 6), 2),
#     ('reacher-easy',       28, rg(1, 6), 4),
#     ('ball_in_cup-catch',  28, rg(1, 6), 4),
#     ('cartpole-swingup',   28, rg(1, 6), 8),
# ]
markov_runs = [{
    'alg': 'rad+markov',
    'action_repeat': action_repeat,
    'domain': domain,
    'seed': seed,
    'filename': list(sorted(glob.glob('results/markov/{}-*_{}/eval.log'.format(domain, seed))))[-1]
} for (domain, exp, seeds, action_repeat) in markov_domains for seed in seeds]


for run in rad_runs:
    dfs.append(load_rad(*run))

for run in sac_runs:
    dfs.append(load_sac(*run))

for run in curl_runs:
    dfs.append(load_curl(*run))

for alg, alg_dirname in zhang_algs:
    for domain in zhang_domains:
        for seed in rg(1,10):
            if (alg, domain, seed) == ('CPC', 'cheetah-run', 9):
                continue
            dfs.append(load_zhang(alg, alg_dirname, domain, seed))

for seed in rg(1,10):
    dfs.append(load_ball_in_cup_dbc(seed))

for run in markov_runs:
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

p = sns.color_palette('Set1', n_colors=9, desat=0.5)
_, blue, green, purple, orange, yellow, brown, pink, gray = p

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
    'RAD',
    'CURL',
    'DBC',
    'DeepMDP',
    'CPC',
    'SAC-AE',
    'SAC (expert)',
]

color_order = red, purple, orange, green, blue, yellow, brown, gray
subset.reset_index(inplace=True)

sns.set(context="notebook", style="white", font_scale=1.3)
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.left'] = True
plt.rcParams['axes.linewidth'] = 2

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
    linewidth=2,
    legend=False,
    facet_kws={'sharex': False, 'sharey': True},
)

leg = g.axes.flat[3].legend(alg_order, ncol=2, fontsize=12)
N = len(leg.legendHandles) // 2
leg.legendHandles[N:] = []
leg.set_draggable(True)
# plt.title(d)
# plt.ylim([0,300])
plt.tight_layout()
plt.subplots_adjust(hspace=0.22)
plt.show()
