# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import re
from itertools import combinations
from typing import Dict, List, Set, Tuple

log = logging.getLogger(__name__)

# Mapping of robot types to their respective delta feature indices.
# Delta features represent change in joint observations which can be 
# useful for learning algorithms.
g_delta_feats = {
    'Walker': [1],
    'Humanoid': [0, 1],
    'HumanoidPC': [0, 1],
}

def def_ranges(
    inp: List[Tuple],
    delta_feats: List[int] = None,
    twist_feats: List[int] = None,
) -> Dict[str, List]:
    """
    Function to define ranges for various features based on the provided input.
    
    Parameters:
    -----------
    inp : List[Tuple]
        A list of tuples where each tuple contains information about a feature.
    delta_feats : List[int], optional
        A list of indices for delta features, default is None.
    twist_feats : List[int], optional
        A list of indices for twist features, default is None.
    
    Returns:
    --------
    Dict[str, List]
        A dictionary containing lists for the structure, min, max, delta features, 
        and twist features of the input features.
    """
    return {
        'str': [g[1] for g in inp],  # Extracting structure information
        'min': [g[2] for g in inp],  # Extracting minimum values
        'max': [g[3] for g in inp],  # Extracting maximum values
        'delta_feats': delta_feats if delta_feats else [],  # Using provided delta features or defaulting to empty list
        'twist_feats': twist_feats if twist_feats else [],  # Using provided twist features or defaulting to empty list
    }
# Definitions of goal ranges for two different types of robots. 
# Each tuple in the list contains the index of the parameter, 
# its name, and the minimum and maximum values it can take.

# For a walker type robot, we have:
g_goal_ranges_bodyfeet_walker: List[Tuple] = [
    (0, 'rootz:p', +0.95, +1.50),  # 0.9 is fall-over
    (1, 'rootx:p', -3.00, +3.00),
    (2, 'rooty:p', -1.30, +1.30),  # 1.4 is fall-over
    (3, 'left_foot:px', -0.72, +0.99),
    (4, 'left_foot:pz', -1.30, +0.00),
    (5, 'right_foot:px', -0.72, +0.99),
    (6, 'right_foot:pz', -1.30, +0.00),
]

g_goal_ranges_bodyfeet_humanoid: List[Tuple] = [
    (0, 'root:px', -3.00, +3.00),
    (1, 'root:py', -3.00, +3.00),  # we can rotate so equalize this
    (2, 'root:pz', +0.95, +1.50),  # 0.9 is falling over
    (3, 'root:tz', -1.57, +1.57),  # stay within +- pi/2, i.e. don't turn around
    (4, 'root:sy', -1.57, +1.57),  # Swing around Y axis
    (5, 'root:sx', -0.50, +0.50),  # Swing around X axis (more or less random v)
    (6, 'left_foot:px', -1.00, +1.00),
    (7, 'left_foot:py', -1.00, +1.00),
    (8, 'left_foot:pz', -1.00, +0.20),
    (9, 'right_foot:px', -1.00, +1.00),
    (10, 'right_foot:py', -1.00, +1.00),
    (11, 'right_foot:pz', -1.00, +0.20),
]

# The goal spaces are defined for each type of robot. 
# It uses the def_ranges function (not shown) to generate the ranges.
g_goal_spaces_bodyfeet: Dict[str, Dict[str, List]] = {
    'Walker': def_ranges(g_goal_ranges_bodyfeet_walker, [1]),
    'Humanoid': def_ranges(g_goal_ranges_bodyfeet_humanoid, [0, 1], [3]),
    'HumanoidPC': def_ranges(g_goal_ranges_bodyfeet_humanoid, [0, 1], [3]),
}

# A dictionary that groups the goal spaces by categories (in this case 'bodyfeet' and 'bodyfeet-relz')
g_goal_spaces: Dict[str, Dict[str, Dict[str, List]]] = {
    'bodyfeet': g_goal_spaces_bodyfeet,
    'bodyfeet-relz': g_goal_spaces_bodyfeet,
}

def subsets_task_map(
    features: str, robot: str, spec: str, rank_min: int, rank_max: int
) -> Tuple[List[str], Dict[str, int]]:
    """
    Parses a spec of features and returns the necessary data to construct goal spaces.

    :param features: A string representing the category of features.
    :param robot: A string representing the type of robot ('Walker', 'Humanoid', etc.).
    :param spec: A string specifying the set of features, can be 'all', 'torso', a #-separated list, or a regex.
    :param rank_min: The minimum number of features for a valid subset.
    :param rank_max: The maximum number of features for a valid subset.

    :return: A tuple containing a list of valid feature subsets and a task map mapping each feature to an index.

    :raises ValueError: If a feature is out of range or there are less features to control than the requested rank.
    """
    # gs represents the goal space for the robot's features
    gs = g_goal_spaces[features][robot]

    # n is the total number of features in the goal space
    n = len(gs['str'])

    # dims is the list of features (or dimensions)
    dims: List[str] = []
    if spec == 'all':
        # If spec is 'all', we consider all features
        dims = [str(i) for i in range(n)]
    elif spec == 'torso':
        # If spec is 'torso', we consider only features related to the robot's torso/root
        dims = [
            str(i)
            for i in range(n)
            if gs['str'][i].startswith(':')
            or gs['str'][i].startswith('torso:')
            or gs['str'][i].startswith('root')
        ]
    else:
        try:
            # If spec is a #-separated list, we split it and sort the feature indices
            dims = []
            for d in str(spec).split('#'):
                ds = sorted(map(int, re.split('[-+]', d)))
                dims.append('+'.join(map(str, ds)))
        except:
            # If spec is a regex, we find all matching features
            dims = [
                str(i)
                for i in range(n)
                if re.match(spec, gs['str'][i]) is not None
            ]

    # Helper function to check if a feature is controllable (i.e., has a non-zero range)
    def is_controllable(d: int) -> bool:
        if d < 0:
            raise ValueError(f'Feature {d} out of range')
        if d >= len(gs['min']):
            return False
        # Return whether range is non-zero
        return gs['min'][d] != gs['max'][d]


    # Identify and remove uncontrollable features
    uncontrollable = set()
    for dim in dims:
        for idx in map(int, dim.split('+')):
            if not is_controllable(idx):
                uncontrollable.add(dim)
                log.warning(f'Removing uncontrollable feature {dim}')
                break
    dims = [dim for dim in dims if not dim in uncontrollable]

    # If there are less controllable features than the requested rank, raise an error
    if len(dims) < rank_min:
        raise ValueError('Less features to control than the requested rank')

    # Create a set of unique dimensions (features)
    udims: Set[int] = set()
    for dim in dims:
        for idx in map(int, dim.split('+')):
            udims.add(idx)

    # Create a task map that maps each feature to an index
    task_map: Dict[str, int] = {}
    for idx in sorted(udims):
        task_map[str(idx)] = len(task_map)
    # Helper function to unify feature combinations and sort them
    def unify(comb) -> str:
        udims: Set[str] = set()
        for c in comb:
            for d in c.split('+'):
                if d in udims:
                    raise ValueError(f'Overlapping feature dimensions: {comb}')
                udims.add(d)
        return ','.join(
            sorted(comb, key=lambda x: [int(i) for i in x.split('+')])
        )

    # If both rank_min and rank_max are greater than 0, generate all possible combinations of features
    if rank_min > 0 and rank_max > 0:
        cdims = []
        for r in range(rank_min, rank_max + 1):
            for comb in combinations(dims, r):
                # Duplications are ok now
                cdims.append(','.join(comb))
        dims = cdims

    # Return the final list of valid feature subsets and the task map
    return dims, task_map

