# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

import gym
import numpy as np
from dm_control import mujoco
from scipy.spatial.transform import Rotation

from bisk.features import Featurizer
from bisk.features.joints import JointsRelZFeaturizer


class BodyFeetWalkerFeaturizer(Featurizer):
    """
    Featurizer for the Walker robot that extracts its body and feet positions.
    """

    def __init__(self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        assert robot == 'walker', f'Walker robot expected, got "{robot}"'
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def __call__(self) -> np.ndarray:
        """
        Get the feature representation.
        
        Returns:
            np.ndarray: The features.
        """

        # TODO: Provide more explanation about the specific features being extracted here.
        root = self.p.named.data.qpos[[f'{self.prefix}/root{p}' for p in 'zxy']]
        torso_frame = self.p.named.data.xmat[f'{self.prefix}/torso'].reshape(3, 3)
        torso_pos = self.p.named.data.xpos[f'{self.prefix}/torso']
        positions = []
        for side in ('left', 'right'):
            torso_to_limb = self.p.named.data.xpos[f'{self.prefix}/{side}_foot'] - torso_pos
            positions.append(torso_to_limb.dot(torso_frame)[[0, 2]])
        extremities = np.hstack(positions)
        return np.concatenate([root, extremities])

    def feature_names(self) -> List[str]:
        """
        Get the names of the features.
        
        Returns:
            List[str]: The feature names.
        """

        names = ['rootz:p', 'rootx:p', 'rooty:p']
        names += [f'left_foot:p{p}' for p in 'xz']
        names += [f'right_foot:p{p}' for p in 'xz']
        return names

# similar explanations can be added for the other classes as well.


class BodyFeetRelZWalkerFeaturizer(BodyFeetWalkerFeaturizer):
    """
    This class is a variation of the BodyFeetWalkerFeaturizer that computes relative
    z-coordinate feature for the Walker robot.
    """

    def __init__(self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        """
        Compute the relative z-coordinate feature.
        
        Returns:
            float: The relative z-coordinate feature.
        """
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        """
        Get the feature representation with relative z-coordinate.
        
        Returns:
            np.ndarray: The features.
        """
        obs = super().__call__()
        obs[0] = self.relz()
        return obs


class BodyFeetHumanoidFeaturizer(Featurizer):
    """
    This class is a Featurizer for a Humanoid robot, specifically extracting the body
    and feet positions.
    """

    def __init__(self, p: mujoco.Physics, robot: str, prefix: str = 'robot', exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        self.for_pos = None
        self.for_twist = None
        self.foot_anchor = 'pelvis'
        self.reference = 'torso'
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    @staticmethod
    def decompose_twist_swing_z(q):
        """
        Decomposes the quaternion 'q' into a twist and a swing.
        
        Args:
            q (np.array): Input quaternion.
        
        Returns:
            tuple: The twist and swing.
        """
        p = [0.0, 0.0, q[2]]
        twist = Rotation.from_quat(np.array([p[0], p[1], p[2], q[3]]))
        swing = Rotation.from_quat(q) * twist.inv()
        return twist, swing

    def __call__(self) -> np.ndarray:
        """
        Get the feature representation for the Humanoid robot.
        
        Returns:
            np.ndarray: The features.
        """
        # TODO: Provide more explanation about the specific features being extracted here.
        root = self.p.data.qpos[0:3]
        if self.for_pos is not None:
            root = root.copy()
            root[0:2] -= self.for_pos
            root[0:2] = self.for_twist.apply(root * np.array([1, 1, 0]))[0:2]
        q = self.p.data.qpos[3:7]
        t, s = self.decompose_twist_swing_z(q[[1, 2, 3, 0]])
        tz = t.as_rotvec()[2]
        e = s.as_euler('yzx')
        sy, sx = e[0], e[2]

        # Feet positions are relative to pelvis position and its heading
        pelvis_q = self.p.named.data.xquat[f'{self.prefix}/{self.foot_anchor}']
        pelvis_t, pelvis_s = self.decompose_twist_swing_z(pelvis_q[[1, 2, 3, 0]])
        pelvis_pos = self.p.named.data.xpos[f'{self.prefix}/{self.foot_anchor}']
        positions = []
        for ex in ('foot',):
            for side in ('left', 'right'):
                pelvis_to_limb = (
                    self.p.named.data.xpos[f'{self.prefix}/{side}_{ex}']
                    - pelvis_pos
                )
                positions.append(pelvis_t.apply(pelvis_to_limb))
        extremities = np.hstack(positions)
        return np.concatenate([root, np.asarray([tz, sy, sx]), extremities])

    def feature_names(self) -> List[str]:
        """
        Get the names of the features.
        
        Returns:
            List[str]: The feature names.
        """
        names = [f'root:p{f}' for f in 'xyz']
        names += [f'root:t{f}' for f in 'z']
        names += [f'root:s{f}' for f in 'yx']
        names += [f'left_foot:p{f}' for f in 'xyz']
        names += [f'right_foot:p{f}' for f in 'xyz']
        return names



class BodyFeetRelZHumanoidFeaturizer(BodyFeetHumanoidFeaturizer):
    """
    The BodyFeetRelZHumanoidFeaturizer class extends BodyFeetHumanoidFeaturizer by adding 
    relative z-height feature. It helps in extracting more complex features from the robot's state.
    
    Attributes:
    -----------
    relzf : JointsRelZFeaturizer
        An object of the JointsRelZFeaturizer class which is used to calculate the relative z-height.
    """

    def __init__(
        self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None
    ):
        """
        Initializes the BodyFeetRelZHumanoidFeaturizer with the provided parameters.
        
        Parameters:
        -----------
        p : mujoco.Physics
            The physics model of the environment.
        robot : str
            The type of the robot.
        prefix : str
            The prefix for the robot's joints.
        exclude : str, optional
            The joints to be excluded, default is None.
        """
        # Call the constructor of the superclass
        super().__init__(p, robot, prefix, exclude)
        # Initialize the JointsRelZFeaturizer object
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        """
        Method to calculate the relative z-height of the robot.
        
        Returns:
        --------
        float
            The relative z-height of the robot.
        """
        # Call the relz method of the JointsRelZFeaturizer object
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        """
        Override the call method to provide the state of the robot as a numpy array.
        
        Returns:
        --------
        np.ndarray
            The state of the robot as a numpy array.
        """
        # Get the state of the robot from the superclass
        obs = super().__call__()
        # Replace the third element of the state with the relative z-height
        obs[2] = self.relz()
        # Return the updated state
        return obs


def bodyfeet_featurizer(
    p: mujoco.Physics, robot: str, prefix: str, *args, **kwargs
):
    """
    Function to create an instance of the appropriate BodyFeetFeaturizer based on the robot type.
    
    Parameters:
    -----------
    p : mujoco.Physics
        The physics model of the environment.
    robot : str
        The type of the robot.
    prefix : str
        The prefix for the robot's joints.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the BodyFeetFeaturizer constructor.
    
    Returns:
    --------
    BodyFeetFeaturizer
        An instance of the appropriate BodyFeetFeaturizer.
    
    Raises:
    -------
    ValueError
        If the robot type is not 'walker' or 'humanoid'/'humanoidpc'.
    """
    # Check the robot type and return the appropriate featurizer
    if robot == 'walker':
        return BodyFeetWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoid' or robot == 'humanoidpc':
        return BodyFeetHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        # Raise an exception if the robot type is not supported
        raise ValueError(f'No bodyfeet featurizer for robot "{robot}"')

def bodyfeet_relz_featurizer(
    p: mujoco.Physics, robot: str, prefix: str, *args, **kwargs
):
    """
    Function to create an instance of the appropriate BodyFeetRelZFeaturizer based on the robot type.
    
    Parameters:
    -----------
    p : mujoco.Physics
        The physics model of the environment.
    robot : str
        The type of the robot.
    prefix : str
        The prefix for the robot's joints.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the BodyFeetRelZFeaturizer constructor.
    
    Returns:
    --------
    BodyFeetRelZFeaturizer
        An instance of the appropriate BodyFeetRelZFeaturizer.
    
    Raises:
    -------
    ValueError
        If the robot type is not 'walker' or 'humanoid'/'humanoidpc'.
    """
    # Check the robot type and return the appropriate featurizer
    if robot == 'walker':
        return BodyFeetRelZWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == 'humanoid' or robot == 'humanoidpc':
        return BodyFeetRelZHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        # Raise an exception if the robot type is not supported
        raise ValueError(f'No bodyfeet-relz featurizer for robot "{robot}"')

