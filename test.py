import os

from habitat_baselines.common.environments import RearrangeRLEnv
from habitat_baselines.utils.gym_adapter import HabGymWrapper

import habitat
import habitat_baselines.utils.gym_definitions as habitat_gym
import numpy as np


config = habitat.get_config(
        os.path.join(
            habitat_gym.config_base_dir,
            "configs/tasks/rearrange/pick.yaml",
        )
    )
config.defrost()
config.TASK.ACTIONS.ARM_ACTION.ARM_CONTROLLER = "ArmRAPSAction"
config.TASK.ACTIONS.ARM_ACTION.TYPE = "ArmRAPSAction"
config.TASK.ACTIONS.ARM_ACTION.GOTO_POSE_ITERATIONS = 50
config.TASK.ACTIONS.ARM_ACTION.ACTION_SCALE = 0.5
config.TASK.ACTIONS.ARM_ACTION.EE_CTRL_LIM = 0.005
config.TASK.ACTIONS.ARM_ACTION.EE_QUAT_LIM = 0.005
config.ENVIRONMENT.MAX_EPISODE_STEPS = 5
config.TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER = "MagicGraspAction"
config.RL.GYM_OBS_KEYS = ('is_holding', 'obj_goal_pos_sensor', 'relative_resting_position', 'ee_pos', 'robot_head_depth')
config.DATASET.DATA_PATH = "data/datasets/rearrange_pick/replica_cad/v0/rearrange_pick_replica_cad_v0/pick.json.gz"
config.freeze()
env = RearrangeRLEnv(
    config
)
env = HabGymWrapper(env, config)
done = False
observations = env.reset()  # noqa: F841
observations = env.reset()  # noqa: F841
print("Agent acting inside environment.")
count_steps = 0

from moviepy.editor import *
for i in range(5):
    a = env.action_space.sample()
    a = np.zeros_like(a)
    # a[11] = 1
    # a[13+13] = .25
    a[3] = 1
    a[13+7] = .1
    o, r, d, i = env.step(a)  # noqa: F841
    # img_array = env._env.habitat_env.task.actions['ARM_ACTION'].img_array
    # clip = ImageSequenceClip(list(img_array), fps=20)
    # clip.write_gif('test.gif', fps=20)
    count_steps += 1
    done = d
    if done:
        env.reset()
        print(count_steps, i['rearrangepick_success'])
