#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# flake8: noqa
# These actions need to be imported since there is a Python evaluation
# statement which dynamically creates the desired grip controller.
from habitat.tasks.rearrange.grip_actions import (
    GripSimulatorTaskAction,
    MagicGraspAction,
    SuctionGraspAction,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import rearrange_collision
import quaternion


@registry.register_task_action
class EmptyAction(SimulatorTaskAction):
    """A No-op action useful for testing and in some controllers where we want
    to wait before the next operation.
    """

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "empty_action": spaces.Box(
                    shape=(1,),
                    low=-1,
                    high=1,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.EMPTY)


@registry.register_task_action
class ArmAction(SimulatorTaskAction):
    """An arm control and grip control into one action space."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        arm_controller_cls = eval(self._config.ARM_CONTROLLER)
        self._sim: RearrangeSim = sim
        self.arm_ctrlr = arm_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )

        if self._config.GRIP_CONTROLLER is not None:
            grip_controller_cls = eval(self._config.GRIP_CONTROLLER)
            self.grip_ctrlr: Optional[
                GripSimulatorTaskAction
            ] = grip_controller_cls(*args, config=config, sim=sim, **kwargs)
        else:
            self.grip_ctrlr = None

        self.disable_grip = False
        if "DISABLE_GRIP" in config:
            self.disable_grip = config["DISABLE_GRIP"]

    def reset(self, *args, **kwargs):
        self.arm_ctrlr.reset(*args, **kwargs)
        if self.grip_ctrlr is not None:
            self.grip_ctrlr.reset(*args, **kwargs)

    @property
    def action_space(self):
        action_spaces = {
            "arm_action": self.arm_ctrlr.action_space,
        }
        if self.grip_ctrlr is not None and self.grip_ctrlr.requires_action:
            action_spaces["grip_action"] = self.grip_ctrlr.action_space
        return spaces.Dict(action_spaces)

    def step(self, arm_action, grip_action=None, *args, **kwargs):
        self.arm_ctrlr.step(arm_action, should_step=False)
        if self.grip_ctrlr is not None and not self.disable_grip:
            self.grip_ctrlr.step(grip_action, should_step=False)

        return self._sim.step(HabitatSimActions.ARM_ACTION)


@registry.register_task_action
class ArmRelPosAction(SimulatorTaskAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        # The actual joint positions
        self._sim: RearrangeSim
        self._sim.robot.arm_motor_pos = (
            delta_pos + self._sim.robot.arm_motor_pos
        )

        if should_step:
            return self._sim.step(HabitatSimActions.ARM_VEL)
        return None


@registry.register_task_action
class ArmRelPosKinematicAction(SimulatorTaskAction):
    """
    The arm motor targets are offset by the delta joint values specified by the
    action
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        if self._config.get("SHOULD_CLIP", True):
            # clip from -1 to 1
            delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config.DELTA_POS_LIMIT
        self._sim: RearrangeSim

        set_arm_pos = delta_pos + self._sim.robot.arm_joint_pos
        self._sim.robot.arm_joint_pos = set_arm_pos
        self._sim.robot.fix_joint_values = set_arm_pos
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_VEL)
        return None


@registry.register_task_action
class ArmAbsPosAction(SimulatorTaskAction):
    """
    The arm motor targets are directly set to the joint configuration specified
    by the action.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, set_pos, should_step=True, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self._sim.robot.arm_motor_pos = set_pos
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_ABS_POS)
        else:
            return None


@registry.register_task_action
class ArmAbsPosKinematicAction(SimulatorTaskAction):
    """
    The arm is kinematically directly set to the joint configuration specified
    by the action.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.ARM_JOINT_DIMENSIONALITY,),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def step(self, set_pos, should_step=True, *args, **kwargs):
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self._sim: RearrangeSim
        self._sim.robot.arm_joint_pos = set_pos
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_ABS_POS_KINEMATIC)
        else:
            return None


@registry.register_task_action
class BaseVelAction(SimulatorTaskAction):
    """
    The robot base motion is constrained to the NavMesh and controlled with velocity commands integrated with the VelocityControl interface.

    Optionally cull states with active collisions if config parameter `ALLOW_DYN_SLIDE` is True
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True

        self.end_on_stop = self._config.END_ON_STOP

    @property
    def action_space(self):
        lim = 20
        return spaces.Dict(
            {
                "base_vel": spaces.Box(
                    shape=(2,), low=-lim, high=lim, dtype=np.float32
                )
            }
        )

    def _capture_robot_state(self, sim):
        return {
            "forces": sim.robot.sim_obj.joint_forces,
            "vel": sim.robot.sim_obj.joint_velocities,
            "pos": sim.robot.sim_obj.joint_positions,
        }

    def _set_robot_state(self, sim: RearrangeSim, set_dat):
        sim.robot.sim_obj.joint_positions = set_dat["forces"]
        sim.robot.sim_obj.joint_velocities = set_dat["vel"]
        sim.robot.sim_obj.joint_forces = set_dat["pos"]

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    def update_base(self):
        ctrl_freq = self._sim.ctrl_freq

        before_trans_state = self._capture_robot_state(self._sim)

        trans = self._sim.robot.sim_obj.transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )

        target_rigid_state = self.base_vel_ctrl.integrate_transform(
            1 / ctrl_freq, rigid_state
        )
        end_pos = self._sim.step_filter(
            rigid_state.translation, target_rigid_state.translation
        )

        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(), end_pos
        )
        self._sim.robot.sim_obj.transformation = target_trans

        if not self._config.get("ALLOW_DYN_SLIDE", True):
            # Check if in the new robot state the arm collides with anything.
            # If so we have to revert back to the previous transform
            self._sim.internal_step(-1)
            colls = self._sim.get_collisions()
            did_coll, _ = rearrange_collision(
                colls, self._sim.snapped_obj_id, False
            )
            if did_coll:
                # Don't allow the step, revert back.
                self._set_robot_state(self._sim, before_trans_state)
                self._sim.robot.sim_obj.transformation = trans

    def step(self, base_vel, should_step=True, *args, **kwargs):
        lin_vel, ang_vel = base_vel
        lin_vel = np.clip(lin_vel, -1, 1)
        lin_vel *= self._config.LIN_SPEED
        ang_vel = np.clip(ang_vel, -1, 1) * self._config.ANG_SPEED

        if (
            self.end_on_stop
            and abs(lin_vel) < self._config.MIN_ABS_LIN_SPEED
            and abs(ang_vel) < self._config.MIN_ABS_ANG_SPEED
        ):
            self.does_want_terminate = True

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base()

        if should_step:
            return self._sim.step(HabitatSimActions.BASE_VELOCITY)
        else:
            return None


@registry.register_task_action
class ArmEEAction(SimulatorTaskAction):
    """Uses inverse kinematics (requires pybullet) to apply end-effector position control for the robot's arm."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        self.ee_target = None
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def get_ee_pos(self, ):
        cur_ee = self._sim.ik_helper.calc_fk(
            np.array(self._sim.robot.arm_joint_pos)
        )[:3]
        return cur_ee

    def reset(self, *args, **kwargs):
        super().reset()
        cur_ee = self.get_ee_pos()
        self.ee_target = cur_ee

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-1, high=1, dtype=np.float32)

    def apply_ee_constraints(self):
        self.ee_target = np.clip(
            self.ee_target,
            self._sim.robot.params.ee_constraint[:, 0],
            self._sim.robot.params.ee_constraint[:, 1],
        )

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> np.ndarray:
        self.ee_target = np.array(ee_pos) + self.get_ee_pos()
        self.apply_ee_constraints()

        ik = self._sim.ik_helper

        joint_pos = np.array(self._sim.robot.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        ik.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = ik.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        self._sim.robot.arm_motor_pos = des_joint_pos

        return des_joint_pos

    def step(self, ee_pos, should_step=True, **kwargs):
        ee_pos = np.clip(ee_pos, -1, 1)
        ee_pos *= self._config.EE_CTRL_LIM
        self.set_desired_ee_pos(ee_pos)

        if self._config.get("RENDER_EE_TARGET", False):
            global_pos = self._sim.robot.base_transformation.transform_point(
                self.ee_target
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )

        if should_step:
            return self._sim.step(HabitatSimActions.ARM_EE)
        else:
            return None

@registry.register_task_action
class ArmFullEEAction(ArmEEAction):
    """Uses inverse kinematics (requires pybullet) to apply end-effector full (position+orientation) control for the robot's arm."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        self.ee_target = None
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def reset(self, *args, **kwargs):
        super().reset()
        cur_ee = self._sim.ik_helper.calc_fk(
            np.array(self._sim.robot.arm_joint_pos)
        )
        self.ee_target = cur_ee

    @property
    def action_space(self):
        # return spaces.Box(shape=(6,), low=-1, high=1, dtype=np.float32)
        return spaces.Box(shape=(7,), low=-1, high=1, dtype=np.float32)

    def apply_ee_constraints(self):
        self.ee_target[:3] = np.clip(
            self.ee_target[:3],
            self._sim.robot.params.ee_constraint[:, 0],
            self._sim.robot.params.ee_constraint[:, 1],
        )

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> np.ndarray:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        ik = self._sim.ik_helper

        joint_pos = np.array(self._sim.robot.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        ik.set_arm_state(joint_pos, joint_vel)
        des_joint_pos = ik.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        self._sim.robot.arm_motor_pos = des_joint_pos

        return des_joint_pos

    def step(self, ee_pos, should_step=True, **kwargs):
        ee_pos = np.clip(ee_pos, -1, 1)
        ee_pos[:3] *= self._config.EE_CTRL_LIM
        ee_pos[3:] *= self._config.EE_CTRL_QUAT_LIM
        # quat = self.rpy_to_quat(ee_pos[3:])
        quat = ee_pos[3:]
        ee_pos = np.concatenate((ee_pos[:3], quat))
        self.set_desired_ee_pos(ee_pos)

        if self._config.get("RENDER_EE_TARGET", False):
            global_pos = self._sim.robot.base_transformation.transform_point(
                self.ee_target
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )
        if should_step:
            return self._sim.step(HabitatSimActions.ARM_EE)
        else:
            return None

    def rpy_to_quat(self, rpy):
        q = quaternion.from_euler_angles(rpy)
        return np.array([q.x, q.y, q.z, q.w])

@registry.register_task_action
class ArmRAPSAction(SimulatorTaskAction):
    """Incorporate the RAPS primitives onto the robot."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        self.ee_target = None
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

        if self._config.GRIP_CONTROLLER is not None:
            grip_controller_cls = eval(self._config.GRIP_CONTROLLER)
            self.grip_ctrlr: Optional[
                GripSimulatorTaskAction
            ] = grip_controller_cls(*args, config=config, sim=sim, **kwargs)
        else:
            self.grip_ctrlr = None

        self.disable_grip = False
        if "DISABLE_GRIP" in config:
            self.disable_grip = config["DISABLE_GRIP"]
        self.action_scale = self._config.ACTION_SCALE
        # primitives
        self.primitive_idx_to_name = {
            0: "move_delta_ee_pose",
            1: "top_x_y_grasp",
            2: "lift",
            3: "drop",
            4: "move_left",
            5: "move_right",
            6: "move_forward",
            7: "move_backward",
            8: "open_gripper",
            9: "close_gripper",
        }
        self.primitive_name_to_func = dict(
            move_delta_ee_pose=self.move_delta_ee_pose,
            top_x_y_grasp=self.top_x_y_grasp,
            lift=self.lift,
            drop=self.drop,
            move_left=self.move_left,
            move_right=self.move_right,
            move_forward=self.move_forward,
            move_backward=self.move_backward,
            open_gripper=self.open_gripper,
            close_gripper=self.close_gripper,
        )
        self.primitive_name_to_action_idx = dict(
            move_delta_ee_pose=[0, 1, 2],
            top_x_y_grasp=[3, 4, 5],
            lift=6,
            drop=7,
            move_left=8,
            move_right=9,
            move_forward=10,
            move_backward=11,
            open_gripper=[],
            close_gripper=[],
        )
        self.max_arg_len = 12
        self.num_primitives = len(self.primitive_name_to_func)

    def reset(self, *args, **kwargs):
        super().reset()
        if self.grip_ctrlr is not None:
            self.grip_ctrlr.reset(*args, **kwargs)

    @property
    def action_space(self):
        action_space_low = -1 * np.ones(self.max_arg_len)
        action_space_high = np.ones(self.max_arg_len)
        act_lower_primitive = np.zeros(self.num_primitives)
        act_upper_primitive = np.ones(self.num_primitives)
        act_lower = np.concatenate((act_lower_primitive, action_space_low))
        act_upper = np.concatenate(
            (
                act_upper_primitive,
                action_space_high,
            )
        )
        action_space = spaces.Box(act_lower, act_upper, dtype=np.float32)
        action_spaces = {'action': action_space}
        return spaces.Dict(action_spaces)

    def apply_ee_constraints(self, ee_target):
        ee_target = np.clip(
            ee_target,
            self._sim.robot.params.ee_constraint[:, 0],
            self._sim.robot.params.ee_constraint[:, 1],
        )
        return ee_target

    def set_desired_ee_pos(self, delta_ee: np.ndarray) -> np.ndarray:
        ee_target = delta_ee + self.get_endeff_pos()
        ee_target = self.apply_ee_constraints(ee_target)
        self.ee_target = ee_target

        ik = self._sim.ik_helper

        joint_pos = np.array(self._sim.robot.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        ik.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = ik.calc_ik(ee_target)
        des_joint_pos = list(des_joint_pos)
        self._sim.robot.arm_motor_pos = des_joint_pos

        return des_joint_pos

    def step(self, action, should_step=True, **kwargs):
        action = np.clip(action, -1.0, 1.0)
        self.img_array = []
        stats, o = self.act(action)
        return o

    def get_endeff_pos(self,):
        cur_ee = self._sim.ik_helper.calc_fk(
            np.array(self._sim.robot.arm_joint_pos)
        )[:3]
        return cur_ee

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx

    def set_gripper_action(self, gripper_ctrl):
        if self.grip_ctrlr is not None and not self.disable_grip:
            self.grip_ctrlr.step(gripper_ctrl, should_step=False)

    def _set_action(self, action):

        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= self._config.EE_CTRL_LIM

        # Apply action to simulation.
        self.set_desired_ee_pos(pos_ctrl)
        if gripper_ctrl is not None:
            self.set_gripper_action(gripper_ctrl)

    def close_gripper(self, unused):
        total_reward, total_success = 0, 0
        for _ in range(1):
            a = np.array([0.0, 0.0, 0.0, 1]) #1 means close the gripper
            self._set_action(a)
            o = self._sim.step(HabitatSimActions.ARM_ACTION)
        return np.array((total_reward, total_success)), o

    def open_gripper(self, unused):
        total_reward, total_success = 0, 0
        for _ in range(1):
            a = np.array([0.0, 0.0, 0.0, -1]) #-1 means open the gripper
            self._set_action(a)
            o = self._sim.step(HabitatSimActions.ARM_ACTION)
        return np.array((total_reward, total_success)), o

    def goto_pose(self, pose, grasp=False):
        total_reward, total_success = 0, 0
        for i in range(self._config.GOTO_POSE_ITERATIONS):
            delta = pose - self.get_endeff_pos()
            a = np.array([delta[0], delta[1], delta[2], None])
            self._set_action(a)
            o = self._sim.step(HabitatSimActions.ARM_ACTION)
            # self.img_array.append(o['robot_third_rgb'])
        return np.array((total_reward, total_success)), o

    def top_x_y_grasp(self, xyz):
        x_dist, y_dist, z_dist = xyz
        stats, _ = self.move_delta_ee_pose(np.array([x_dist, y_dist, 0]))
        stats += self.drop(z_dist)[0]
        stats_, o = self.close_gripper(1)
        stats += stats_
        return stats, o

    def move_delta_ee_pose(self, pose):
        stats, o = self.goto_pose(self.get_endeff_pos() + pose)
        return stats, o

    def lift(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats, o = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, z_dist]), grasp=True
        )
        return stats, o

    def drop(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats, o = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, -z_dist]), grasp=True
        )
        return stats, o

    def move_left(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats, o = self.goto_pose(
            self.get_endeff_pos() + np.array([-x_dist, 0.0, 0.0]), grasp=True
        )
        return stats, o

    def move_right(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats, o = self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0.0]), grasp=True
        )
        return stats, o

    def move_forward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats, o = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0.0]), grasp=True
        )
        return stats, o

    def move_backward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats, o = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, -y_dist, 0.0]), grasp=True
        )
        return stats, o

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def act(self, a):
        primitive_idx, primitive_args = (
            np.argmax(a[: self.num_primitives]),
            a[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        stats, o = primitive(
            primitive_action,
        )
        return stats, o


