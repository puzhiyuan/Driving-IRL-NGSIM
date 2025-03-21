from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
import numpy as np

from NGSIM_env import utils
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env import utils
from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, NGSIMVehicle
from NGSIM_env.road.lane import LineType, StraightLane
from NGSIM_env.utils import *
from NGSIM_env.data.data_process import build_trajecotry

class NGSIMEnv(AbstractEnv):
    """
    A highway driving environment with NGSIM data.
    """
    def __init__(self, scene, period, vehicle_id, IDM=False):
        self.vehicle_id = vehicle_id
        self.scene = scene
        self.trajectory_set = build_trajecotry(scene, period, vehicle_id)
        self.ego_length = self.trajectory_set['ego']['length'] / 3.281
        self.ego_width = self.trajectory_set['ego']['width'] / 3.281
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']
        self.duration = len(self.ego_trajectory) - 3
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        self.surrounding_vehicles.pop(0)
        self.run_step = 0
        self.human = False
        self.IDM = IDM
        super(NGSIMEnv, self).__init__()

    def process_raw_trajectory(self, trajectory):
        trajectory = np.array(trajectory)
        for i in range(trajectory.shape[0]):
            x = trajectory[i][0] - 6
            y = trajectory[i][1]
            speed = trajectory[i][2]
            trajectory[i][0] = y / 3.281
            trajectory[i][1] = x / 3.281
            trajectory[i][2] = speed / 3.281

        return trajectory

    def default_config(self):
        config = super().default_config()
        config.update({
            "observation": {"type": "Kinematics"},
            "vehicles_count": 10,
            "show_trajectories": True,
            "screen_width": 800,
            "screen_height": 300,
        })

        return config

    def reset(self, human=False, reset_time=1):
        '''
        Reset the environment at a given time (scene) and specify whether use human target 
        '''

        self.human = human
        self._create_road()
        self._create_vehicles(reset_time)
        self.steps = 0

        return super(NGSIMEnv, self).reset()

    def _create_road(self):
        """
        Create a road composed of NGSIM road network
        """
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        if self.scene == 'us-101':
            length = 2150 / 3.281 # m
            width = 12 / 3.281 # m
            ends = [0, 560/3.281, (698+578+150)/3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('merge_in', 's2', StraightLane([480/3.281, 5.5*width], [ends[1], 5*width], width=width, line_types=[c, c], forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(6):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))
            
            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_out lanes
            net.add_lane('s3', 'merge_out', StraightLane([ends[2], 5*width], [1550/3.281, 7*width], width=width, line_types=[c, c], forbidden=True))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        
        elif self.scene == 'i-80':
            length = 1700 / 3.281
            lanes = 6 
            width = 12 / 3.281
            ends = [0, 600/3.281, 700/3.281, 900/3.281, length]
            
            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s1', 's2', StraightLane([380/3.281, 7.1*width], [ends[1], 6*width], width=width, line_types=[c, c], forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s2', 's3', StraightLane([ends[1], 6*width], [ends[2], 6*width], width=width, line_types=[s, c]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lane
            net.add_lane('s3', 's4', StraightLane([ends[2], 6*width], [ends[3], 5*width], width=width, line_types=[n, c]))

            # forth section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[3], lane * width]
                end = [ends[4], lane * width]
                net.add_lane('s4', 's5', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self, reset_time):
        """
        Create ego vehicle and NGSIM vehicles and add them on the road.
        """
        whole_trajectory = self.process_raw_trajectory(self.ego_trajectory)
        ego_trajectory = whole_trajectory[reset_time:]
        ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time-1][2]) / 0.1
        self.vehicle = HumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length, self.ego_width,
                                               ego_trajectory, acc=ego_acc, velocity=ego_trajectory[0][2], human=self.human, IDM=self.IDM)
        self.road.vehicles.append(self.vehicle)

        for veh_id in self.surrounding_vehicles:
            other_trajectory = self.process_raw_trajectory(self.trajectory_set[veh_id]['trajectory'])[reset_time:]
            self.road.vehicles.append(NGSIMVehicle.create(self.road, veh_id, other_trajectory[0][:2], self.trajectory_set[veh_id]['length']/3.281,
                                                          self.trajectory_set[veh_id]['width']/3.281, other_trajectory, velocity=other_trajectory[0][2]))

    def step(self, action=None):
        """
        Perform a MDP step
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        
        if action is not None:
            lateral, target_speed = self.sampling_space()
            action = (lateral[action[0]], target_speed[action[1]], 5)

        features = self._simulate(action)
        reward = self._reward(features)
        obs = self.observation.observe()
        terminated = self._is_terminal()
        truncated = self._is_truncated()
        info = {
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            'offroad': not self.vehicle.on_road,
            "action": action,
            "time": self.time
        }

        # observations,
        # rewards,
        # terminateds,
        # truncateds,
        # infos,

        return obs, reward, terminated, truncated, info
    
    # def _reward(self, features):
    #     speed = features[0]         # 速度
    #     accel_long = features[1]    # 纵向加速度
    #     accel_lat = features[2]     # 横向加速度
    #     jerk_long = features[3]     # 纵向冲击
    #     thw_front = features[4]     # 前车时距
    #     thw_back = features[5]      # 后车时距
    #     collision = features[6]     # 碰撞 (0: 无碰撞, 1: 发生碰撞)
    #     social_impact = features[7] # 超车影响别的车减速
    #     human_likeness = features[8] # 人类驾驶相似度

    #     # 速度奖励 (希望接近 v_target)
    #     v_target = 10.0  # 目标速度 (m/s)
    #     speed_reward = -abs(speed - v_target) * 0.5  # 偏离目标速度的惩罚

    #     # 纵向加速度惩罚 (减少急加速或急刹车)
    #     accel_long_penalty = -abs(accel_long) * 0.2

    #     # 横向加速度惩罚 (避免剧烈变道)
    #     accel_lat_penalty = -abs(accel_lat) * 0.3

    #     # 纵向冲击惩罚 (jerk 过大会影响舒适度)
    #     jerk_long_penalty = -abs(jerk_long) * 0.1

    #     # 车距奖励 (合理的前车时距)
    #     if thw_front < 1.0:  # 过近，给予惩罚
    #         thw_front_reward = -5.0
    #     elif 1.5 <= thw_front <= 3.0:  # 理想范围，给予奖励
    #         thw_front_reward = 2.0
    #     else:
    #         thw_front_reward = 0.0  # 正常情况，无奖励无惩罚

    #     # 后车时距惩罚 (避免后车太近)
    #     thw_back_penalty = -0.5 if thw_back < 1.0 else 0.0

    #     # 碰撞惩罚 (若发生碰撞，直接给予大额负奖励)
    #     collision_penalty = -100.0 if collision > 0 else 0.0

    #     # 社会影响惩罚 (影响其他车辆越大，惩罚越大)
    #     social_impact_penalty = -social_impact * 2.0

    #     # 人类驾驶相似度奖励 (行为越接近人类驾驶，奖励越高)
    #     human_likeness_reward = human_likeness * 5.0  # 最高 +5 奖励

    #     # 计算最终奖励
    #     total_reward = (
    #         speed_reward + 
    #         accel_long_penalty + 
    #         accel_lat_penalty + 
    #         jerk_long_penalty + 
    #         thw_front_reward + 
    #         thw_back_penalty + 
    #         collision_penalty + 
    #         social_impact_penalty + 
    #         human_likeness_reward
    #     )

    #     return total_reward

    def _reward(self, features):
        speed = features[0]         # 速度
        accel_long = features[1]    # 纵向加速度
        accel_lat = features[2]     # 横向加速度
        jerk_long = features[3]     # 纵向冲击
        thw_front = features[4]     # 前车时距
        thw_back = features[5]      # 后车时距
        collision = features[6]     # 碰撞
        social_impact = features[7] # 社会影响
        human_likeness = features[8] # 人类驾驶相似度

        v_target = 10.0  # 目标速度 (m/s)

        # 速度奖励：二次函数，使其在 v_target 处最优
        speed_reward = - (speed - v_target) ** 2 * 0.1

        # 运动奖励：防止小车停滞
        movement_bonus = 1.0 if speed > 0.1 else -10.0

        # 调整加速度和冲击惩罚，避免过度惩罚
        accel_long_penalty = -abs(accel_long) * 0.1
        accel_lat_penalty = -abs(accel_lat) * 0.3
        jerk_long_penalty = -abs(jerk_long) * 0.05

        # 车距奖励
        if thw_front < 1.0:
            thw_front_reward = -5.0
        elif 1.5 <= thw_front <= 3.0:
            thw_front_reward = 2.0
        else:
            thw_front_reward = 0.0

        thw_back_penalty = -0.5 if thw_back < 1.0 else 0.0

        # 碰撞惩罚
        collision_penalty = -100.0 if collision > 0 else 0.0

        # 社会影响惩罚
        social_impact_penalty = -social_impact * 2.0

        # 人类驾驶奖励
        human_likeness_reward = human_likeness * 5.0

        # 总奖励计算
        total_reward = (
            speed_reward +
            movement_bonus +
            accel_long_penalty +
            accel_lat_penalty +
            jerk_long_penalty +
            thw_front_reward +
            thw_back_penalty +
            collision_penalty +
            social_impact_penalty +
            human_likeness_reward
        )

        return total_reward


    def _simulate(self, action):
        """
        Perform several steps of simulation with the planned trajectory
        """
        trajectory_features = []
        T = action[2] if action is not None else 5

        for i in range(int(T * self.SIMULATION_FREQUENCY)-1):
            if i == 0:
                if action is not None: # sampled goal
                    self.vehicle.trajectory_planner(action[0], action[1], action[2])
                else: # human goal
                    self.vehicle.trajectory_planner(self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10][1], 
                                                   (self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10][0]-self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10-1][0])/0.1, T)
                self.run_step = 1

            self.road.act(self.run_step)
            self.road.step(1/self.SIMULATION_FREQUENCY)
            self.time += 1
            self.run_step += 1
            features = self._features()
            trajectory_features.append(features)

            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        self.enable_auto_render = False

        human_likeness = features[-1]
        interaction = np.max([feature[-2] for feature in trajectory_features])
        trajectory_features = np.sum(trajectory_features, axis=0)
        trajectory_features[-1] = human_likeness
        
        return trajectory_features

    def _features(self):
        """
        Hand-crafted features
        :return: the array of the defined features
        """
        # ego motion
        ego_longitudial_positions = self.vehicle.traj.reshape(-1, 2)[self.time-3:, 0]
        ego_longitudial_speeds = (ego_longitudial_positions[1:] - ego_longitudial_positions[:-1]) / 0.1 if self.time >= 3 else [0]
        ego_longitudial_accs = (ego_longitudial_speeds[1:] - ego_longitudial_speeds[:-1]) / 0.1 if self.time >= 3 else [0]
        ego_longitudial_jerks = (ego_longitudial_accs[1:] - ego_longitudial_accs[:-1]) / 0.1 if self.time >= 3 else [0]

        ego_lateral_positions = self.vehicle.traj.reshape(-1, 2)[self.time-3:, 1]
        ego_lateral_speeds = (ego_lateral_positions[1:] - ego_lateral_positions[:-1]) / 0.1 if self.time >= 3 else [0]
        ego_lateral_accs = (ego_lateral_speeds[1:] - ego_lateral_speeds[:-1]) / 0.1 if self.time >= 3 else [0]

        # travel efficiency
        ego_speed = abs(ego_longitudial_speeds[-1])

        # comfort
        ego_longitudial_acc = ego_longitudial_accs[-1]
        ego_lateral_acc = ego_lateral_accs[-1]
        ego_longitudial_jerk = ego_longitudial_jerks[-1]
 
        # time headway front (THWF) and time headway behind (THWB)
        THWFs = [100]; THWBs = [100]
        for v in self.road.vehicles:
            if v.position[0] > self.vehicle.position[0] and abs(v.position[1]-self.vehicle.position[1]) < self.vehicle.WIDTH and self.vehicle.velocity >= 1:
                THWF = (v.position[0] - self.vehicle.position[0]) / self.vehicle.velocity
                THWFs.append(THWF)
            elif v.position[0] < self.vehicle.position[0] and abs(v.position[1]-self.vehicle.position[1]) < self.vehicle.WIDTH and v.velocity >= 1:
                THWB = (self.vehicle.position[0] - v.position[0]) / v.velocity
                THWBs.append(THWB)

        THWF = np.exp(-min(THWFs))
        THWB = np.exp(-min(THWBs)) 

        # avoid collision
        collision = 1 if self.vehicle.crashed or not self.vehicle.on_road else 0

        # interaction (social) impact
        social_impact = 0
        for v in self.road.vehicles:
            if isinstance(v, NGSIMVehicle) and v.overtaken and v.velocity != 0:
                social_impact += np.abs(v.velocity - v.velocity_history[-1])/0.1 if v.velocity - v.velocity_history[-1] < 0 else 0

        # ego vehicle human-likeness
        ego_likeness = self.vehicle.calculate_human_likeness()

        # feature array
        fetures = np.array([ego_speed, abs(ego_longitudial_acc), abs(ego_lateral_acc), abs(ego_longitudial_jerk),
                            THWF, THWB, collision, social_impact, ego_likeness])

        return fetures

    def _is_terminal(self):
        """
        The episode is over if the ego vehicle crashed or go off road or the time is out.
        """
        return self.vehicle.crashed or self.time >= self.duration or self.vehicle.position[0] >= 2150/3.281 or not self.vehicle.on_road
    
    def _is_truncated(self):
        """
        The episode is truncated if the ego vehicle crashed or go off road.
        """
        return self.vehicle.velocity == 0 or self.vehicle.crashed or not self.vehicle.on_road

    def sampling_space(self):
        """
        The target sampling space (longitudinal speed and lateral offset)
        """
        lane_center = self.vehicle.lane.start[1]
        current_y = self.vehicle.position[1]
        current_speed = self.vehicle.velocity
        lateral_offsets = np.array([lane_center-12/3.281, current_y, lane_center+12/3.281])
        min_speed = current_speed - 5 if current_speed > 5 else 0
        max_speed = current_speed + 5
        target_speeds = np.linspace(min_speed, max_speed, 10)

        return lateral_offsets, target_speeds
    
