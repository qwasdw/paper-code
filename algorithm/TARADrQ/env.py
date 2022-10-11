import time
import numpy as np
import airsim
import gym
import math
import random
from math import pi
import cv2
from PIL import Image

timeslice = 0.1
velocity = 1.0




class DroneEnv:
    _max_episode_steps = 500

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4, 144, 256), dtype=np.float32)
        self.goal_set = [[[0.0, 0.0], [6.0, 6.0]], [[0.0, 0.0], [6.0, -6.0]],
                         [[0.0, 0.0], [-6.0, 6.0]], [[0.0, 0.0], [-6.0, -6.0]],
                         [[0.0, 0.0], [0.0, 6.0]], [[0.0, 0.0], [0.0, -6.0]],
                         [[0.0, 0.0], [-6.0, 0.0]], [[0.0, 0.0], [6.0, 0.0]],
                         [[6.0, 6.0], [6.0, -6.0]], [[6.0, 6.0], [-6.0, 6.0]],
                         [[6.0, 6.0], [-6.0, -6.0]], [[6.0, 6.0], [0.0, 6.0]],
                         [[6.0, 6.0], [0.0, -6.0]], [[6.0, 6.0], [-6.0, 0.0]],
                         [[6.0, 6.0], [6.0, 0.0]], [[6.0, -6.0], [-6.0, 6.0]],
                         [[6.0, -6.0], [-6.0, -6.0]], [[6.0, -6.0], [0.0, 6.0]],
                         [[6.0, -6.0], [0.0, -6.0]], [[6.0, -6.0], [-6.0, 0.0]],
                         [[6.0, -6.0], [6.0, 0.0]], [[-6.0, 6.0], [-6.0, -6.0]],
                         [[-6.0, 6.0], [0.0, 6.0]], [[-6.0, 6.0], [0.0, -6.0]],
                         [[-6.0, 6.0], [-6.0, 0.0]], [[-6.0, 6.0], [6.0, 0.0]],
                         [[-6.0, -6.0], [0.0, 6.0]], [[-6.0, -6.0], [0.0, -6.0]],
                         [[-6.0, -6.0], [-6.0, 0.0]], [[-6.0, -6.0], [6.0, 0.0]],
                         [[0.0, 6.0], [0.0, -6.0]], [[0.0, 6.0], [-6.0, 0.0]],
                         [[0.0, 6.0], [6.0, 0.0]], [[0.0, -6.0], [-6.0, 0.0]],
                         [[0.0, -6.0], [6.0, 0.0]], [[-6.0, 0.0], [6.0, 0.0]]]
        self.index = 0
        self.goal = [0.0, 0.0]

        self.index_eval = 0
        self.goal_eval = [0.0, 0.0]

        self.histState = self.initializeHistState()
        self.last_distance = 0.0
        self.dis = 10000.0



    def initializeHistState(self):
        histState = np.concatenate((self.get_state(), self.get_state()), axis=1)
        histState = np.concatenate((histState, self.get_state()), axis=1)
        histState = np.concatenate((histState, self.get_state()), axis=1)
        return histState

    def reset(self):
        self.goal = self.goal_set[self.index][0]
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.goal[0]
        pose.position.y_val = self.goal[1]
        self.client.simSetVehiclePose(pose, True)

        self.client.moveByVelocityZAsync(0.0, 0.0, -0.5, 0.1).join()
        time.sleep(0.1)
        self.histState = self.initializeHistState()

        new_goal = self.goal_set[self.index][1]
        d = np.sqrt(np.square(self.goal[0] - new_goal[0]) + np.square(self.goal[1] - new_goal[1]))
        self.last_distance = d
        self.dis = d
        self.goal = new_goal
        others = self.get_others()
        self.index = (self.index + 1) % 36
        return self.histState, others

    def reset_eval(self):
        self.goal_eval = self.goal_set[self.index_eval][0]
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.goal_eval[0]
        pose.position.y_val = self.goal_eval[1]
        self.client.simSetVehiclePose(pose, True)

        self.client.moveByVelocityZAsync(0.0, 0.0, -0.5, 0.1).join()
        time.sleep(0.1)
        self.histState = self.initializeHistState()

        new_goal = self.goal_set[self.index_eval][1]
        d = np.sqrt(np.square(self.goal_eval[0] - new_goal[0]) + np.square(self.goal_eval[1] - new_goal[1]))
        self.last_distance = d
        self.dis = d
        self.goal_eval = new_goal
        self.index_eval = (self.index_eval + 1) % 36
        others = self.get_others()
        return self.histState, others

    def step(self, action):

        has_collided = False
        angle_rate = action[0] * 60
        self.client.moveByVelocityZBodyFrameAsync(velocity, 0.0, -0.5, 0.1,
                                             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                             yaw_mode=airsim.YawMode(True, angle_rate))
        time.sleep(0.02)
        s1 = self.get_state()
        time.sleep(0.02)
        s2 = self.get_state()
        time.sleep(0.02)
        s3 = self.get_state()
        time.sleep(0.02)
        s4 = self.get_state()

        self.client.moveByVelocityZBodyFrameAsync(0,0,-0.5,100)

        next_state = np.concatenate((s1, s2, s3, s4), axis=1)

        collided = self.client.simGetCollisionInfo()
        if collided.object_name != '':
            has_collided = True

        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        pos = [quad_pos.x_val, quad_pos.y_val]
        reward, has_done, arrived = self.get_reward(has_collided, pos)
        others = self.get_others()

        done = has_collided or has_done

        return next_state, others, reward, done, arrived

    def step_eval(self, action):

        has_collided = False
        angle_rate = action[0] * 60
        self.client.moveByVelocityZBodyFrameAsync(velocity, 0.0, -0.5, 0.1,
                                             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                             yaw_mode=airsim.YawMode(True, angle_rate))
        time.sleep(0.02)
        s1 = self.get_state()
        time.sleep(0.02)
        s2 = self.get_state()
        time.sleep(0.02)
        s3 = self.get_state()
        time.sleep(0.02)
        s4 = self.get_state()

        self.client.moveByVelocityZBodyFrameAsync(0,0,-0.5,100)

        next_state = np.concatenate((s1, s2, s3, s4), axis=1)

        collided = self.client.simGetCollisionInfo()
        if collided.object_name != '':
            has_collided = True

        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        pos = [quad_pos.x_val, quad_pos.y_val]
        reward, has_done, arrived = self.get_reward_eval(has_collided, pos)
        others = self.get_others()

        done = has_collided or has_done

        return next_state, others, reward, done, arrived

    def get_reward(self, collided, pos):
        arrived = False
        data = self.client.getDistanceSensorData()
        data = data.distance
        current_distance = np.sqrt(np.square(self.goal[0] - pos[0]) + np.square(self.goal[1] - pos[1]))
        done = False
        if current_distance <= 0.3:
            arrived = True
            done = True
        if collided or data <= 0.3:
            reward = -1.0
            done = True
        elif arrived:
            reward = 1.0
        else:
            reward = self.last_distance - current_distance
        self.last_distance = current_distance
        return reward, done, arrived

    def get_reward_eval(self, collided, pos):
        arrived = False
        data = self.client.getDistanceSensorData()
        data = data.distance
        current_distance = np.sqrt(np.square(self.goal_eval[0] - pos[0]) + np.square(self.goal_eval[1] - pos[1]))
        done = False
        if current_distance <= 0.3:
            arrived = True
            done = True
        if collided or data <= 0.3:
            reward = -1.0
            done = True
        elif arrived:
            reward = 1.0
        else:
            reward = self.last_distance - current_distance
        self.last_distance = current_distance
        return reward, done, arrived

    def get_others(self):
        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        yaw = round(math.degrees(yaw))

        if yaw >= 0:
            yaw = yaw
        else:
            yaw = yaw + 360

        rel_dis_x = round(self.goal[0] - quad_pos.x_val, 1)
        rel_dis_y = round(self.goal[1] - quad_pos.y_val, 1)

        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        distance = self.last_distance / self.dis
        yaw = yaw / 360
        rel_theta = rel_theta / 360
        diff_angle = diff_angle / 180
        others = np.array([distance, yaw, rel_theta, diff_angle])
        others = np.reshape(others, (1, 4))


        return others

    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')

    def get_stacked_state(self):
        self.histState = np.delete(self.histState, 0, axis=1)
        state = self.get_state()
        self.histState = np.concatenate((self.histState, state), axis=1)
        return self.histState

    def get_state(self):
        try:
            responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True)])
            state = self.transform_input(responses)
        except:
            state = np.zeros([1, 1, 144, 256])
        return state

    def transform_input(self, responses, img_height=144, img_width=256):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        image = np.float32(img1d)
        image = np.reshape(image, (img_height, img_width))
        depth = np.clip(image, 0.0, 10.0) / 10.0

        depth = np.reshape(depth, (1, 1, 144, 256))
        return depth


    def get_quad_pos(self):
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        return quad_pos

