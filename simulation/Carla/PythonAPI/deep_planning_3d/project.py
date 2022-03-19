import math
import sys
import os
import time
import random
import json
import logging
import datetime
import carla
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from utils import CarlaSyncMode, SpectatorController, CarlaPygameHelper


# from scipy import signal
# Makes the drone fly and get Lidar data
# from use_input_pipeline import input_fn

from models import build_cnn_3d

# Carla constants
TOWN_NAME = "Town10HD_Opt"
SIM_FPS = 40
HEIGHT, WIDTH = 600, 800
START_LOC = {"x": 0.0, "y": 0.0, "z": 2.0}
START_ROT = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
GOAL = {"x": 40, "y": 30, "z": 20}
SECURITY_DISTANCE = 1
SUCCESS_DISTANCE = 1
MAX_EXP_ITERATIONS = 50
N_EXPS = 100

# Model constants
MODEL_PATH = "./checkpoints/cp-final.ckpt"
EXPORT_DIRECTORY = "./exps/"
GOAL_RADIUS = 0.09
SIGMAG = 1e6
SIGMAE = 100
SIGMAO = 0
INPUT_SHAPE = (101, 101, 101, 1)
ADAM_L_R = 1
ADAM_MAX_ITER = 100
ADAM_THRESHOLD = 0.01


def init_world_and_controller(
    start_location={"x": 0.0, "y": 0.0, "z": 50.0},
    start_rotation={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
):
    actor_list = []
    # Connect to the CARLA server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    if world.get_map().name.split("/")[-1] != "Town10HD_Opt":
        world = client.load_world(
            "Town10HD_Opt"
        )  # it takes a while to load, so the client timeout needs to afford that.

    # https://carla.readthedocs.io/en/latest/bp_library/#sensor

    sensors = []

    # https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera
    camera_rgb_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    sensors.append({"name": "camera_rgb", "blueprint": camera_rgb_bp})

    # https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor
    lidar_raycast_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    lidar_raycast_bp.set_attribute("range", str(500))
    lidar_raycast_bp.set_attribute("upper_fov", str(90.0))
    lidar_raycast_bp.set_attribute("lower_fov", str(-90.0))
    lidar_raycast_bp.set_attribute("horizontal_fov", str(360.0))
    sensors.append({"name": "lidar_raycast", "blueprint": lidar_raycast_bp})

    # camera_semantic_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    # sensors.append({'name':'camera_semantic','blueprint':camera_semantic_bp})

    spec_ctrl = SpectatorController(world, sensors)
    actor_list.append(
        spec_ctrl.sensors["camera_rgb"]
    )  # so it will be destroyed at the end
    actor_list.append(
        spec_ctrl.sensors["lidar_raycast"]
    )  # so it will be destroyed at the end
    # actor_list.append(spec_ctrl.sensors['camera_semantic']) # so it will be destroyed at the end
    spec_ctrl.spectator.set_transform(
        carla.Transform(
            carla.Location(**start_location), carla.Rotation(**start_rotation)
        )
    )
    # print("setted loc: ", start_location)
    time.sleep(1)
    # print("current loc: ", spec_ctrl.spectator.get_transform().location)
    
    return world, spec_ctrl, actor_list


def transform_lidar_data(lidar_data, spec_ctrl):
    """transforms lidar point cloud from the local corrdinates to the world cordinates."""

    if lidar_data:
        # https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py
        # Get the lidar data and convert it to a numpy array.
        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype("f4")))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

        # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
        # focus on the 3D points.
        intensity = np.array(p_cloud[:, 3])

        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_points = np.array(p_cloud[:, :3]).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_points = np.r_[local_points, [np.ones(local_points.shape[1])]]

        # This (4, 4) matrix transforms the points from lidar space to world space.
        lidar_2_world = spec_ctrl.sensors["lidar_raycast"].get_transform().get_matrix()

        # Transform the points from lidar space to world space.
        world_points = np.dot(lidar_2_world, local_points)

        # set shapes to (p_cloud_size, 3)
        world_points = world_points[:3, :].T
        local_points = local_points[:3, :].T
    else:
        local_points, world_points = np.array([]), np.array([])

    logger.info(f"Number of lidar 3D points: {len(world_points)}")

    return local_points, world_points


def advance_simulation(
    pgh, spec_ctrl, sync_mode, next_loc=None, next_rot=None, auto_mode=True
):
    if auto_mode and (next_loc is None and next_rot is None):
        raise ValueError(
            "To use auto mode with the advance_simulation() function, "
            "next_loc and next_rot attributes can not be None."
        )

    # if pgh.should_quit():
    #     return
    # pgh.clock.tick(SIM_FPS)

    # for automatic controll, key parsing is desactivated
    if auto_mode:
        spec_ctrl.spectator.set_transform(carla.Transform(next_loc, next_rot))
        time.sleep(0.5)
    else:
        # spec_ctrl.parse_keys(pgh.clock.get_time())
        raise NotImplementedError

    # Advance the simulation and wait for the data.
    # snapshot, image_rgb, image_semantic = sync_mode.tick(timeout=2.0)
    snapshot, image_rgb, lidar_data = sync_mode.tick(timeout=2.0)

    # image_semseg.convert(carla.ColorConverter.CityScapesPalette)
    # fps = round(1.0 / snapshot.timestamp.delta_seconds)

    # Draw the display.
    # if image_rgb:
    #     pgh.draw_image(image_rgb)

    # Overlay the semantic segmentation
    # image_semantic.convert(carla.ColorConverter.CityScapesPalette)
    # pgh.draw_image(image_semantic, blend=True)

    # msg = "UpKey:Forward, DownKey:Backward, LeftKey:+Yaw, RightKey:-Yaw, W:+Pitch, S:-Pitch, SPACE: Stop, ESC: Exit"
    # pgh.blit(pgh.font.render(msg, True, (255, 255, 255)), (8, 10))
    # pgh.blit(
    #     pgh.font.render(f"{pgh.clock.get_fps()} FPS (real)", True, (255, 255, 255)),
    #     (8, 30),
    # )
    # pgh.blit(pgh.font.render(f"{fps} FPS (simulated)", True, (255, 255, 255)), (8, 50))
    # pgh.blit(
    #     pgh.font.render(
    #         f"{spec_ctrl.spectator.get_transform().location}", True, (255, 255, 255)
    #     ),
    #     (8, 70),
    # )
    # pgh.blit(
    #     pgh.font.render(
    #         f"{spec_ctrl.spectator.get_transform().rotation}", True, (255, 255, 255)
    #     ),
    #     (8, 90),
    # )
    # pgh.flip()

    return lidar_data


def adjust_path_to_goal(path, goal):
    stop_index = len(path)
    goal_reached = False
    for i, point in enumerate(path):
        if np.sqrt(np.sum(np.square(point - (goal["x"],goal["y"],goal["z"])))) < SUCCESS_DISTANCE:
            stop_index = i
            goal_reached = True
            logger.info("path shortened to stop at the goal.")
            break
    return path[:stop_index+1,:], goal_reached
            

def detect_obstacles(local_lidar_points, horizontal_angle):
    distance_from_lidar = pairwise_distances(local_lidar_points, np.array([0,0,0]).reshape(-1, 3), metric="euclidean")
    logger.info(
        f"minimal distance to obstables (from Lidar {horizontal_angle}): {distance_from_lidar.min()}"
    )
    

def detect_collision(
    obstacles,
    path,
):
    assert path.shape[1] == 3, "The path does not have 3 coordinates."
    assert obstacles.shape[1] == 3, "The obstacles do not have 3 coordinates."

    distance_matrix = pairwise_distances(obstacles, path, metric="euclidean")
    logger.info(f"min distance between path and obstacles: {distance_matrix.min()}")
    
    collision = distance_matrix.min() < SECURITY_DISTANCE
    
    return collision


def draw_path(debug, path, thickness=0.1, color=carla.Color(255, 0, 0), lt=0):
    if len(path) < 2:
        raise ValueError("The path must have 2 points minimally.")
    for i in range(len(path) - 1):
        w0 = carla.Location(x=path[i][0], y=path[i][1], z=path[i][2])
        w1 = carla.Location(x=path[i + 1][0], y=path[i + 1][1], z=path[i + 1][2])
        debug.draw_line(
            w0.transform.location + carla.Location(z=0.25),
            w1.transform.location + carla.Location(z=0.25),
            thickness=thickness,
            color=color,
            life_time=lt,
        )
        debug.draw_point(
            w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False
        )


def draw_goal(debug, goal, color=carla.Color(255, 0, 0), lt=0):
    location = carla.Location(**goal)
    debug.draw_point(location, size=0.5, color=color, life_time=lt)


class LidarTest:
    def __init__(
        self,
        start_point,
        goal,
        export_directory,
        vehicle_name,
        lidar_name,
        radius,
        angle,
    ):

        # init variables
        self.drone_name = vehicle_name
        self.lidar_name = lidar_name
        self.x_goal = goal["x"]
        self.y_goal = goal["y"]
        self.z_goal = goal["z"]
        self.position_x = start_point["x"]
        self.position_y = start_point["y"]
        self.position_z = start_point["z"]
        self.offset_x = start_point["x"]
        self.offset_y = start_point["y"]
        self.offset_z = start_point["z"]
        self.export_directory = export_directory
        self.command = ""
        self.exit = 0

        self.radius = radius
        self.angle = angle
        # plt.switch_backend("Agg")
    
    def set_position(self, carla_location):
        self.position_x = carla_location.x
        self.position_y = carla_location.y
        self.position_z = carla_location.z

    def Compute_distance_fn(self, x_goal, y_goal, z_goal, x, y, z):

        return np.sqrt(
            np.power(np.subtract(x, x_goal), 2)
            + np.power(np.subtract(y, y_goal), 2)
            + np.power(np.subtract(z, z_goal), 2)
        )

    def in_range(self, x1, y1, z1):
        if x1 <= 50 and y1 <= 50 and z1 <= 50 and x1 >= -50 and y1 >= -50 and z1 >= -50:
            return True
        else:
            return False

    def normalize_minmax_fn(self, x):
        """normalize values of x between 0 and 1000"""
        min = np.amin(x)
        max = np.amax(x)
        return np.multiply(1000, np.divide(np.subtract(x, min), np.subtract(max, min)))

    def Compute_conductivity_fn(self, x_goal, y_goal, z_goal, x, y, z, d):
        # data = []
        # print("d", d.shape)
        a = np.add(
            np.power(np.subtract(x, x_goal), 2),
            np.power(np.subtract(y, y_goal), 2),
            np.power(np.subtract(z, z_goal), 2),
        )
        # for i in range(0, a.shape[0]):
        #     if a[i] <= pow(GOAL_RADIUS, 2):
        #         data.append(SIGMAG)
        #     else:
        #         data.append(SIGMAE / d[i])

        data = np.where(a <= pow(GOAL_RADIUS, 2), SIGMAG, SIGMAE / d)

        return data
    
    def add_goal_constraint(self, prediction, goal_body):
        """make the magnetic field around the goal to be the maximum"""
        
        max_prediction = np.amax(prediction)
        
        x_min = math.ceil((goal_body[0] - GOAL_RADIUS)*10)
        x_max = math.floor((goal_body[0] + GOAL_RADIUS)*10)
        y_min = math.ceil((goal_body[1] - GOAL_RADIUS)*10)
        y_max = math.floor((goal_body[1] + GOAL_RADIUS)*10)
        z_min = math.ceil((goal_body[2] - GOAL_RADIUS)*10)
        z_max = math.floor((goal_body[2] + GOAL_RADIUS)*10)
        
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                for z in range(z_min, z_max+1):
                    if x <= 101 and y <= 101 and z <= 101 and x >= 0 and y >= 0 and z >= 0:
                        prediction[x, y, z] = max_prediction

        return prediction
        
    def predict(self, model, local_lidar_points, iteration, goal_constraint=False):

        scene, goal_body = self.get_scene(local_lidar_points)
        logger.info(f"local goal is: {goal_body}")

        prediction = model(scene, training=False)
        prediction = np.reshape(prediction, (101, 101, 101))
        
        if goal_constraint:
            prediction = self.add_goal_constraint(prediction.copy(), goal_body)
        
        prediction = self.normalize_minmax_fn(prediction)

        grad_x, grad_y, grad_z = np.gradient(prediction, edge_order=1)

        x_list, y_list, z_list = self.adam(
            self.position_x,
            self.position_y,
            self.position_z,
            ADAM_L_R,
            prediction,
            grad_x,
            grad_y,
            grad_z,
            l_r_decay=True,
        )

        curr_pos_x = int(round(self.position_x))
        curr_pos_y = int(round(self.position_y))
        curr_pos_z = int(round(self.position_z))

        adjuster_x = -(curr_pos_x - 50)
        adjuster_y = -(curr_pos_y - 50)
        adjuster_z = -(curr_pos_z - 50)

        scene = np.reshape(scene, (101, 101, 101))
        # self.plot_conductivity(scene, [i * 10 for i in goal_body])

        self.plot_prediction_path(
            prediction,
            [i * 10 for i in goal_body],
            [i + adjuster_x for i in x_list],
            [i + adjuster_y for i in y_list],
            [i + adjuster_z for i in z_list],
        )

        return x_list, y_list, z_list

    def adam(
        self,
        x_new,
        y_new,
        z_new,
        l_r,
        prediction,
        gradient_x,
        gradient_y,
        gradient_z,
        l_r_decay=False,
    ):
        """
        https://arxiv.org/pdf/1412.6980.pdf

        Description: This function takes in an initial or previous value for (x,y,z), updates it based on
        steps taken via the learning rate and outputs the most optimum position (x,y,z) that reaches the threshold satisfaction.

        Arguments:

        x_new, y_new, z_new - a starting position that will get updated based on the learning rate

        threshold - a precision that determines the stop of the stepwise descent

        l_r - the learning rate (size of each descent step)

        Output:

        1. x_list : x coordinates of the way points which equates to the number of gradient descent steps
        2. y_list : y coordinates of the way points which equates to the number of gradient descent steps
        3. z_list : z coordinates of the way points which equates to the number of gradient descent steps


        """

        # create empty lists where the updated values of x and y will be appended during each iteration to build the path
        beta1 = 0.9
        beta2 = 0.999
        decay = 0.0
        eps_stable = 1e-8
        sqr_x = 0.0
        sqr_y = 0.0
        sqr_z = 0.0
        v_x = 0.0
        v_y = 0.0
        v_z = 0.0
        x_prev = -1
        y_prev = -1
        z_prev = -1
        threshold = ADAM_THRESHOLD
        max_iter = ADAM_MAX_ITER

        curr_pos_x = int(round(x_new))
        curr_pos_y = int(round(y_new))
        curr_pos_z = int(round(z_new))

        adjuster_x = -(curr_pos_x - 50)
        adjuster_y = -(curr_pos_y - 50)
        adjuster_z = -(curr_pos_z - 50)

        # print ("indexes",curr_pos_x + adjuster_x,curr_pos_y +adjuster_y)
        # MF_value = prediction[curr_pos_x + adjuster_x]\
        #                      [curr_pos_y + adjuster_y]\
        #                      [curr_pos_z + adjuster_z]
        MF_value = prediction[50][50][50]

        # print ("MF_value",MF_value, x_new,y_new)

        gradient_x = np.reshape(gradient_x, (101, 101, 101))
        gradient_y = np.reshape(gradient_y, (101, 101, 101))
        gradient_z = np.reshape(gradient_z, (101, 101, 101))

        # initialize variables for the loop
        x_list, y_list, z_list, MF_list = [x_new], [y_new], [z_new], [MF_value]
        
        # save the original learning rate for the decay
        if l_r_decay:
            original_l_r = l_r
        # keep looping until your desired precision
        # the while condition needs to be checked
        while (
            abs(MF_value - np.amax(prediction)) > threshold and len(x_list) < max_iter
        ):
            # abs(x_new - x_prev) > threshold or abs(y_new - y_prev) > threshold:
            
            # learning rate decay
            # we consider that the l_r is logarithmically proportional to the distance to the goal 
            # and that the orignal value of l_r to correspond to the maximum distance in the map (sqrt(3)*50) 
            if l_r_decay:
                max_dist_to_goal = math.sqrt(3)*50
                curr_dist_to_goal = np.linalg.norm(np.array((self.x_goal, self.y_goal, self.z_goal)) - np.array((x_new, y_new, z_new)))
                # we want l_r to be equal original_l_r at the max_dist_to_goal
                l_r = original_l_r * math.log(curr_dist_to_goal+1) / math.log(max_dist_to_goal + 1)

            # change the value of x
            x_prev = x_new
            y_prev = y_new
            z_prev = z_new

            # get the derivation of the old value

            d_x = gradient_x[(int(round(x_prev)) + adjuster_x)] \
                            [(int(round(y_prev)) + adjuster_y)] \
                            [(int(round(z_prev)) + adjuster_z)]

            d_y = gradient_y[(int(round(x_prev)) + adjuster_x)] \
                            [(int(round(y_prev)) + adjuster_y)] \
                            [(int(round(z_prev)) + adjuster_z)]

            d_z = gradient_z[(int(round(x_prev)) + adjuster_x)] \
                            [(int(round(y_prev)) + adjuster_y)] \
                            [(int(round(z_prev)) + adjuster_z)]
            # print (d_x, d_y, d_z)

            # get your new position by adding the previous, the multiplication of the derivative and the learning rate
            t = len(x_list) + 1
            v_x = beta1 * v_x + (1.0 - beta1) * d_x
            v_y = beta1 * v_y + (1.0 - beta1) * d_y
            v_z = beta1 * v_z + (1.0 - beta1) * d_z

            sqr_x = beta2 * sqr_x + (1.0 - beta2) * np.square(d_x)
            sqr_y = beta2 * sqr_y + (1.0 - beta2) * np.square(d_y)
            sqr_z = beta2 * sqr_z + (1.0 - beta2) * np.square(d_z)

            v_bias_corr_x = v_x / (1.0 - beta1**t)
            v_bias_corr_y = v_y / (1.0 - beta1**t)
            v_bias_corr_z = v_z / (1.0 - beta1**t)

            sqr_bias_corr_x = sqr_x / (1.0 - beta2**t)
            sqr_bias_corr_y = sqr_y / (1.0 - beta2**t)
            sqr_bias_corr_z = sqr_z / (1.0 - beta2**t)
            
            div_x = l_r * v_bias_corr_x / (np.sqrt(sqr_bias_corr_x) + eps_stable)
            div_y = l_r * v_bias_corr_y / (np.sqrt(sqr_bias_corr_y) + eps_stable)
            div_z = l_r * v_bias_corr_z / (np.sqrt(sqr_bias_corr_z) + eps_stable)

            x_new = x_prev + div_x
            if x_new > 50.0 + curr_pos_x or x_new < -50.0 + curr_pos_x:
                x_new = x_prev
            y_new = y_prev + div_y
            if y_new > 50.0 + curr_pos_y or y_new < -50.0 + curr_pos_y:
                y_new = y_prev
            z_new = z_prev + div_z
            if z_new > 50.0 + curr_pos_z or z_new < -50.0 + curr_pos_z:
                z_new = z_prev

            x_list.append(x_new)
            y_list.append(y_new)
            z_list.append(z_new)

            MF_value = prediction[int(round(x_new)) + adjuster_x][
                int(round(y_new)) + adjuster_y
            ][int(round(z_new)) + adjuster_z]

            # append the new value of MF to a list of all predictions-s for later visualization of path
            MF_list.append((MF_value))

            # print("MF values", MF_list)
            # print("max_pred", "MF value")
            # print(np.amax(prediction), MF_value)
            # print (MF_value>numpy.amax(prediction))
        
        logger.info(f"ADAM: The learning rate of the last iteration is {l_r}")
        if len(x_list) >= max_iter:
            logger.info("ADAM: the model is diverging")
            # print(MF_list)
            # raise Exception ("the model is diverging")

        else:
            logger.info("ADAM: path successfully computed")
            logger.info(f"ADAM: global maximum occurs at: ({x_new}, {y_new}, {z_new})")
        logger.info(f"ADAM: Number of steps: {len(x_list)}")
        # print([item for item in zip (x_list, y_list)])
        return x_list, y_list, z_list

    def get_scene(self, lidar_points):

        curr_pos_x = int(round(self.position_x))
        curr_pos_y = int(round(self.position_y))
        curr_pos_z = int(round(self.position_z))

        # goal coordinates in robot coordinates system:
        x_goal_body = (self.x_goal - curr_pos_x + 50) / 10
        y_goal_body = (self.y_goal - curr_pos_y + 50) / 10
        z_goal_body = (self.z_goal - curr_pos_z + 50) / 10

        goal_body = (x_goal_body, y_goal_body, z_goal_body)

        # coordinates of points surrounding the robot (101x101x101)
        coordsx = []
        coordsy = []
        coordsz = []

        for i in np.arange(0, 10 + 0.1, 0.1):
            for k in range(0, 101):
                for j in range(0, 101):
                    coordsx.append(i)

        for k in range(0, 101):
            for i in np.arange(0, 10 + 0.1, 0.1):
                for j in range(0, 101):
                    coordsy.append(i)

        for k in range(0, 101):
            for j in range(0, 101):
                for i in np.arange(0, 10 + 0.1, 0.1):
                    coordsz.append(i)

        # distances of all surrounding points to the goal
        distance = self.Compute_distance_fn(
            x_goal_body, y_goal_body, z_goal_body, coordsx, coordsy, coordsz
        )

        # conductivities of all surrounding points
        conductivity = np.reshape(
            self.Compute_conductivity_fn(
                x_goal_body,
                y_goal_body,
                z_goal_body,
                coordsx,
                coordsy,
                coordsz,
                distance,
            ),
            (101, 101, 101),
        )
        assert not np.isinf(
            conductivity
        ).any(), "The conductivity map contains inf element."

        # add obstacles information by setting conductivities to 0
        if len(lidar_points) >= 1:
            for k, j, i in zip(
                lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
            ):
                # x = int(round(k))
                # y = int(round(j))
                # z = int(round(i))
                # conductivity[x+50][y+50][z+50] = 0

                x1 = math.ceil(k)
                x2 = math.floor(k)
                y1 = math.ceil(j)
                y2 = math.floor(j)
                z1 = math.ceil(i)
                z2 = math.floor(i)
                if self.in_range(x1, y1, z1):
                    conductivity[x1 + 50][y1 + 50][z1 + 50] = 0
                if self.in_range(x1, y1, z2):
                    conductivity[x1 + 50][y1 + 50][z2 + 50] = 0
                if self.in_range(x2, y2, z1):
                    conductivity[x2 + 50][y2 + 50][z1 + 50] = 0
                if self.in_range(x2, y2, z2):
                    conductivity[x2 + 50][y2 + 50][z2 + 50] = 0
                if self.in_range(x1, y2, z1):
                    conductivity[x1 + 50][y2 + 50][z1 + 50] = 0
                if self.in_range(x1, y2, z2):
                    conductivity[x1 + 50][y2 + 50][z2 + 50] = 0
                if self.in_range(x2, y1, z1):
                    conductivity[x2 + 50][y1 + 50][z1 + 50] = 0
                if self.in_range(x2, y1, z2):
                    conductivity[x2 + 50][y1 + 50][z2 + 50] = 0

        scene = np.reshape(conductivity, (1, 101, 101, 101, 1))
        scene = self.normalize_minmax_fn(scene)

        return scene, goal_body

    def plot_prediction_path(self, prediction, goal, x_list, y_list, z_list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y, z, c = [], [], [], []
        for i in range(0, 101, 20):
            for j in range(0, 101, 20):
                for k in range(0, 101, 20):
                    x.append(i)
                    y.append(j)
                    z.append(k)
                    c.append(prediction[i, j, k])

        img = ax.scatter(x, y, z, c=c, cmap=plt.jet(), label="magnetic field")
        fig.colorbar(img)
        ax.plot(goal[0], goal[1], goal[2], "x", label="goal")
        ax.plot(x_list, y_list, z_list, linewidth=3, label="path")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        plt.show()

    def plot_conductivity(self, conductivity, goal):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y, z, c = [], [], [], []
        for i in range(0, 101, 20):
            for j in range(0, 101, 20):
                for k in range(0, 101, 20):
                    x.append(i)
                    y.append(j)
                    z.append(k)
                    c.append(conductivity[i, j, k])

        img = ax.scatter(x, y, z, c=c, cmap=plt.jet(), label="magnetic field")
        fig.colorbar(img)
        ax.plot(goal[0], goal[1], goal[2], "x", label="goal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        plt.show()


def run_experiment(experiment=None, name=None, goal_constraint=False):
    if name is not None:
        logger.info(f"############################{name}############################")
    
    results = {}
    if experiment is None:
        start_location = START_LOC
        goal_location = GOAL
    else:
        start_location = experiment["start_location"]
        goal_location = experiment["goal_location"]
    
    results["start_location"] = start_location 
    results["goal_location"] = goal_location
    
    # pgh = CarlaPygameHelper(height=HEIGHT, width=WIDTH)
    pgh = None
    
    drone = LidarTest(
        start_point=start_location,
        goal=goal_location,
        export_directory=EXPORT_DIRECTORY,
        vehicle_name="Drone1",
        lidar_name="Lidar1",
        radius=0,
        angle=0,
    )
    logger.info(f"{drone.drone_name}'s goal: {(drone.x_goal, drone.y_goal, drone.z_goal)}")

    try:
        # init NN model
        model = build_cnn_3d(
            input_shape=INPUT_SHAPE, kernel_regularizer=tf.keras.regularizers.l2(l=0.1)
        )
        model.load_weights(MODEL_PATH).expect_partial() 
        logger.info("Model is sucessflly loaded")

        # init carla
        world, spec_ctrl, actor_list = init_world_and_controller(
            start_location=start_location, start_rotation=START_ROT
        )
        # debug = world.debug
        next_loc = spec_ctrl.spectator.get_transform().location
        next_rot = spec_ctrl.spectator.get_transform().rotation
        
        distance_to_goal = np.linalg.norm(np.array((drone.x_goal, drone.y_goal, drone.z_goal)) - np.array((drone.position_x, drone.position_y, drone.position_z)))
        min_dist_to_goal = distance_to_goal 
        
        # draw the goal
        # draw_goal(debug, GOAL)

        iteration = 0
        logger.info("start")
        with CarlaSyncMode(
            world=world, sensor_list=actor_list, fps=SIM_FPS
        ) as sync_mode:
            while iteration<MAX_EXP_ITERATIONS:
                start_time = time.time()
                iteration += 1
                logger.info("########################################")
                logger.info(f"number of iteration {iteration}")
                lidar_data = advance_simulation(
                    pgh,
                    spec_ctrl,
                    sync_mode,
                    next_loc=next_loc,
                    next_rot=next_rot,
                    auto_mode=True,
                )
                logger.info(f"current loc: {spec_ctrl.spectator.get_transform().location}")
                drone.set_position(next_loc)
                logger.info(f"drone position {(drone.position_x, drone.position_y, drone.position_z)}")
                logger.info(f"{drone.drone_name}'s goal: {(drone.x_goal, drone.y_goal, drone.z_goal)}")
                
                distance_to_goal = np.linalg.norm(np.array((drone.x_goal, drone.y_goal, drone.z_goal)) - np.array((drone.position_x, drone.position_y, drone.position_z)))
                logger.info(f"distance to goal is {distance_to_goal}")
                if distance_to_goal < min_dist_to_goal:
                    min_dist_to_goal = distance_to_goal
                
                local_points, world_points = transform_lidar_data(lidar_data, spec_ctrl)
                                
                # detect_obstacles(local_points, lidar_data.horizontal_angle)
                time_1 = time.time()
                x_list, y_list, z_list = drone.predict(model, local_points, iteration, goal_constraint=goal_constraint)
                time_2 = time.time()
                logger.info(f"prediction time: {(time_2 - time_1):.4f} seconds")
                
                world_path = np.array(list(zip(x_list, y_list, z_list)))
                # print(f"Number of predicted path points: {len(world_path)}")

                # draw the path in Carla
                # draw_path(debug, world_path)
                
                # check whether the goal is on the path
                world_path, goal_reached = adjust_path_to_goal(path=world_path, goal=goal_location)
                
                # check collision for the path
                if world_points.size != 0:
                    collision = detect_collision(obstacles=world_points, path=world_path)     
                else:
                    logger.info(f"No obstacles around the drone.")
                    collision = False

                # set the spectator position to the last point in the path
                # if all points on the path have a security distance to obstacles
                if not collision:
                    next_loc = carla.Location(x=world_path[-1,0], y=world_path[-1,1], z=world_path[-1,2])
                    # same rotation
                    # next_rot = next_rot
                    
                    logger.info(f"No collision")
                else:
                    # raise RuntimeError("A collision happened, path planning is stopped")
                    logger.info("A collision happened, path planning is stopped")
                    break
                # if goal reached, stop the drone
                if goal_reached:
                    logger.info("The goal is reached, the drone is stopping.")
                    logger.info(f"The final position is {next_loc}")
                    break
                
                end_time = time.time()
                logger.info(f"iteration time: {(end_time - start_time):.4f} seconds")
            
            drone.set_position(next_loc)
            spec_ctrl.spectator.set_transform(carla.Transform(next_loc, next_rot))
            time.sleep(0.5)
            distance_to_goal = np.linalg.norm(np.array((drone.x_goal, drone.y_goal, drone.z_goal)) - np.array((drone.position_x, drone.position_y, drone.position_z)))
            if distance_to_goal < min_dist_to_goal:
                min_dist_to_goal = distance_to_goal
            # save results of the exp
            results["n_iterations"] = iteration
            results["goal_reached"] = bool(goal_reached)
            results["collision"] = bool(collision)
            results["min_dist_to_goal"] = min_dist_to_goal
            results["final_dist_to_goal"] = distance_to_goal
            
    finally:

        logger.info("destroying actors.")
        for actor in actor_list:
            actor.destroy()

        # pgh.quit()
        logger.info("done.")
        
        return results


def create_random_experiments(n_exps=20,
                              environment={"x_max":400, "y_max":400,"z_max":50,}, 
                              d_drone_obstacles={"min":50,"max":float('inf')}, 
                              d_goal_obstacles={"min":50,"max":float('inf')},
                              d_goal_drone={"min":20,"max":40}):
    
    experiments = []
    logger.info("started creating random experiments")
    # init carla
    world, spec_ctrl, actor_list = init_world_and_controller(
        start_location=START_LOC, start_rotation=START_ROT
    )
    logger.info("world initialized")
    
    try:
        with CarlaSyncMode(world=world, sensor_list=actor_list, fps=SIM_FPS) as sync_mode:
            for i in range(n_exps):
                # until all conditions satisfied
                
                # select drone position  
                while True:
                
                    x_drone = random.uniform(0, environment["x_max"])
                    y_drone = random.uniform(0, environment["y_max"])
                    z_drone = random.uniform(0, environment["z_max"])
                    
                    start_location =  {"x": x_drone, "y": y_drone, "z": z_drone}
                    
                    spec_ctrl.spectator.set_transform(carla.Transform(carla.Location(**start_location), carla.Rotation(**START_ROT)))
                    time.sleep(0.5)
                    
                    snapshot, image_rgb, lidar_data = sync_mode.tick(timeout=2.0)
                    local_points, world_points = transform_lidar_data(lidar_data, spec_ctrl)
                    if len(local_points)>0: 
                        lidar_distance = pairwise_distances(local_points, np.array([0,0,0]).reshape(-1, 3), metric="euclidean").min()
                    
                        if lidar_distance>d_drone_obstacles["min"] and lidar_distance<d_drone_obstacles["max"]:
                            break
                    else:
                        if d_drone_obstacles["max"]==float("inf"):
                            break
                logger.info("drone position selected.")
            
                # select goal position
                while True:
                    r = random.uniform(d_goal_drone["min"], d_goal_drone["max"])
                    alpha = random.uniform(0, 2*math.pi)
                    beta = random.uniform(0, 2*math.pi)
                    
                    x_goal = r * math.cos(alpha) + x_drone
                    y_goal = r * math.sin(alpha) + y_drone
                    z_goal = r * math.cos(beta) + z_drone
                    
                    if not (x_goal > 0 and x_goal < environment["x_max"] and \
                       y_goal > 0 and y_goal < environment["y_max"] and \
                       z_goal > 0 and z_goal < environment["z_max"]):
                        continue
                    
                    goal_location = {"x":x_goal, "y":y_goal, "z":z_goal}
                    
                    spec_ctrl.spectator.set_transform(carla.Transform(carla.Location(**goal_location), carla.Rotation(**START_ROT)))
                    time.sleep(0.5)
                    
                    snapshot, image_rgb, lidar_data = sync_mode.tick(timeout=2.0)
                    local_points, world_points = transform_lidar_data(lidar_data, spec_ctrl)
                    if len(local_points)>0: 
                        lidar_distance = pairwise_distances(local_points, np.array([0,0,0]).reshape(-1, 3), metric="euclidean").min()
                        if lidar_distance>d_goal_obstacles["min"] and lidar_distance<d_goal_obstacles["max"]:
                            break
                    else:
                        if d_goal_obstacles["max"]==float("inf"):
                            break
                            
                logger.info("goal position selected.")

                experiment = {"start_location": {"x": x_drone, "y": y_drone, "z": z_drone},
                              "goal_location": {"x": x_goal, "y": y_goal, "z": z_goal}}
                experiments.append(experiment)
                logger.info(f"new experiment generated: {experiment}")
    
    finally:

        logger.info("destroying actors.")
        for actor in actor_list:
            actor.destroy()
        
        logger.info("finshed creating random experiments")
 
    return experiments


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
    
if __name__ == "__main__":
    
    global logger
    n_exps = N_EXPS
    try:
        # no obstacle around drone and goal

        log_file_path = EXPORT_DIRECTORY + f"exps_type_1_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
        logger = setup_logger('logger_1', log_file_path)
        results_list = []
        results_list_w_constraint = []
        
        # we choose to create one experiment at each time then execute it, because the environment is dynamic
        for i in range(n_exps):
            experiments = create_random_experiments(n_exps=1,
                                                    environment={"x_max":400, "y_max":400,"z_max":50}, 
                                                    d_drone_obstacles={"min":40,"max":float('inf')}, 
                                                    d_goal_obstacles={"min":40,"max":float('inf')},
                                                    d_goal_drone={"min":40,"max":45})
            
            logger.info(f"started executing experiment {i+1}")
            results = run_experiment(experiments[0], name="no obstacle around drone and goal", goal_constraint=False)
            results_list.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_1.json','w')as outfile:
                json.dump(results_list, outfile)
        
            results = run_experiment(experiments[0], name="no obstacle around drone and goal - with goal constraint", goal_constraint=True)
            results_list_w_constraint.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_1_w_constraint.json','w')as outfile:
                json.dump(results_list_w_constraint, outfile)

        #################################################################################################################
        
        # obstacles around drone and goal

        log_file_path = EXPORT_DIRECTORY + f"exps_type_2_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
        logger = setup_logger('logger_2', log_file_path)
        results_list = []
        results_list_w_constraint = []
        
        # we choose to create one experiment at each time then execute it, because the environment is dynamic
        for i in range(n_exps):
            experiments = create_random_experiments(n_exps=1,
                                                    environment={"x_max":400, "y_max":400,"z_max":50}, 
                                                    d_drone_obstacles={"min":10,"max":40}, 
                                                    d_goal_obstacles={"min":10,"max":40},
                                                    d_goal_drone={"min":40,"max":45})

            results = run_experiment(experiments[0], name="obstacles around drone and goal", goal_constraint=False)
            results_list.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_2.json','w')as outfile:
                json.dump(results_list, outfile)

            results = run_experiment(experiments[0], name="obstacles around drone and goal - with goal constraint", goal_constraint=True)
            results_list_w_constraint.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_2_w_constraint.json','w')as outfile:
                json.dump(results_list_w_constraint, outfile)

        #################################################################################################################

        # obstacles around drone and no around goal
        
        log_file_path = EXPORT_DIRECTORY + f"exps_type_3_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
        logger = setup_logger('logger_3', log_file_path)
        results_list = []
        results_list_w_constraint = []
        
        # we choose to create one experiment at each time then execute it, because the environment is dynamic
        for i in range(n_exps):
            experiments = create_random_experiments(n_exps=1,
                                                    environment={"x_max":400, "y_max":400,"z_max":50}, 
                                                    d_drone_obstacles={"min":10,"max":40}, 
                                                    d_goal_obstacles={"min":40,"max":float('inf')},
                                                    d_goal_drone={"min":40,"max":45})
        

            results = run_experiment(experiments[0], name="obstacles around drone and no around goal", goal_constraint=False)
            results_list.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_3.json','w')as outfile:
                json.dump(results_list, outfile)

            results = run_experiment(experiments[0], name="obstacles around drone and no around goal - with goal constraint", goal_constraint=True)
            results_list_w_constraint.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_3_w_constraint.json','w')as outfile:
                json.dump(results_list_w_constraint, outfile)

        #################################################################################################################

        # obstacles around goal and no around drone
        
        log_file_path = EXPORT_DIRECTORY + f"exps_type_4_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
        logger = setup_logger('logger_4', log_file_path)
        results_list = []
        results_list_w_constraint = []
        
        # we choose to create one experiment at each time then execute it, because the environment is dynamic
        for i in range(n_exps):
            experiments = create_random_experiments(n_exps=1,
                                                    environment={"x_max":400, "y_max":400,"z_max":50}, 
                                                    d_drone_obstacles={"min":40,"max":float('inf')}, 
                                                    d_goal_obstacles={"min":10,"max":40},
                                                    d_goal_drone={"min":40,"max":45})
            

            results = run_experiment(experiments[0], name="obstacles around goal and no around drone", goal_constraint=False)
            results_list.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_4.json','w')as outfile:
                json.dump(results_list, outfile)

            results = run_experiment(experiments[0], name="obstacles around goal and no around drone - with goal constraint", goal_constraint=True)
            results_list_w_constraint.append(results)
            with open(EXPORT_DIRECTORY+'exps_type_4_w_constraint.json','w')as outfile:
                json.dump(results_list_w_constraint, outfile)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
