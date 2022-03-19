# Python client example to get Lidar data from a drone

import setup_path
from airsim import client
import airsim
from airsim.types import YawMode
from airsim.types import DrivetrainType

import sys
import math
import time
import argparse

# import pprint
import numpy as np
import matplotlib.pyplot as plt

import os
import tensorflow as tf
import timeit

# from scipy import signal
# Makes the drone fly and get Lidar data
# from use_input_pipeline import input_fn

from models import build_cnn_3d

MODEL_PATH = "./checkpoints/cp-final.ckpt"
EXPORT_DIRECTORY = "/bigdrive/drone_simulation/exps"

SIGMAG = 1e6
SIGMAE = 100
SIGMAO = 0


class LidarTest:

    def __init__(
        self,
        start_point_x,
        start_point_y,
        start_point_z,
        export_directory,
        vehicle_name,
        lidar_name,
        radius,
        angle,
    ):

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name)

        # init variables
        self.drone_name = vehicle_name
        self.lidar_name = lidar_name
        self.x_goal = 0
        self.y_goal = 0
        self.z_goal = 0
        self.position_x = start_point_x
        self.position_y = start_point_y
        self.position_z = start_point_z
        self.export_directory = export_directory
        self.command = ""
        self.exit = 0
        self.offset_x = start_point_x
        self.offset_y = start_point_y
        self.offset_z = start_point_z
        self.radius = radius
        self.angle = angle
        # plt.switch_backend("Agg")

    def normalize_input(self, input, min, max):
        """

        :param input: is a tensor with shape (101,101,1)
        :return: normalized input in the range of [1,100]
        """

        range = max - min
        return np.add(np.multiply(input, range), min)

    def Compute_distance_fn(self, x_goal, y_goal, z_goal, x, y, z):

        return np.sqrt(
            np.power(np.subtract(x, x_goal), 2) +
            np.power(np.subtract(y, y_goal), 2) +
            np.power(np.subtract(z, z_goal), 2))

    def in_range(self, x1, y1, z1):
        if x1 <= 50 and y1 <= 50 and z1 <= 50 and x1 >= -50 and y1 >= -50 and z1 >= -50:
            return True
        else:
            return False

    def Normalize_minmax_fn(self, x):
        """normalize values of x between 0 and 1000"""
        min = np.amin(x)
        max = np.amax(x)
        return np.multiply(
            1000, np.divide(np.subtract(x, min), np.subtract(max, min)))

    def Compute_conductivity_fn(self, x_goal, y_goal, z_goal, x, y, z, d):
        radius = 0.09
        # data = []
        # print("d", d.shape)
        a = np.add(
            np.power(np.subtract(x, x_goal), 2),
            np.power(np.subtract(y, y_goal), 2),
            np.power(np.subtract(z, z_goal), 2),
        )
        # for i in range(0, a.shape[0]):
        #     if a[i] <= pow(radius, 2):
        #         data.append(SIGMAG)
        #     else:
        #         data.append(SIGMAE / d[i])

        data = np.where(a <= pow(radius, 2), SIGMAG, SIGMAE / d)

        return data

    def bilinear_interpolation(self, x, y, points):
        """Interpolate (x,y) from values associated with four points.
        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.
            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0

        """
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
        # order points by x, then by y
        points = sorted(points)
        (x1, y1, g11), (_x1, y2, g12), (x2, _y1, g21), (_x2, _y2, g22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            print("points do not form a rectangle")
            raise ValueError("points do not form a rectangle")
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            print(x, y, x1, x2, y1, y2)
            print("(x, y) not within the rectangle")
            raise ValueError("(x, y) not within the rectangle")

        return (g11 * (x2 - x) * (y2 - y) + g21 * (x - x1) * (y2 - y) + g12 *
                (x2 - x) * (y - y1) + g22 * (x - x1) *
                (y - y1)) / ((x2 - x1) * (y2 - y1) + 0.0)

    def predict(self, model, iteration):

        rounded_position_x = int(round(self.position_x))
        rounded_position_y = int(round(self.position_y))
        rounded_position_z = int(round(self.position_z))

        scene, goal_body = self.get_data(
            rounded_position_x,
            rounded_position_y,
            rounded_position_z,
            self.lidar_name,
            self.drone_name,
        )

        # subinputs = np.split(scene, 2, axis=4)
        # in_cond = subinputs[1]
        # print(in_cond.shape)
        prediction = model(scene, training=False)
        prediction = np.reshape(prediction, (101, 101, 101))

        grad_x, grad_y, grad_z = np.gradient(prediction, edge_order=1)

        x_list, y_list, z_list = self.adam(
            self.position_x,
            self.position_y,
            self.position_z,
            0.5,
            prediction,
            grad_x,
            grad_y,
            grad_z,
        )
        # print ("plot data")
        # if self.drone_name=="Drone1":
        #    self.plot_data_and_path (points_coords, scene,points,prediction, x_list, y_list,iteration)
        # self.plot_data_and_path (points_coords, scene,points,prediction, x_list, y_list,iteration)
        # print("end plot data")

        print("goal is: ", goal_body)
        curr_pos_x = int(round(self.position_x))
        curr_pos_y = int(round(self.position_y))
        curr_pos_z = int(round(self.position_z))

        adjuster_x = -(curr_pos_x - 50)
        adjuster_y = -(curr_pos_y - 50)
        adjuster_z = -(curr_pos_z - 50)

        scene = np.reshape(scene, (101, 101, 101))
        # self.plot_conductivity(scene, [i * 10 for i in goal_body])

        self.plot_prediction_path(prediction, [i * 10 for i in goal_body],
                                  [i + adjuster_x for i in x_list],
                                  [i + adjuster_y for i in y_list],
                                  [i + adjuster_z for i in z_list])

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
    ):
        """
        https://arxiv.org/pdf/1412.6980.pdf

        Description: This function takes in an initial or previous value for (x,y), updates it based on
        steps taken via the learning rate and outputs the most optimum position (x,y) that reaches the threshold satisfaction.

        Arguments:

        x_new, y_new - a starting position that will get updated based on the learning rate

        x_prev, y_prev - the previous position that is getting updated to the new one

        threshold - a precision that determines the stop of the stepwise descent

        l_r - the learning rate (size of each descent step)

        Output:

        1. x_list : x coordinates of the way points which equates to the number of gradient descent steps
        2. y_list : y coordinates of the way points which equates to the number of gradient descent steps


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
        threshold = 1
        max_iter = 100

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

        # keep looping until your desired precision
        # the while condition needs to be checked
        while abs(MF_value - np.amax(prediction)) > threshold \
              and len(x_list) < max_iter:
            # abs(x_new - x_prev) > threshold or abs(y_new - y_prev) > threshold:

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

            div_x = l_r * v_bias_corr_x / (np.sqrt(sqr_bias_corr_x) +
                                           eps_stable)
            div_y = l_r * v_bias_corr_y / (np.sqrt(sqr_bias_corr_y) +
                                           eps_stable)
            div_z = l_r * v_bias_corr_z / (np.sqrt(sqr_bias_corr_z) +
                                           eps_stable)

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

            MF_value = prediction[int(round(x_new)) + adjuster_x] \
                                 [int(round(y_new)) + adjuster_y] \
                                 [int(round(z_new)) + adjuster_z]

            # append the new value of MF to a list of all predictions-s for later visualization of path
            MF_list.append((MF_value))

            # print("MF values", MF_list)
            print("max_pred", "MF value")
            print(np.amax(prediction), MF_value)
            # print (MF_value>numpy.amax(prediction))

        if len(x_list) >= max_iter:
            print("the model is diverging")
            # print(MF_list)
            # raise Exception ("the model is diverging")

        else:
            print("path successfully computed")
            print(f"global maximum occurs at: ({x_new}, {y_new}, {z_new})")
        print(f"Number of steps: {len(x_list)}")
        # print([item for item in zip (x_list, y_list)])
        return x_list, y_list, z_list

    # def access_dataset(self):
    #     testing_dataset = input_fn("C://AIM//tfdata//",
    #                                batch_size=1,
    #                                num_epochs=1,
    #                                dataset_type="testing")

    #     testing_iterator = testing_dataset.make_one_shot_iterator()
    #     return testing_iterator

    def get_data(self, curr_pos_x, curr_pos_y, curr_pos_z, lidar_name,
                 drone_name):

        lidarData = self.client.getLidarData(lidar_name=lidar_name,
                                             vehicle_name=drone_name)

        if len(lidarData.point_cloud) < 3:
            print("\tNo points received from Lidar data")
            points = np.ones(1)
        else:
            points = self.parse_lidarData(lidarData)

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
                    coordsz.append(i)

        for k in range(0, 101):
            for i in np.arange(0, 10 + 0.1, 0.1):
                for j in range(0, 101):
                    coordsy.append(i)

        for k in range(0, 101):
            for j in range(0, 101):
                for i in np.arange(0, 10 + 0.1, 0.1):
                    coordsx.append(i)

        # distances of all surrounding points to the goal
        distance = self.Compute_distance_fn(x_goal_body, y_goal_body,
                                            z_goal_body, coordsz, coordsy,
                                            coordsx)
        # print ("distance computed")

        # conductivities of all surrounding points
        conductivity = np.reshape(
            self.Compute_conductivity_fn(
                x_goal_body,
                y_goal_body,
                z_goal_body,
                coordsz,
                coordsy,
                coordsx,
                distance,
            ),
            (101, 101, 101),
        )
        assert not np.isinf(
            conductivity).any(), "The conductivity map contains inf element."
        # distance = self.Normalize_minmax_fn(distance)
        # distance_data = np.reshape(distance, (101, 101, 101))

        # add obstacles information by setting conductivities to 0
        if len(points.flatten()) >= 3:
            for k, j, i in zip(points[:, 0], points[:, 1], points[:, 2]):
                # x = int(round(k))
                # y = int(round(j))
                # conductivity[x+50][y+50] = 0
                for iter in range(1, 3):

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
        # print ("obstacles information added to the map")

        # print ("scene built")

        scene = np.reshape(conductivity, (1, 101, 101, 101, 1))
        scene = self.Normalize_minmax_fn(scene)
        # print ("scene reshaped")

        # print("min cond",numpy.amin(conductivity))
        # print("max cond",numpy.amax(conductivity))

        return scene, goal_body

    def parse_lidarData(self, data):

        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype("f4"))
        # print ('shape of points before reshape',points.shape)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        # print ('shape of points after reshape',points.shape)

        return points

    def build_conductivity_map(self, coordsx, coordsy, data_conductivity,
                               imagewidth, step, ax, title_str):
        marker_size = 5
        # coordsx = []
        # coordsy = []
        area_length = 10
        # for i in numpy.arange(0, area_length+step,step):
        #    for j in range(0,imagewidth):
        #        coordsy.append(i)
        # for j in range(0,imagewidth):
        #    for i in numpy.arange(0, area_length+step,step):
        #        coordsx.append(i)

        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)
        ax.scatter(coordsx,
                   coordsy,
                   marker_size,
                   c=data_conductivity,
                   cmap="jet",
                   alpha=1)
        ax.set_title(title_str)

    def write_lidarData_to_disk(self, points):
        # TODO
        print("not yet implemented")

    def stop(self):
        self.client.landAsync(vehicle_name=self.drone_name)
        self.client.hoverAsync(vehicle_name=self.drone_name).join()
        #

        self.client.armDisarm(False, self.drone_name)
        self.client.reset()

        self.client.enableApiControl(False, self.drone_name)
        print("Done!\n")

    # main
    def plot_data_and_path(self, points_coords, scene, points, prediction,
                           x_list, y_list, iteration):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(35, 10))
        self.build_conductivity_map(
            points_coords[:, 1],
            points_coords[:, 0],
            scene[0, :, :, 1].flatten(),
            101,
            0.1,
            ax1,
            " conductivity 101x101",
        )
        self.build_conductivity_map(
            points_coords[:, 1],
            points_coords[:, 0],
            scene[0, :, :, 0].flatten(),
            101,
            0.1,
            ax2,
            " dist 101x101",
        )
        self.build_conductivity_map(
            points_coords[:, 1],
            points_coords[:, 0],
            prediction.flatten(),
            101,
            0.1,
            ax3,
            " prediction 101x101",
        )
        # self.build_conductivity_map(points_coords[:,1],points_coords[:,0],x_list.flatten(), 101, 0.1,ax1, '101x101')
        # self.build_conductivity_map(points_coords[:,1],points_coords[:,0],y_list.flatten(), 101, 0.1,ax2, '101x101')
        ax4.scatter(points[:, 1], points[:, 0], s=0.2, c="g")

        ax3.scatter(y_list, x_list, s=80, color="g", marker=(5, 2))

        plt.savefig("C://Users//Aymen//Documents//AirSim//scenes" +
                    str(iteration) + ".png")

    def moveToPosition(self, x, y, z, v):
        currentPos = self.client.getMultirotorState(
            vehicle_name=self.drone_name).kinematics_estimated.position
        currentPos_x = currentPos.x_val + drone.offset_x
        currentPos_y = currentPos.y_val + drone.offset_y
        currentPos_z = currentPos.z_val + drone.offset_z

        t = ((currentPos_x - x)**2 + (currentPos_y - y)**2 +
             (currentPos_z - z)**2)**0.5 / v
        delta_x = x - currentPos_x
        delta_y = y - currentPos_y
        delta_z = z - currentPos_z
        vx = delta_x / t
        vy = delta_y / t
        vz = delta_z / t
        self.client.moveByVelocityAsync(vx,
                                        vy,
                                        vz,
                                        t,
                                        vehicle_name=self.drone_name)
        time.sleep(t)
        self.client.moveByVelocityAsync(0,
                                        0,
                                        0,
                                        1,
                                        vehicle_name=self.drone_name)
        self.client.hoverAsync(vehicle_name=self.drone_name).join()

    def plot_prediction_path(self, prediction, goal, x_list, y_list, z_list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
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
        ax.plot(goal[0], goal[1], goal[2], 'x', label="goal")
        ax.plot(x_list, y_list, z_list, linewidth=3, label="path")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        plt.show()

    def plot_conductivity(self, conductivity, goal):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
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
        ax.plot(goal[0], goal[1], goal[2], 'x', label="goal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # use tensorflow with cpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # args = sys.argv
    # args.pop(0)

    # arg_parser = argparse.ArgumentParser(
    #     "Lidar.py makes drone fly and gets Lidar data")

    # arg_parser.add_argument("-save-to-disk",
    #                         type=bool,
    #                         help="save Lidar data to disk",
    #                         default=False)

    # args = arg_parser.parse_args(args)

    nb_drones = 1
    x_center = -20
    y_center = -10  # 50
    z_center = 0

    lidarTest1 = LidarTest(
        start_point_x=0,
        start_point_y=0,
        start_point_z=0,
        export_directory=EXPORT_DIRECTORY,
        vehicle_name="Drone1",
        lidar_name="Lidar1",
        radius=0,
        angle=0,
    )

    lidarTest1.client.simPlotPoints(
        [airsim.Vector3r(x_center, y_center, z_center)],
        color_rgba=[0.0, 1.0, 0.0, 1.0],
        size=20.0,
        duration=-1.0,
        is_persistent=True)

    Drones_list = [lidarTest1]
    for drone in Drones_list:
        drone.x_goal = drone.radius * math.cos(drone.angle) + x_center
        drone.y_goal = drone.radius * math.sin(drone.angle) + y_center
        drone.z_goal = z_center
        print(drone.drone_name, " ",
              (drone.x_goal, drone.y_goal, drone.z_goal))

    print("arming the drones...")
    for drone in Drones_list:
        drone.client.armDisarm(True, drone.drone_name)

    airsim.wait_key("Press any key to takeoff")
    for drone in Drones_list:
        drone.command = drone.client.takeoffAsync(
            vehicle_name=drone.drone_name)

    # all join() to wait for task to complete
    for drone in Drones_list:
        f = drone.command
        f.join()
        pos = drone.client.getMultirotorState(
            vehicle_name=drone.drone_name).kinematics_estimated.position
        drone.position_x = pos.x_val + drone.offset_x
        drone.position_y = pos.y_val + drone.offset_y
        drone.position_z = pos.z_val + drone.offset_z
        print("x_position", drone.position_x)
        print("y_position", drone.position_y)
        print("z_position", drone.position_z)

    try:
        input_shape = (101, 101, 101, 1)
        model = build_cnn_3d(
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(l=0.1))
        model.load_weights(MODEL_PATH)
        print("Model is sucessflly loaded")

        iteration = 0
        print("start")
        while True:
            iteration += 1
            print("number of iteration", iteration)
            start = timeit.default_timer()

            for drone in Drones_list:
                if drone.exit == 0:

                    x_list, y_list, z_list = drone.predict(model, iteration)

                    print(list(zip(x_list, y_list, z_list)))

                    airsim.wait_key("Press any key to follow the path")

                    if (drone.Compute_distance_fn(
                            drone.x_goal,
                            drone.y_goal,
                            drone.z_goal,
                            drone.position_x,
                            drone.position_y,
                            drone.position_z,
                    ) > 2):

                        drone.command = drone.client.moveOnPathAsync(
                            [
                                airsim.Vector3r(
                                    x_list[i] - drone.offset_x,
                                    y_list[i] - drone.offset_y,
                                    z_list[i] - drone.offset_z,
                                ) for i in range(0, len(x_list))
                            ],
                            3,
                            300,
                            airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 0),
                            20,
                            1,
                            vehicle_name=drone.drone_name,
                        )

                        # for x, y, z in list(zip(x_list, y_list, z_list)):
                        #     drone.client.moveToPositionAsync(x, y, z, 1)
                        print("distance > 2")
                    else:
                        drone.command = drone.client.moveOnPathAsync(
                            [
                                airsim.Vector3r(
                                    x_list[i] - drone.offset_x,
                                    y_list[i] - drone.offset_y,
                                    z_list[i] - drone.offset_z,
                                ) for i in range(0,
                                                 len(x_list) - 3)
                            ],
                            3,
                            120,
                            airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 0),
                            20,
                            1,
                            vehicle_name=drone.drone_name,
                        )
                        drone.exit = 1

            for drone in Drones_list:
                # print ("##################### run commands for "+  drone.drone_name+ "###############################")

                if drone.exit == 0:
                    f = drone.command
                    # f.join()
                    # drone.client.hoverAsync(vehicle_name=drone.drone_name).join()
                    print(drone.drone_name + "  is executing its path")
                elif drone.exit == 1:
                    f = drone.command
                    # f.join()
                    # drone.client.hoverAsync(vehicle_name=drone.drone_name).join()
                    # drone.moveToPosition(drone.x_goal,drone.y_goal, -3,3)
                    # drone.client.moveToPositionAsync(drone.x_goal,drone.y_goal, -3, 1,vehicle_name=drone.drone_name).join()

                    # drone.client.hoverAsync(vehicle_name=drone.drone_name).join()

                    drone.exit = 2

                    print(drone.drone_name + "  has reached its goal")
                else:
                    drone.angle = drone.angle + (math.pi) / 2
                    drone.x_goal = drone.radius * \
                        math.cos(drone.angle) + x_center
                    drone.y_goal = drone.radius * \
                        math.sin(drone.angle) + y_center
                    drone.z_goal = drone.radius * \
                        math.sin(drone.angle) + z_center
                    print(
                        "%%%%%%%%%%%%%%%%%%%%%%%% new_goal %%%%%%%%%%%%%%%%%" +
                        drone.drone_name + "%%%%%%%%%%%%%%%%",
                        (drone.x_goal, drone.y_goal, drone.z_goal),
                    )
                    print(
                        "%%%%%%%%%%%%%%%%%%%%%%%% new_angle %%%%%%%%%%%%%%%%%",
                        drone.angle,
                    )
                    # drone.moveToPosition(drone.x_goal,drone.y_goal, -3,3)
                    # drone.client.hoverAsync(vehicle_name=drone.drone_name).join()

                    print(drone.drone_name +
                          " has reached its goal and  it is waiting others")
                    drone.exit = 0

                airsim.wait_key("Press any key to start new prediction")

                pos = drone.client.getMultirotorState(
                    vehicle_name=drone.drone_name
                ).kinematics_estimated.position
                drone.position_x = pos.x_val + drone.offset_x
                drone.position_y = pos.y_val + drone.offset_y
                drone.position_z = pos.z_val + drone.offset_z
                print("x_position", drone.position_x)
                print("y_position", drone.position_y)
                print("z_position", drone.position_z)
                # pos = drone.client.simGetVehiclePose(vehicle_name=drone.drone_name).position
                # pos = drone.client.simGetVehiclePose(vehicle_name=drone.drone_name).position

            stop = timeit.default_timer()
            print("iteration time", stop - start)

            # except:
            #    print ("a problem in iteration "+str(iteration)+".")

    finally:
        airsim.wait_key("Press any key to reset to original state")
        for drone in Drones_list:
            # drone.client.moveToPositionAsync(drone.x_goal,drone.y_goal, -3, 1.5,vehicle_name=drone.drone_name).join()
            # drone.client.hoverAsync(vehicle_name=drone.drone_name).join()
            drone.stop()
