from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import numba


def get_goal(filepath):
    """
    :param filepath:
    :return: the goal defined as a list of  4 elements: center coordinates x, y, z and the radius
    """

    with open(filepath, "r") as f:
        content = f.readlines()
        goal_index = len(content) - 1
        goal = content[goal_index].rstrip("\n").split(",")
        goal = [float(item) for item in goal]
        if goal[3] >= goal[4]:
            radius = goal[4] / 2
        else:
            radius = goal[3] / 2
        x_center = goal[0] + radius
        y_center = goal[1] + radius
        z_center = goal[2] + radius
        goal = [x_center, y_center, z_center, radius]
    return goal


def Normalize_minmax_fn(x):
    """normalize values of x between 0 and 1000"""
    min = np.amin(x)
    max = np.amax(x)
    return np.multiply(1000, np.divide(np.subtract(x, min), np.subtract(max, min)))


def Normalize_log_minmax_fn(x):
    """normalize values of x between 0 and 1000"""
    # min = np.amin(x)
    max = np.amax(x)
    return np.multiply(1000, np.divide(np.log10(x), np.log10(max)))


def preprocess_data(config):
    cfg = config.comsol_data

    data_path = Path(cfg.data_path)
    output_path = Path(cfg.output_path)
    sigmae = cfg.sigmae
    sigmag = cfg.sigmag
    sigmao = cfg.sigmao

    # point_precision = 1 / 10

    data_files = [x.name for x in sorted(data_path.glob("magneticfield_*.txt"))]

    for data_file in tqdm(data_files):
        # set paths
        mf_path = data_path / data_file
        scene_file = data_file.replace("magneticfield_C", "scene_")
        scene_path = data_path / scene_file
        numpy_file_path = output_path / (data_file[:-4] + ".npy")

        x_goal, y_goal, z_goal, radius = get_goal(scene_path)

        data = pd.read_csv(
            mf_path,
            delim_whitespace=True,
            skip_blank_lines=True,
            comment="%",
            header=None,
            names=["X", "Y", "Z", "Radius", "Color"],
            low_memory=False, 
            # dtype={
            #     "X": np.float64,
            #     "Y": np.float64,
            #     "Z": np.float64,
            #     "Radius": np.int64,
            #     "Color": np.float64,
            # },
        )
        data.X = data.X.round(decimals=1)
        data.Y = data.Y.round(decimals=1)
        data.Z = data.Z.round(decimals=1)
        data.sort_values(by=["X", "Y", "Z"], inplace=True)

        @numba.vectorize
        def Compute_distance_fn(x, y, z):
            return math.sqrt(
                pow((x - x_goal), 2) + pow((y - y_goal), 2) + pow((z - z_goal), 2)
            )

        @numba.vectorize
        def Compute_conductivity_fn(x, y, z, d):
            # why not use d instead of the calculation
            if pow((x - x_goal), 2) + pow((y - y_goal), 2) + pow(
                (z - z_goal), 2
            ) <= pow(radius, 2):
                return sigmag
            else:
                return sigmae / d

        ######################## add distance to goal information #####################
        # print("add distance to goal information")
        data["Distance"] = Compute_distance_fn(
            data.X.values, data.Y.values, data.Z.values
        )

        ######################## add conductivity information #########################
        # print("add conductivity information")

        data["conductivity"] = Compute_conductivity_fn(
            data.X.values, data.Y.values, data.Z.values, data.Distance.values
        )

        ######################## add Obstacles information ##############################

        # # The dimensions of data_distance are in this order: (x,y,z)
        # data_distance = (
        #     data.pivot(index="X", columns=["Y", "Z"], values="Distance")
        #     .to_numpy()
        #     .reshape(data.X.nunique(), data.Y.nunique(), -1)
        # )

        # start = timeit.default_timer()
        # # Replace_NAN_fn()
        # nanlist = np.argwhere(np.isnan(data_distance))
        # for item in nanlist:
        #     data_distance[item[0], item[1], item[2]] = Compute_distance_fn(
        #         item[0] * point_precision,
        #         item[1] * point_precision,
        #         item[2] * point_precision,
        #     )
        # stop = timeit.default_timer()
        # print("time to change nan: ", stop - start)

        # The dimensions of data_conductivity are in this order: (x,y,z)
        data_conductivity = (
            data.pivot(index="X", columns=["Y", "Z"], values="conductivity")
            .to_numpy()
            .reshape(data.X.nunique(), data.Y.nunique(), -1)
        )
        assert (
            np.nanmin(data_conductivity) > 0
        ), f"there are negative conductivity values in {data_file}"
        data_conductivity[np.isnan(data_conductivity)] = sigmao
        data_conductivity = Normalize_minmax_fn(data_conductivity)
        assert (data_conductivity >= 0).all() and (
            data_conductivity <= 1000
        ).all(), f"There are normalized conductivity values in {data_file} not between 0 and 1000 or nan."

        data_labels = (
            data.pivot(index="X", columns=["Y", "Z"], values="Color")
            .to_numpy()
            .reshape(data.X.nunique(), data.Y.nunique(), -1)
        )
        obstacle_label = 0
        if not np.nanmin(data_labels) > 0:
            obstacle_label = np.nanmin(data_labels)
        data_labels[np.isnan(data_labels)] = obstacle_label
        data_labels = Normalize_minmax_fn(data_labels)
        assert (data_labels >= 0).all() and (
            data_labels <= 1000
        ).all(), f"There are normalized MF values in {data_file} not between 0 and 1000"

        ######################## Build data matrix (101x101x101x2) ###########################
        data_matrix = (
            np.stack(
                (data_conductivity, data_labels),
                axis=3,
            )
        ).astype(np.float32)

        np.save(numpy_file_path, data_matrix)
