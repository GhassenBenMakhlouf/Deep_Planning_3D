import pandas as pd
import numpy as np
import cmath, math
import timeit
import numba
import tensorflow.compat.v1 as tf
import concurrent.futures
import glob
import sys

# ==============================================DEFINE YOUR ARGUMENTS=============================================#
dataset_dir = "./data_comsol/data"
scenes_dir = "./data_comsol/data"
npy_dir = "./numpydata_training"

flags = tf.app.flags

# State your dataset directory
flags.DEFINE_string("dataset_dir", dataset_dir, "String; Your dataset directory")
# State your scenes directory
flags.DEFINE_string("scenes_dir", scenes_dir, "String; Your scenes directory")
# State your npy directory
flags.DEFINE_string("npy_dir", npy_dir, "String; Your numpy data directory")
#  environment conductivity
flags.DEFINE_float("sigmae", 100, "Float: The conductivity of the environment")
#  obstacle conductivity
flags.DEFINE_float("sigmao", 0, "Float: The conductivity of the obstacles")
#  goal conductivity
flags.DEFINE_float("sigmag", 1000000, "Float: The conductivity of the goal")


FLAGS = flags.FLAGS
# ============================================== END DEFINE YOUR ARGUMENTS============================================#


# ============================================== HELPER FUNCTIONS ====================================================#


def Get_Goal(filepath):
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


@numba.jit
def skip_rows_array(reduce_factor=10):
    list = []
    image_width = 1001
    if reduce_factor == 5:
        for i in range(0, image_width - 10, 10):
            for x in range(1 + i, i + 10):
                if x != i + 5:
                    list.append(x)
    else:
        for i in range(0, image_width - 10, 10):
            for x in range(1 + i, i + 10):

                list.append(x)

    return list


def reduce_resolution(data_distance, data_conductivity, data_labels, reduce_factor=10):
    data_distance = np.delete(
        data_distance, skip_rows_array(reduce_factor=reduce_factor), axis=0
    )
    data_distance = np.delete(
        data_distance, skip_rows_array(reduce_factor=reduce_factor), axis=1
    )
    print("data_conductivity shape", data_distance.shape)
    data_conductivity = np.delete(
        data_conductivity, skip_rows_array(reduce_factor=reduce_factor), axis=0
    )
    data_conductivity = np.delete(
        data_conductivity, skip_rows_array(reduce_factor=reduce_factor), axis=1
    )
    print("data_conductivity shape", data_conductivity.shape)
    data_labels = np.delete(
        data_labels, skip_rows_array(reduce_factor=reduce_factor), axis=0
    )
    data_labels = np.delete(
        data_labels, skip_rows_array(reduce_factor=reduce_factor), axis=1
    )
    print("data_labels shape", data_labels.shape)
    return data_distance, data_conductivity, data_labels


# ============================================== END HELPER FUNCTIONS =============================================#


def Compute_Data_Matrix(data_filename):
    """
    Compute complementary data out of the csv file and build the data matrix (1001x1001x3)
    Features: conductivity and distance to goal
    labels  : gradient phase
    """
    sigmae = FLAGS.sigmae
    sigmag = FLAGS.sigmag
    sigmao = FLAGS.sigmao

    point_precision = 1 / 10

    npy_dir = FLAGS.npy_dir

    # scene_filename = scenes_directory+data_filename.split('/')[-1]
    scene_filename = data_filename.replace("gradient_C", "scene_")
    # numpyfile = data_filename.replace("datasetnew/", "datasetnew/numpydata_training/")
    filename = data_filename.split("/")[-1]
    numpyfile = f"{npy_dir}/{filename}"

    numpyfile = numpyfile[:-4] + ".npy"
    print("numpyfile: ", numpyfile)

    x_goal, y_goal, z_goal, radius = Get_Goal(scene_filename)
    # print (scene_filename+x_goal+y_goal+radius)
    print("get data dataframe")
    data = pd.read_csv(
        data_filename,
        delim_whitespace=True,
        skip_blank_lines=True,
        comment="%",
        header=None,
        names=["X", "Y", "Z", "Xvector", "Yvector", "Zvector"],
    )

    @numba.vectorize
    def Compute_distance_fn(x, y, z):
        return math.sqrt(
            pow((x - x_goal), 2) + pow((y - y_goal), 2) + pow((z - z_goal), 2)
        )

    @numba.vectorize
    def Compute_phase_phi_fn(x, y, z):
        """phi is the angle between the vector and the x axis
        theta is the angle between the vector and the xy plane"""
        r, phi = cmath.polar(complex(x, y))
        if phi < 0:
            phi + 2 * math.pi

        return phi

    @numba.vectorize
    def Compute_phase_theta_fn(x, y, z):
        """phi is the angle between the vector and the x axis
        theta is the angle between the vector and the xy plane"""
        r, phi = cmath.polar(complex(x, y))
        if phi < 0:
            phi + 2 * math.pi

        theta = cmath.phase(complex(r, z))
        if theta < 0:
            theta = theta + 2 * math.pi

        return theta

    @numba.vectorize
    def Compute_conductivity_fn(x, y, z, d):
        # why not use d instead of the calculation
        if pow((x - x_goal), 2) + pow((y - y_goal), 2) + pow((z - z_goal), 2) <= pow(
            radius, 2
        ):
            return sigmag
        else:
            return sigmae / d

    ######################## add distance to goal information #####################
    print("add distance to goal information")
    data["Distance"] = Compute_distance_fn(data.X.values, data.Y.values, data.Z.values)

    ######################## add conductivity information #########################
    print("add conductivity information")
    data["conductivity"] = Compute_conductivity_fn(
        data.X.values, data.Y.values, data.Z.values, data.Distance.values
    )

    ######################## add label (phi) information ###########################
    print("add label (phi, theta) information")
    data["Phi_label"] = Compute_phase_phi_fn(
        data.Xvector.values, data.Yvector.values, data.Zvector.values
    )
    data["Theta_label"] = Compute_phase_theta_fn(
        data.Xvector.values, data.Yvector.values, data.Zvector.values
    )

    ######################## add Obstacles information ##############################
    # The dimensions of data_distance are in this order: (x,y,z)
    data_distance = (
        data.pivot(index="X", columns=["Y", "Z"], values="Distance")
        .to_numpy()
        .reshape(data.X.nunique(), data.Z.nunique(), -1)
        .transpose(0, 2, 1)
    )

    start = timeit.default_timer()
    # Replace_NAN_fn()
    nanlist = np.argwhere(np.isnan(data_distance))
    for item in nanlist:
        data_distance[item[0], item[1], item[2]] = Compute_distance_fn(
            item[0] * point_precision,
            item[1] * point_precision,
            item[2] * point_precision,
        )
    stop = timeit.default_timer()
    print("time to change nan: ", stop - start)

    # The dimensions of data_conductivity are in this order: (x,y,z)
    data_conductivity = (
        data.pivot(index="X", columns=["Y", "Z"], values="conductivity")
        .to_numpy()
        .reshape(data.X.nunique(), data.Z.nunique(), -1)
        .transpose(0, 2, 1)
    )
    data_conductivity[np.isnan(data_conductivity)] = sigmao

    data_labels_phi = (
        data.pivot(index="X", columns=["Y", "Z"], values="Phi_label")
        .to_numpy()
        .reshape(data.X.nunique(), data.Z.nunique(), -1)
        .transpose(0, 2, 1)
    )
    data_labels_phi[np.isnan(data_labels_phi)] = 0

    data_labels_theta = (
        data.pivot(index="X", columns=["Y", "Z"], values="Theta_label")
        .to_numpy()
        .reshape(data.X.nunique(), data.Z.nunique(), -1)
        .transpose(0, 2, 1)
    )
    data_labels_theta[np.isnan(data_labels_theta)] = 0

    # resolution reduction for 3D not implemented
    # ######################### Reduce the resolution to 101 X 101 if needed##########################

    # data_distance, data_conductivity, data_labels = reduce_resolution(
    #     data_distance, data_conductivity, data_labels, reduce_factor=10
    # )

    ######################## Build data matrix (101x101x101x4) ###########################
    data_matrix = np.stack(
        (data_distance, data_conductivity, data_labels_phi, data_labels_theta), axis=3
    )
    print("Build data matrix (101x101x101x4)")

    np.save(numpyfile, data_matrix)


def main():
    # ==================================================== CHECKS ========================================================#

    # Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
        raise ValueError(
            "dataset_directory is empty. Please state a dataset_directory argument."
        )
    # Check if there is the scenes' directory entered
    if not FLAGS.scenes_dir:
        raise ValueError(
            "scenes_directory is empty. Please state a scenes_directory argument."
        )

    # ================================================== END OF CHECKS ====================================================#

    start = timeit.default_timer()

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print("start")
        data_files = glob.glob(FLAGS.dataset_dir + "/gradient_C*.txt")
        print(data_files)
        # Process the list of files, but split the work across the process pool to use all CPUs!
        executor.map(Compute_Data_Matrix, data_files)
        print("finish")

    stop = timeit.default_timer()

    print("total time: ", stop - start)


if __name__ == "__main__":
    main()
