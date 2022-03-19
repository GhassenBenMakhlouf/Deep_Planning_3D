import setup_path
from airsim import client
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
name = "Drone1"
client.enableApiControl(True, name)

client.simPlotPoints([airsim.Vector3r(0, 0, 0)],
                     color_rgba=[0.0, 1.0, 0.0, 1.0],
                     size=20.0,
                     duration=-1.0,
                     is_persistent=True)

client.armDisarm(True, name)
airsim.wait_key("Press any key to takeoff")
c = client.takeoffAsync(vehicle_name=name)
c.join()

while True:
    airsim.wait_key("Press any key to move to 0 0 0")

    client.moveOnPathAsync([
        airsim.Vector3r(0, 0, 0),
        airsim.Vector3r(10, 0, 0),
        airsim.Vector3r(10, 20, 0)
    ],
                           3,
                           120,
                           airsim.DrivetrainType.MaxDegreeOfFreedom,
                           airsim.YawMode(False, 0),
                           20,
                           1,
                           vehicle_name=name)
    airsim.wait_key("Press any key to move to 10 0 0")

    client.moveOnPathAsync([airsim.Vector3r(10, 0, 0)],
                           3,
                           120,
                           airsim.DrivetrainType.MaxDegreeOfFreedom,
                           airsim.YawMode(False, 0),
                           20,
                           1,
                           vehicle_name=name)

    airsim.wait_key("Press any key to move to 0 0 0")

    client.moveOnPathAsync([
        airsim.Vector3r(10, 0, 0),
        airsim.Vector3r(5, 0, 0),
        airsim.Vector3r(0, 0, 0)
    ],
                           3,
                           120,
                           airsim.DrivetrainType.MaxDegreeOfFreedom,
                           airsim.YawMode(False, 0),
                           20,
                           1,
                           vehicle_name=name)