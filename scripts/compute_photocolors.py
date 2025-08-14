from typing import List
from dataclasses import asdict
import numpy as np
from turbojpeg import TurboJPEG

from scale_json_binary import JSONBinaryEncoder
from scale_sensor_fusion_io.loaders.sfs_loader import SFSLoader
from scale_sensor_fusion_io.loaders.bs5_loader import BS5Loader
from scale_sensor_fusion_io.model_converters.sfs import to_scene_spec_sfs
from scale_sensor_fusion_io.utils.video_helpers import VideoWriter
from scale_sensor_fusion_io.models.sensors import Sensor, CameraSensor
from scale_sensor_fusion_io.models import SensorKind
from scale_sensor_fusion_io.utils.photocolors import compute_photocolors


DATA_DIR = "/path/to/data/scenes"

if __name__ == "__main__":
    scene_url = f"{DATA_DIR}/scene.debug.calibrated.2.bs5"
    output_url = f"{DATA_DIR}/scene.debug.calibrated.photocolors.2.bs5"
    scene = SFSLoader(scene_url).load()
    print("Total sensors", len(scene.sensors))

    print([sensor.type for sensor in scene.sensors])

    camera_sensors = [
        sensor for sensor in scene.sensors if sensor.type is SensorKind.Camera
    ]

    # for sensor in camera_sensors:
    #     with open(f"{DATA_DIR}/foal/sensor-{sensor.id}.mp4", "wb") as fd:
    #         fd.write(sensor.video.content.tobytes())

    print("Num camera sensors: ", len(camera_sensors))

    lidar_sensors = [
        sensor for sensor in scene.sensors if sensor.type is SensorKind.Lidar
    ]

    print("Num lidar sensors: ", len(lidar_sensors))
    for sensor in lidar_sensors:
        for frame_num, frame in enumerate(sensor.frames):
            print(f"Lidar sensor {sensor.id}: frame {frame_num}")
            frame.points.colors = compute_photocolors(
                frame.points.positions,
                camera_sensors,
                frame.points.timestamps,
                frame.timestamp,
            )

    scene_sfs = to_scene_spec_sfs(scene)

    encoder = JSONBinaryEncoder().write_file(output_url, asdict(scene_sfs))
