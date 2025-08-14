# sensor-fusion-io

SDK for working with sensor fusion scenes

## Deployment

Bumping the version at packages/sensor-fusion-io/py/pyproject.toml is required for the package to be published by CircleCI workflow. Otherwise, the publishing step will be successful but the upload itself will be skipped:

```
Publishing scale_sensor_fusion_io (0.4.8) to scale-pypi
 - Uploading scale_sensor_fusion_io-0.4.8.tar.gz File exists. Skipping
```

### Publishing to scale-pypi

Handled by CICD

### Publishing to pypi

1. Create an account on [pypi](https://pypi.org/account/register/) 
2. Ask to be added to this package project.
3. Create a token (see [here](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#create-an-account) for more details)
4. Add the token info to $HOME/.pypirc
```
[pypi]
username = __token__
password = <actual token without quotes>
```

5. Install dependencies:
```
python3 -m pip install twine
python3 -m pip install build
```

6. Build dist:
```
python3 -m build --sdist
```

7. Upload to pypi:
```
twine upload dist/*
```

# FAQ

## Resulting scene file is too large

For scenes that span a large timeframe, the size of the resulting .sfs file may increase to multi-GBs. This is not ideal for loading onto LidarLite.

### Video encoding

One easy way to reduce scene size is to encode camera content as video, as the video content can be more easily compressed. The tradeoff is the potentially reduced quality of images, but for labeling 3D scenes, this is often sufficient.

See utils/video_helpers/ for helper functions

### Downsample point clouds

Another option is to downsample lidar point clouds. If your scene is used primarily for cuboid annotation, we recommend voxel downsampling using voxel sizes of at most 20mm.

A good heuristic for efficient loading and labeling is to have a scene contain no more than 100,000 points.
