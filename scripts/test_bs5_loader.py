from scale_sensor_fusion_io import BS5Loader

if __name__ == "__main__":
    # scene_url = "/path/to/scene.bs5"
    ann_url = "/path/to/ann.bs5"
    # scene = BS5Loader(scene_url, skip_typecheck=True).load()
    # print(scene)

    annotations_scene = BS5Loader(ann_url).load()
    print(annotations_scene.annotations[:2])
