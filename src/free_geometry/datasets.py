"""Image-only discovery. GT readers are imported only by evaluation."""

from pathlib import Path

from .sampling import SceneSource

DATASETS = ("images", "eth3d", "hiroom", "7scenes", "scannetpp", "dtu", "dtu64")


def scene_source(dataset, root, scene):
    root = Path(root).resolve()
    folders = {
        "images": root,
        "eth3d": root / scene / "images",
        "hiroom": root / scene / "image",
        "7scenes": root
        / "7Scenes"
        / scene
        / ("seq-02" if scene == "stairs" else "seq-01"),
        "scannetpp": root / scene / "merge_dslr_iphone" / "images",
        "dtu": root / "Rectified" / scene,
        "dtu64": root / scene / "image",
    }
    if dataset not in folders:
        raise ValueError(f"unknown dataset {dataset}")
    source = SceneSource.from_directory(folders[dataset], scene)
    keep = lambda p: (
        ".color." in p
        if dataset == "7scenes"
        else "iphone" in Path(p).name
        if dataset == "scannetpp"
        else True
    )
    rows = [
        (fid, path)
        for fid, path in zip(source.frame_ids, source.image_files)
        if keep(path)
    ]
    source.frame_ids = [r[0] for r in rows]
    source.image_files = [r[1] for r in rows]
    source.dataset = dataset
    return source


def metric_dataset(name, root, gt_root=None):
    from importlib import import_module

    table = {
        "eth3d": ("eth3d", "ETH3D"),
        "hiroom": ("hiroom", "HiRoomDataset"),
        "7scenes": ("sevenscenes", "SevenScenes"),
        "scannetpp": ("scannetpp", "ScanNetPP"),
        "dtu": ("dtu", "DTU"),
        "dtu64": ("dtu64", "DTU64"),
    }
    if name not in table:
        raise ValueError("metrics require a supported benchmark dataset")
    module, cls = table[name]
    obj = getattr(import_module("depth_anything_3.bench.datasets." + module), cls)()
    obj.data_root = str(Path(root).resolve())
    if gt_root is not None:
        obj.gt_root_path = str(Path(gt_root).resolve())
    return obj
