"""
VGGT Dataset Registry.

Reuses DA3 dataset implementations since they are generic
(load images, poses, GT meshes).
"""

from typing import Any, Dict


class VGGTRegistry:
    """Simple registry for VGGT datasets."""

    def __init__(self):
        self._map: Dict[str, Any] = {}

    def register(self, name: str, cls: Any) -> None:
        """Register a dataset class with a name."""
        self._map[name] = cls

    def get(self, name: str) -> Any:
        """Get a dataset class by name."""
        return self._map[name]

    def has(self, name: str) -> bool:
        """Check if a dataset is registered."""
        return name in self._map

    def all(self) -> Dict[str, Any]:
        """Get all registered datasets."""
        return self._map


# Create VGGT-specific registry
VGGT_MV_REGISTRY = VGGTRegistry()


def _register_datasets():
    """Register all benchmark datasets for VGGT evaluation."""
    # Import DA3 dataset implementations (they are generic)
    from depth_anything_3.bench.datasets.eth3d import ETH3D
    from depth_anything_3.bench.datasets.sevenscenes import SevenScenes
    from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
    from depth_anything_3.bench.datasets.hiroom import HiRoomDataset
    from depth_anything_3.bench.datasets.dtu import DTU

    # Register datasets in VGGT registry
    VGGT_MV_REGISTRY.register("eth3d", ETH3D)
    VGGT_MV_REGISTRY.register("7scenes", SevenScenes)
    VGGT_MV_REGISTRY.register("scannetpp", ScanNetPP)
    VGGT_MV_REGISTRY.register("hiroom", HiRoomDataset)
    VGGT_MV_REGISTRY.register("dtu", DTU)


# Register datasets on module import
_register_datasets()
