import json
import importlib.util
import sys


def test_hard_manifest_preserves_student_positions():
    spec = importlib.util.spec_from_file_location("hard_views", "scripts/build_hard_view_subsets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scene = module.Scene("eth3d", "unit", ["a", "b", "c", "d", "e", "f", "g", "h"],
                         ["/tmp/a"] * 8, None, None, None, None)
    record = module.manifest_record(scene, list(range(8)), "eval", 0, None, "texture", [0, 4], .9, False)
    assert record["four_local_indices"] == [0, 2, 4, 6]
    assert record["four_frame_ids"] == ["a", "c", "e", "g"]
    assert json.loads(json.dumps(record))["hard_student_positions"] == [0, 4]
