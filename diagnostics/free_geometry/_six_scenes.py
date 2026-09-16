import json
print(" ".join(sorted(json.load(open("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json"))["scenes"])))
