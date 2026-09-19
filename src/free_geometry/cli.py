"""fg prepare / inspect / train / evaluate. Heavy imports are command-local."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .datasets import DATASETS, scene_source
from .sampling import build_manifest, inspect_manifest, validate_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fg", description="Unified per-scene Free-Geometry protocol"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="validate images and fix frame tasks without loading models"
    )
    prepare.add_argument("--dataset", choices=DATASETS, default="images")
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--scene", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config")
    inspect = sub.add_parser("inspect", help="audit manifest and frame coverage")
    inspect.add_argument("manifest")
    train = sub.add_parser(
        "train", help="adapt one scene; repeat over manifests for benchmark suites"
    )
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config")
    train.add_argument("--resume")
    evaluate = sub.add_parser(
        "evaluate",
        help="export frozen or adapted multiview predictions and optional GT metrics",
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--config")
    mode = evaluate.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint")
    mode.add_argument("--baseline", action="store_true")
    evaluate.add_argument(
        "--data-root", help="benchmark root; supplying it enables GT metrics"
    )
    evaluate.add_argument("--gt-root")
    evaluate.add_argument("--skip-reconstruction", action="store_true")
    for p in (prepare, train, evaluate):
        p.add_argument(
            "--set", action="append", default=[], metavar="SECTION.KEY=VALUE"
        )
    for p in (train, evaluate):
        p.add_argument("--model")
        p.add_argument("--weights")
        p.add_argument("--source")
        p.add_argument("--device")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(
            json.dumps(
                inspect_manifest(json.loads(Path(args.manifest).read_text())), indent=2
            )
        )
        return
    import yaml

    overrides = {}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator or "." not in key:
            parser.error("--set requires SECTION.KEY=VALUE")
        overrides[key] = yaml.safe_load(value)
    for key, dotted in [
        ("model", "model.name"),
        ("weights", "model.weights"),
        ("source", "model.source"),
        ("device", "train.device"),
    ]:
        if getattr(args, key, None) is not None:
            overrides[dotted] = getattr(args, key)
    config = load_config(args.config, overrides)
    if args.command == "prepare":
        manifest = build_manifest(
            scene_source(args.dataset, args.data_root, args.scene), config.sampling
        )
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(inspect_manifest(manifest), indent=2))
        return
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text()))
    if manifest["status"] == "skipped":
        print(json.dumps({"status": "skipped", "reason": manifest["reason"]}))
        return
    from .adapters import get_adapter

    adapter = get_adapter(config.model)
    if args.command == "train":
        from .trainer import train_scene

        result = train_scene(adapter, manifest, config, args.output, resume=args.resume)
    else:
        from .evaluation import benchmark_export, export_scene

        result = {
            "export": str(
                export_scene(adapter, manifest, config, args.output, args.checkpoint)
            )
        }
        if args.data_root:
            result["metrics"] = benchmark_export(
                manifest,
                args.output,
                args.data_root,
                args.gt_root,
                not args.skip_reconstruction,
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
