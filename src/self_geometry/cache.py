"""Reuse immutable baseline/matching artifacts, never another seed's LoRA state."""
import errno
import json
import os
from pathlib import Path
import shutil
from .common import digest, write_json


BASELINE_KEYS = ('weights', 'data_root', 'max_frames', 'image_size', 'precision')


def copy_cached(src, dst):
    """Large immutable files use hard links when possible; metadata stays separate."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if src.suffix in ('.pt', '.npz', '.ply'):
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    shutil.copy2(src, dst)


def reuse(source, dest, c, completed=False):
    from .training import identity
    source, dest = Path(source), Path(dest)
    count = 0
    if source.resolve() == dest.resolve() or not source.exists():
        return count
    for path in source.rglob('manifest.json'):
        scene, target = path.parent, dest / path.parent.relative_to(source)
        protocol_path = scene / 'baseline/protocol.json'
        prediction = scene / 'baseline/exports/mini_npz/results.npz'
        # Do not change a destination with existing metadata or an active run.
        if (target / 'manifest.json').exists() or not (protocol_path.exists() and prediction.exists()):
            continue
        protocol = json.loads(protocol_path.read_text())
        old = protocol['config']
        if old.get('model','vggt') != c.get('model','vggt') or any(old[k] != c[k] for k in BASELINE_KEYS):
            continue
        manifest = json.loads(path.read_text())
        if protocol['identity'] != digest(identity(old, manifest)):
            continue
        # Input images and weights must still be the exact files in the record.
        if any(not Path(r['path']).is_file() or Path(r['path']).stat().st_size != r['bytes'] or
               Path(r['path']).stat().st_mtime_ns != r['mtime_ns'] for r in manifest['images']):
            continue
        for filename in ('manifest.json', 'gt_meta.npz', 'baseline/exports/mini_npz/results.npz', 'baseline/metrics.json'):
            src = scene / filename
            if src.exists():
                copy_cached(src, target / filename)
        if old['keypoints'] == c['keypoints'] and (scene / 'matches.pt').exists():
            copy_cached(scene / 'matches.pt', target / 'matches.pt')
        protocol.update(identity=digest(identity(c, manifest)), config=c, reused_from=str(protocol_path))
        write_json(target / 'baseline/protocol.json', protocol)
        done = scene / 'adapted/complete.json'
        # Only exact same config (including seed), completed training can migrate.
        if completed and old == c and done.exists():
            info = json.loads(done.read_text())
            if info['identity'] == digest(identity(c, manifest)):
                for src in (scene / 'adapted').rglob('*'):
                    if src.is_file() and not src.name.endswith(('.lock', '.tmp')):
                        copy_cached(src, target / src.relative_to(scene))
        count += 1
    return count
