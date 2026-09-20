"""Benchmark scope and metric definitions shared by runners and reports."""

PAPER_DATASETS = ('eth3d', '7scenes', 'scannetpp', 'hiroom')
DATASETS = (*PAPER_DATASETS, 'dtu')
SEEDS = (0, 1, 2)
SCENE_COUNTS = dict(eth3d=11, **{'7scenes': 7}, scannetpp=20, hiroom=30, dtu=22)
POSE_METRICS = ('auc01', 'auc03', 'auc30')
F1_METRICS = ('recon_unposed_fscore', 'recon_posed_fscore')
DTU_METRICS = tuple(f'{mode}_{key}' for mode in ('recon_unposed', 'recon_posed')
                    for key in ('acc', 'comp', 'overall'))


def metrics_for(name):
    return (*POSE_METRICS, *(DTU_METRICS if name == 'dtu' else F1_METRICS))


def metric_direction(key):
    return 'lower' if key in DTU_METRICS else 'higher'
