"""Dataset macro means followed by across-seed mean and sample standard deviation."""
import numpy as np
from .benchmark import metrics_for, metric_direction


def aggregate_seeds(summaries, names, seeds):
    result = {}
    for name in names:
        dataset_result = {}
        for stage in ('baseline', 'adapted'):
            entries = {str(seed): summaries[str(seed)]['datasets'][name][stage] for seed in seeds}
            complete = bool(seeds) and all(row['complete'] for row in entries.values())
            metrics = {}
            if complete:
                for key in metrics_for(name):
                    values = [entries[str(seed)]['means'][key] for seed in seeds]
                    metrics[key] = dict(mean=float(np.mean(values)),
                                        sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                        direction=metric_direction(key))
            dataset_result[stage] = dict(complete=complete, per_seed=entries, metrics=metrics)
        paired = {}
        if all(dataset_result[stage]['complete'] for stage in ('baseline', 'adapted')):
            for key in metrics_for(name):
                deltas = [dataset_result['adapted']['per_seed'][str(s)]['means'][key] -
                          dataset_result['baseline']['per_seed'][str(s)]['means'][key] for s in seeds]
                paired[key] = dict(mean=float(np.mean(deltas)),
                                   sample_std=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None)
        dataset_result['paired_adapted_minus_baseline'] = paired
        result[name] = dataset_result
    return result
