"""Validate published Self-Geometry train/eval settings before formal runs."""

PAPER_VALUES = {
    'max_frames': 100, 'iterations': 50, 'lr': 5e-5, 'final_lr': 1e-8,
    'warmup_fraction': .05, 'weight_decay': .05, 'clip': 5.,
    'rank': 64, 'alpha': 64., 'dropout': .1,
    'fan': True, 'filter_each_step': True, 'gd': 'mvc_orthogonal_ec',
    'camera_lora': False, 'bin_width': 15, 'bins_compat9': False,
}


def paper_protocol(c, methods, skip_evaluation=False):
    expected = dict(PAPER_VALUES)
    # Resolution follows the retained DA3 benchmark adapter. Not specified in
    # the paper itself; do not present this as an independently published value.
    expected.update(image_size=504, ref_view_strategy='first')
    if 'test3r' in methods:
        expected.update(test3r_epochs=2, test3r_accum=4,
                        test3r_prompt_size=32, test3r_lr=1e-5,
                        test3r_max_updates=None, test3r_vggt_points='native')
        cap = c.get('test3r_max_triplets')
        if cap is not None:
            # User-requested additional comparison; not a published SG baseline.
            expected['test3r_max_triplets'] = 1000
    if 'tco' in methods:
        expected.update(tco_steps=None, tco_lr=None, tco_photo_weight=None,
                        tco_intrinsics_weight=0.)
    differences = {key: {'expected': value, 'actual': c.get(key)}
                   for key, value in expected.items() if c.get(key) != value}
    if skip_evaluation:
        differences['skip_evaluation'] = {'expected': False, 'actual': True}
    if differences:
        raise ValueError(f'Formal protocol mismatch: {differences}. '
                         'Use a separate custom experiment for variants/smoke tests.')
    return dict(
        reference='https://arxiv.org/html/2608.10708v2',
        profile='self_geometry_paper', checked_settings=expected,
        frame_sampling='retained DA3 Evaluator._sample_frames, seed 42',
        train_eval='same immutable per-scene RGB manifest; FAN subsets during Self-Geometry training',
        evaluation=['auc01', 'auc03', 'auc30', 'recon_unposed', 'recon_posed'],
        seed_meaning='independent adapter initialization/training RNG; fixed RGB sampling',
        extensions=['DTU with official DA3 distance metrics',
                    'Test3R port with user-requested 1000-triplet cap per epoch, two epochs'
                    if 'test3r' in methods and c.get('test3r_max_triplets') is not None
                    else 'Test3R port with its own exhaustive schedule'],
        limitations='Published settings plus documented implementation assumptions; '
                    'not proof of identical unreleased author code. See docs/method.md and docs/comparisons.md.',
    )
