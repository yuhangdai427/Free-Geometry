# Self-Geometry train/eval alignment

The user confirmed that the reference is **Self-Geometry, arXiv:2608.10708v2**.
The unrelated local SelfEvo train/eval recipe is not used.

The earlier 3-frame / 2-update runs are execution smoke tests. Their AUC/F1
values do not establish performance under the formal paper protocol.

## Formal settings

| Item | Applied setting | Source |
|---|---|---|
| Scene input | Up to 100 frames; use all if fewer | Paper IV-A |
| Ordered RGB subset | Original retained DA3 sampler, seed 42 | DA3 benchmark implementation |
| Self-Geometry training | 50 iterations, AdamW, lr 5e-5 → 1e-8, cosine, 5% warmup, wd .05, clip 5 | Appendix S.1 |
| LoRA | QKV, rank/alpha 64, dropout .1 | Appendix S.V |
| Training views | FAN subsets of the same scene manifest | Paper III-C / Algorithm 1 |
| Checkpoint | Minimum robust geometric score; full-scene final prediction | Algorithm 1; exact pre-update state convention documented in method.md |
| Pose metrics | AUC@1, AUC@3, AUC@30 | Paper IV-A |
| Reconstruction | Both posed and unposed F1 | Paper IV-A |
| Resolution / reference | Model-native DA3 benchmark preprocessing, 504 long side / first view | Retained local benchmark adapter, not an explicit paper hyperparameter |
| Three seeds | Independent training RNG 0/1/2; same sampled RGB frames | Requested repetition extension |
| DTU | Official DA3 22 scenes, normally 49 views, distance metrics in mm | Requested extension; paper excludes DTU |
| Test3R | Shared scene inputs and evaluator; user-requested 50 optimizer updates; 4 triplets per update; sample pool capped at 1000 | Additional comparison, not a baseline evaluated in this paper |

TCO uses the pinned author loss and documented dataset settings with frozen
predicted camera priors, following Self-Geometry IV-A. Unpublished choices and
architecture mappings remain listed in `method.md` and `comparisons.md`;
passing a configuration guard does not prove equivalence to unreleased author
code. Runtime caching, pair batching, fused optimizers and activation
checkpoint choices do not shorten method training schedules.

## Commands and checks

```bash
# Formal full matrix: two models, four methods, five datasets, three seeds.
bash scripts/run_paper_comparison.sh --root artifacts/paper_comparison_50updates

# Initial paper-length validation: one fixed scene per dataset, seed 0.
# This covers baseline + Self-Geometry + TCO; it is not the full matrix.
bash scripts/run_paper_comparison.sh --root artifacts/paper_protocol_first \
  --methods baseline self_geometry tco --seeds 0 --first-only

# Actual selected-scene AUC/F1 table and separately labeled dataset coverage.
.venv/bin/python scripts/summarize_comparison.py --root artifacts/paper_protocol_first

# Verify actual ordered input/eval manifests, completion identities and metrics.
.venv/bin/python scripts/audit_paper_run.py --root artifacts/paper_protocol_first
# Add --require-complete to fail if any evaluation is still pending.
```

The formal launcher rejects shortened frames/Self-Geometry iterations, unsupported Test3R update budgets,
overridden TCO schedules, and `--skip-evaluation`. Baseline and adapted methods
must evaluate the exact same ordered scene frames. The audit reads actual
manifests and final exports, not just configuration labels. Results are in
`SELECTED_RESULTS.md`; missing evaluation is shown as pending rather than zero.

Formal first scenes: ETH3D courtyard 38 frames, 7Scenes chess 100, ScanNet++
09c1414f1b 100, HiRoom 20241230/828738/cam_sampled_08 23, DTU scan1 49.
These are single-scene validations, never whole-dataset averages.

The user changed Test3R from the earlier 1000-triplet/two-epoch variant (500
updates at 100 views) to a 50-update budget matching Self-Geometry's update
count. Accumulation remains 4: current validation scenes consume 200 triplets
from the seed-fixed sample of at most 1000, then stop. The retained epochs=2
parameter is an upper bound, not a requirement to keep training beyond 50
updates. Different losses and training views mean equal update counts do not
imply equal FLOPs. Old checkpoints beyond 50 updates are not migrated.

The profile and reports explicitly label this user-requested extra Test3R
comparison. SG settings and actual train/eval frames remain unchanged.
Set both `test3r_max_updates=null` and `test3r_max_triplets=null`, with a separate
root, for the exhaustive variant. Existing historical profiles remain auditable.
