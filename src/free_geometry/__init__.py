"""Free-Geometry unified TTA core: model-agnostic adapters + losses + trainer.

Protocol (2026-09-17 migration): frozen long-context teacher (8 clean views)
supervises a short-context student (4 shared views, 50% block-masked input).
Distillation loss is computed on ALL patch positions (loss_all_pos, per the
2026-09-16 ablation) in each model's native readout space, teacher-confidence
weighted; geometry terms operate on camera CENTERS only (gauge-free):
maskdistill + 1.5*rkd_centers + 1.0*couple_centers (arm "rkdc_allpos"),
or pure maskdistill (arm "m_allpos").
"""
