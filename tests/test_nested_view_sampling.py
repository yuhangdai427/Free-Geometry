from vggt.test_time_adaption.nested_view_sampling import make_records, nested_subset


def test_nested_pools_are_reproducible_and_nested():
    frame_counts = {"scene_a": 80, "scene_b": 33}
    first = make_records(frame_counts, seed=101, samples_per_scene=4)
    assert first == make_records(frame_counts, seed=101, samples_per_scene=4)

    for record in first:
        p16 = record["frame_indices"]
        s8 = nested_subset(p16, 8)
        s4 = nested_subset(p16, 4)
        s2 = s4[::2]
        assert [len(values) for values in (s2, s4, s8, p16)] == [2, 4, 8, 16]
        assert set(s2) < set(s4) < set(s8) < set(p16)
        assert p16[0] == s8[0] == s4[0] == s2[0] == 0
