from configs.training_configs import get_config


def test_gate_on_generalist_scope_is_exactly_the_supported_v8_subset():
    config = get_config("trench_align_generalist_v1")
    paths = [level.maps_path for level in config.maps]

    assert len(paths) == 37
    assert len(set(paths)) == 37
    assert config.enforce_trench_dig_alignment is True
    assert config.require_trench_alignment_metadata is True
    assert config.trench_alignment_observation is True
    assert config.pooled_sampler.rule == "continuous_banded_v3"
    assert all(level.max_steps_in_episode == 450 for level in config.maps)
    assert all(level.rewards_type == "DENSE" for level in config.maps)
    assert all(not level.apply_trench_rewards for level in config.maps)
    assert not any("trn-net4" in path for path in paths)
    assert not any("v7-trn" in path for path in paths)
