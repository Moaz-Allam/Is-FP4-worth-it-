import time_per_token as model


def test_memory_floor_is_bytes_over_bandwidth():
    assert model.memory_floor_seconds(1_000_000, 4.0, 2_000_000.0) == 2.0


def test_decode_time_is_weights_times_time_per_weight():
    assert model.decode_seconds(1_000_000, 2e-9) == 0.002


def test_prediction_takes_the_larger_term_and_names_it():
    machine = model.Machine("toy", 1_000_000.0, ())
    slow_decode = model.prediction(machine, model.Kernel("a", 0.5, 1e-3), "m", 1000)
    slow_memory = model.prediction(machine, model.Kernel("b", 4.0, 1e-9), "m", 1000)
    assert (slow_decode["seconds_per_token"], slow_decode["bound_by"]) == (1.0, "decode")
    assert (slow_memory["seconds_per_token"], slow_memory["bound_by"]) == (0.004, "memory")
