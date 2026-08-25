import sae_trainer_rolling as t


def _history(evs, l0s=None, deads=None):
    l0s = l0s or [50.0] * len(evs)
    deads = deads or [0.0] * len(evs)
    return list(zip(evs, l0s, deads))


def test_plateau_ready_when_ev_stalls_at_target_without_feature_death():
    history = _history([0.854, 0.855, 0.857, 0.857, 0.859, 0.858])
    ready, detail = t._ev_plateau_ready(history, target_l0=50, best_ev=0.859)

    assert ready
    assert detail["gain"] <= 0.005


def test_plateau_rejects_ev_that_is_still_improving():
    history = _history([0.82, 0.83, 0.84, 0.85, 0.86, 0.87])
    ready, _ = t._ev_plateau_ready(history, target_l0=50, best_ev=0.87)
    assert not ready


def test_plateau_rejects_unsettled_l0():
    history = _history(
        [0.854, 0.855, 0.857, 0.857, 0.859, 0.858],
        l0s=[50, 51, 64, 52, 50, 49],
    )
    ready, _ = t._ev_plateau_ready(history, target_l0=50, best_ev=0.859)
    assert not ready


def test_plateau_rejects_feature_death():
    history = _history(
        [0.854, 0.855, 0.857, 0.857, 0.859, 0.858],
        deads=[0, 0, 0, 1.2, 0, 0],
    )
    ready, _ = t._ev_plateau_ready(history, target_l0=50, best_ev=0.859)
    assert not ready


def test_plateau_rejects_state_below_earlier_best():
    history = _history([0.844, 0.845, 0.846, 0.846, 0.847, 0.846])
    ready, _ = t._ev_plateau_ready(history, target_l0=50, best_ev=0.870)
    assert not ready
