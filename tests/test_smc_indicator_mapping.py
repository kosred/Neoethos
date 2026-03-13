from forex_bot.strategy import discovery_tensor, evo_prop


def test_evo_prop_feature_to_indicator_supports_smc_names():
    assert evo_prop._feature_to_indicator("smc_bos", {"RSI"}) == "SMC_BOS"
    assert evo_prop._feature_to_indicator("SMC_CHOCH", set()) == "SMC_CHOCH"


def test_discovery_tensor_feature_to_indicator_supports_smc_names():
    assert discovery_tensor._feature_to_indicator("smc_eqh", {"EMA"}) == "SMC_EQH"
    assert discovery_tensor._feature_to_indicator("SMC_DISPLACEMENT", set()) == "SMC_DISPLACEMENT"
