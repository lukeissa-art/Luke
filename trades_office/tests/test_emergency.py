from app import emergency


def test_gas_is_life_safety():
    m = emergency.check("I think I smell gas in the kitchen")
    assert m.level == "life_safety" and "leave the building" in m.caller_instructions


def test_carbon_monoxide():
    assert emergency.check("my carbon monoxide alarm keeps going off").level == "life_safety"


def test_flooding_is_urgent():
    assert emergency.check("the basement is flooding").level == "urgent"


def test_shop_specific_keyword():
    assert emergency.check("the walk-in freezer died", "walk-in freezer").level == "urgent"


def test_routine_call_is_not_emergency():
    assert emergency.check("my kitchen faucet drips a little") is None
