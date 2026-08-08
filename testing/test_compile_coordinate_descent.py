from toolkit.config_modules import ModelConfig


def test_coordinate_descent_defaults_off():
    config = ModelConfig(name_or_path="test")
    assert config.compile_coordinate_descent is False


def test_coordinate_descent_can_be_enabled_explicitly():
    config = ModelConfig(
        name_or_path="test",
        compile_coordinate_descent=True,
    )
    assert config.compile_coordinate_descent is True
