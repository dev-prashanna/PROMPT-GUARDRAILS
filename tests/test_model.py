import pytest
from src.models.lora_config import LoRAConfigurator


class TestLoRAConfigurator:
    def _make_configurator(self, enabled=True):
        config = {
            "model": {
                "peft": {
                    "enabled": enabled,
                    "target_modules": ["query_proj", "value_proj"],
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "bias": "none",
                    "task_type": "SEQ_CLS",
                    "merge_after_training": True,
                }
            }
        }
        return LoRAConfigurator(config)

    def test_config_creation(self):
        configurator = self._make_configurator()
        assert configurator.rank == 16
        assert configurator.alpha == 32
        assert configurator.target_modules == ["query_proj", "value_proj"]

    def test_lora_config_generation(self):
        configurator = self._make_configurator()
        lora_config = configurator.get_lora_config()
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
