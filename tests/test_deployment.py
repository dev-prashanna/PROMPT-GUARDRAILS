import pytest
from src.deployment.gates import Gate, GateResult, ThreeGateArchitecture


class TestGateResult:
    def test_creation(self):
        result = GateResult(
            gate_name="Test Gate",
            label="safe",
            confidence=0.95,
            blocked=False,
            latency_ms=10.5,
        )
        assert result.gate_name == "Test Gate"
        assert result.label == "safe"
        assert result.confidence == 0.95
        assert result.blocked is False
        assert result.latency_ms == 10.5

    def test_blocked_result(self):
        result = GateResult(
            gate_name="Input Scanner",
            label="jailbreak",
            confidence=0.92,
            blocked=True,
            latency_ms=45.0,
            details={"all_scores": {"jailbreak": 0.92, "safe": 0.08}},
        )
        assert result.blocked is True
        assert result.confidence >= 0.9
