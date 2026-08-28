from fastapi import HTTPException

from morphcloud.serving.server import STATE, GenerateRequest, generate, health


def test_health_does_not_claim_generation_ready():
    result = health()
    assert result["generation_ready"] is False


def test_generate_fails_explicitly_without_data_plane():
    try:
        generate(GenerateRequest(prompt="hello"))
    except HTTPException as exc:
        assert exc.status_code == 501
    else:
        raise AssertionError("generate() must not pretend the missing data plane is available")
