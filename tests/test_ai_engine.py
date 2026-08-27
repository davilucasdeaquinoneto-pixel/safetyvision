import json

import pytest

from ai_engine import VisionResponseError, normalize_provider_response


def risk_payload(**overrides):
    payload = {
        "title": "Cabo atravessando a passagem",
        "description": "Um cabo está sobre a área utilizada para circulação.",
        "category": "falls",
        "severity": "medium",
        "confidence": "high",
        "visual_evidence": "Um cabo preto cruza o piso na região inferior central.",
        "location_description": "Parte inferior central",
        "bounding_box": {"x": 0.2, "y": 0.7, "width": 0.5, "height": 0.15},
        "related_nrs": ["NR 01", "texto inválido"],
        "recommendation": "Retirar ou proteger o cabo.",
        "corrective_actions": ["Reposicionar o cabo.", "Instalar uma proteção adequada."],
    }
    payload.update(overrides)
    return payload


def test_accepts_valid_risk_and_normalizes_nr():
    result = normalize_provider_response(json.dumps({"risks": [risk_payload()], "limitations": []}))

    assert len(result["risks"]) == 1
    assert result["risks"][0]["related_nrs"] == ["NR-01"]
    assert result["risks"][0]["confidence"] == "high"


def test_filters_low_confidence_risk():
    result = normalize_provider_response(
        json.dumps({"risks": [risk_payload(confidence="low")], "limitations": []})
    )

    assert result["risks"] == []


def test_filters_ambiguous_evidence_even_with_high_confidence():
    result = normalize_provider_response(
        json.dumps(
            {
                "risks": [
                    risk_payload(
                        confidence="high",
                        visual_evidence="Pode haver um risco elétrico no ambiente.",
                    )
                ],
                "limitations": [],
            }
        )
    )

    assert result["risks"] == []


def test_invalid_bounding_box_becomes_null():
    result = normalize_provider_response(
        json.dumps(
            {
                "risks": [
                    risk_payload(
                        bounding_box={"x": 0.9, "y": 0.2, "width": 0.5, "height": 0.2}
                    )
                ],
                "limitations": [],
            }
        )
    )

    assert result["risks"][0]["bounding_box"] is None


def test_rejects_non_json_response():
    with pytest.raises(VisionResponseError):
        normalize_provider_response("Não encontrei riscos.")


def test_allows_zero_risks():
    result = normalize_provider_response('{"risks": [], "limitations": ["Imagem parcial"]}')

    assert result["risks"] == []
    assert "Imagem parcial" in result["limitations"]
