import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "safetyvision-test.db"))
    with TestClient(main.app) as test_client:
        yield test_client


def image_bytes(size=(320, 240), color=(220, 220, 220)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG")
    return output.getvalue()


def successful_provider_result():
    return {
        "risks": [
            {
                "id": "risk-1",
                "title": "Obstáculo na passagem",
                "description": "Um objeto está sobre a área de circulação.",
                "category": "falls",
                "severity": "medium",
                "confidence": "high",
                "visual_evidence": "Objeto retangular visível sobre o piso na região central.",
                "location_description": "Região central",
                "bounding_box": {"x": 0.3, "y": 0.5, "width": 0.2, "height": 0.2},
                "related_nrs": ["NR-01"],
                "recommendation": "Remover o objeto da passagem.",
                "corrective_actions": ["Liberar a passagem."],
                "source": "visual",
            }
        ],
        "limitations": ["A imagem não mostra todo o ambiente."],
    }


def post_image(client, data=None, content=None, content_type="image/jpeg"):
    return client.post(
        "/analyze",
        data=data or {"environment": "industrial"},
        files={"image": ("foto.jpg", content or image_bytes(), content_type)},
    )


def test_analyze_returns_real_contract(client, monkeypatch):
    monkeypatch.setattr(main, "analyze_image", lambda *args: successful_provider_result())

    response = post_image(
        client,
        data={
            "environment": "industrial",
            "focuses": ["falls", "electric"],
            "notes": "Área próxima ao estoque",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["risk_count"] == 1
    assert body["focuses"] == ["falls", "electric"]
    assert body["reported_observations"] == ["Área próxima ao estoque"]
    assert body["risks"][0]["confidence"] == "high"


def test_zero_risks_is_valid(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "analyze_image",
        lambda *args: {"risks": [], "limitations": ["Imagem parcial."]},
    )

    response = post_image(client)

    assert response.status_code == 200
    assert response.json()["summary"]["risk_count"] == 0
    assert "Nenhum risco visual" in response.json()["summary"]["message"]


def test_missing_token_returns_503_without_fake_result(client, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    response = post_image(client)

    assert response.status_code == 503
    assert "configurada" in response.json()["detail"]


def test_rejects_invalid_environment(client):
    response = post_image(client, data={"environment": "qualquer-coisa"})

    assert response.status_code == 422


def test_rejects_non_image(client):
    response = post_image(client, content=b"arquivo de texto", content_type="text/plain")

    assert response.status_code == 415


def test_rejects_corrupted_image(client):
    response = post_image(client, content=b"not-a-jpeg")

    assert response.status_code == 400


def test_history_is_not_public(client):
    response = client.get("/history")

    assert response.status_code == 404
