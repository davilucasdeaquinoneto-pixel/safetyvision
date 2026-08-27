from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from ai_engine import (
    DEFAULT_MODEL,
    VisionConfigurationError,
    VisionProviderError,
    VisionResponseError,
    analyze_image,
)
from database import create_database, save_analysis
from models import AnalysisResponse, AnalysisSummary, ImageInfo, RiskItem


ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_ENVIRONMENTS = {
    "industrial",
    "construction",
    "office",
    "warehouse",
    "health",
    "kitchen",
    "general",
}
ALLOWED_FOCUSES = {
    "falls",
    "electric",
    "fire",
    "ppe",
    "ergonomic",
    "circulation",
    "machines",
}
FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_NOTES_LENGTH = 1000
MAX_IMAGE_DIMENSION = 1800
Image.MAX_IMAGE_PIXELS = 25_000_000


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_database()
    yield


app = FastAPI(
    title="SafetyVision API",
    version="2.1.0",
    description="Triagem visual conservadora de riscos ocupacionais.",
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    default = (
        "https://site-projeto-integrador.onrender.com,"
        "http://localhost:10000,http://127.0.0.1:10000"
    )
    return [
        origin.strip().rstrip("/")
        for origin in os.getenv("FRONTEND_ORIGINS", default).split(",")
        if origin.strip()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def home() -> dict:
    return {
        "status": "online",
        "message": "SafetyVision API funcionando.",
        "ai_configured": bool(os.getenv("HF_TOKEN", "").strip()),
        "model": os.getenv("VISION_MODEL", DEFAULT_MODEL),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "ai_configured": bool(os.getenv("HF_TOKEN", "").strip())}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(...),
    environment: str = Form("general"),
    focuses: list[str] = Form(default=[]),
    notes: str = Form(""),
) -> AnalysisResponse:
    normalized_environment = environment.strip().casefold()
    if normalized_environment not in ALLOWED_ENVIRONMENTS:
        raise HTTPException(status_code=422, detail="Tipo de ambiente inválido.")

    normalized_focuses = list(dict.fromkeys(item.strip().casefold() for item in focuses))
    invalid_focuses = [item for item in normalized_focuses if item not in ALLOWED_FOCUSES]
    if invalid_focuses:
        raise HTTPException(status_code=422, detail="Um ou mais focos de análise são inválidos.")

    clean_notes = notes.strip()
    if len(clean_notes) > MAX_NOTES_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"A observação deve ter no máximo {MAX_NOTES_LENGTH} caracteres.",
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="A imagem deve ter no máximo 10 MB.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="O arquivo de imagem está vazio.")

    processed_bytes, processed_mime, image_info = await run_in_threadpool(
        _validate_and_prepare_image,
        image_bytes,
        image.content_type or "",
    )

    try:
        provider_result = await run_in_threadpool(
            analyze_image,
            processed_bytes,
            processed_mime,
            normalized_environment,
            normalized_focuses,
            clean_notes,
        )
    except VisionConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VisionProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except VisionResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    risks = [RiskItem.model_validate(item) for item in provider_result["risks"]]
    summary = _build_summary(risks)
    filename = Path(image.filename or "imagem").name[:180]

    response = AnalysisResponse(
        filename=filename,
        environment=normalized_environment,
        focuses=normalized_focuses,
        image=image_info,
        summary=summary,
        limitations=provider_result["limitations"],
        risks=risks,
        reported_observations=[clean_notes] if clean_notes else [],
        ai_analysis=_legacy_ai_analysis(risks),
    )

    stored = response.model_dump(mode="json")
    analysis_id = await run_in_threadpool(
        save_analysis,
        filename,
        normalized_environment,
        stored,
        "huggingface",
    )
    response.analysis_id = analysis_id
    return response


def _legacy_ai_analysis(risks: list[RiskItem]) -> dict[str, object]:
    severity_map = {
        "high": "Alta",
        "medium": "Media",
        "low": "Baixa",
    }
    highest = next(
        (level for level in ("high", "medium", "low") if any(risk.severity == level for risk in risks)),
        "low",
    )

    return {
        "analysis": {
            "risks": [risk.title for risk in risks],
            "risk_level": severity_map[highest],
            "recommendations": [risk.recommendation for risk in risks],
        }
    }


def _validate_and_prepare_image(
    image_bytes: bytes,
    declared_mime: str,
) -> tuple[bytes, str, ImageInfo]:
    if declared_mime and declared_mime.casefold() not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Envie uma imagem JPEG, PNG ou WebP.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as verification_image:
            verification_image.verify()

        with Image.open(io.BytesIO(image_bytes)) as source:
            detected_format = (source.format or "").upper()
            detected_mime = FORMAT_TO_MIME.get(detected_format)
            if detected_mime is None:
                raise HTTPException(status_code=415, detail="Formato de imagem não suportado.")

            width, height = source.size
            if width < 64 or height < 64:
                raise HTTPException(status_code=422, detail="A imagem é pequena demais para análise.")
            if width * height > Image.MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="A imagem possui dimensões excessivas.")

            prepared = ImageOps.exif_transpose(source).convert("RGB")
            prepared.thumbnail(
                (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
                Image.Resampling.LANCZOS,
            )

            output = io.BytesIO()
            prepared.save(output, format="JPEG", quality=88, optimize=True)
            processed = output.getvalue()
            info = ImageInfo(format=detected_format, width=width, height=height)
            return processed, "image/jpeg", info
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(status_code=400, detail="A imagem está corrompida ou é inválida.") from exc
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="A imagem possui dimensões excessivas.") from exc


def _build_summary(risks: list[RiskItem]) -> AnalysisSummary:
    counts = {
        "high": sum(risk.severity == "high" for risk in risks),
        "medium": sum(risk.severity == "medium" for risk in risks),
        "low": sum(risk.severity == "low" for risk in risks),
    }
    highest = next((level for level in ("high", "medium", "low") if counts[level]), None)

    if not risks:
        message = "Nenhum risco visual foi confirmado com evidência suficiente nesta fotografia."
    elif len(risks) == 1:
        message = "Foi encontrado 1 risco visual com evidência suficiente."
    else:
        message = f"Foram encontrados {len(risks)} riscos visuais com evidência suficiente."

    return AnalysisSummary(
        risk_count=len(risks),
        high_count=counts["high"],
        medium_count=counts["medium"],
        low_count=counts["low"],
        highest_severity=highest,
        message=message,
    )
