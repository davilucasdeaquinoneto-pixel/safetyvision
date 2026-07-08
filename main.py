from database import create_database, save_analysis, get_analyses, get_analysis
from ai_engine import analyze_image
from PIL import Image
import io
from fastapi import UploadFile, File, Form, FastAPI

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_database()


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "SafetyVision API funcionando 🚀"
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    environment: str = Form("geral"),
    notes: str = Form("")
):
    image_bytes = await image.read()

    img = Image.open(io.BytesIO(image_bytes))

    ai_result = analyze_image(
        image_bytes,
        environment,
        notes
    )

    save_analysis(
        image.filename,
        environment,
        ai_result["analysis"]
    )

    return {
        "status": "success",
        "filename": image.filename,
        "image": {
            "format": img.format,
            "width": img.width,
            "height": img.height,
            "mode": img.mode
        },
        "environment": environment,
        "notes": notes,
        "ai_analysis": ai_result
    }


@app.get("/history")
def history():
    return {
        "analyses": get_analyses()
    }

@app.get("/history/{id}")
def history_id(id: int):

    analysis = get_analysis(id)

    if not analysis:
        return {
            "error": "Análise não encontrada"
        }

    return analysis