import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

import llm  # noqa: E402  (nach load_dotenv, damit OPENROUTER_MODEL gelesen wird)
from prompts import Direction, Intensity  # noqa: E402

MAX_TEXT_LENGTH = 500
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Bullshit-Übersetzer API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class TranslateRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    direction: Direction
    intensity: Intensity = "medium"

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text darf nicht leer sein.")
        return value


class TranslateResponse(BaseModel):
    result: str
    direction: Direction
    tokens: int | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": llm.MODEL, "rate_limit": RATE_LIMIT}


@app.post("/api/translate", response_model=TranslateResponse)
@limiter.limit(RATE_LIMIT)
def translate(request: Request, body: TranslateRequest) -> TranslateResponse:
    try:
        result, tokens = llm.translate(body.text, body.direction, body.intensity)
    except llm.LLMTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return TranslateResponse(result=result, direction=body.direction, tokens=tokens)
