import boto3
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.schemas.schemas import PollyRequest

router = APIRouter(tags=["Voice"])

@router.post("/polly")
def text_to_speech(data: PollyRequest, current_user: User = Depends(get_current_user)):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío")

    # Limitar longitud para no exceder límites de Polly
    text = data.text[:3000]

    try:
        polly = boto3.client(
            "polly",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId=settings.POLLY_VOICE,
            LanguageCode=settings.POLLY_LANGUAGE
        )
        audio = response["AudioStream"].read()
        return StreamingResponse(BytesIO(audio), media_type="audio/mpeg")

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error en Amazon Polly: {str(e)}")
