from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-2"
    S3_BUCKET_NAME: str = "nova-backend-pdfs"

    POLLY_VOICE: str = "Lupe"
    POLLY_LANGUAGE: str = "es-US"

    class Config:
        env_file = ".env"

settings = Settings()
