from functools import lru_cache
from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables and .env files.
    """

    app_name: str = "re-take"
    debug: bool = False

    # STT API Keys
    assemblyai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ASSEMBLYAI_API_KEY", "assemblyai_api_key"),
    )
    sarvam_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "SARVAM_API_KEY",
            "SARVAMAI_API_KEY",
            "SARVAM_AI_API_KEY",
            "sarvam_api_key",
        ),
    )

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of application settings."""
    return Settings()


# Singleton instance for easy import across modules
settings = get_settings()
