from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    llm_use_json_mode: bool = Field(default=False, alias="LLM_USE_JSON_MODE")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=700, alias="LLM_MAX_TOKENS")
    llm_timeout_s: int = Field(default=60, alias="LLM_TIMEOUT_S")
    llm_max_retries: int = Field(default=4, alias="LLM_MAX_RETRIES")

    eval_concurrency: int = Field(default=4, alias="EVAL_CONCURRENCY")

    runs_dir: str = Field(default="./runs", alias="RUNS_DIR")
    data_dir: str = Field(default="./data", alias="DATA_DIR")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
