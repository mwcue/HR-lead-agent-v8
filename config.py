# config.py
import os
from dotenv import load_dotenv
from utils.logging_utils import setup_logging

# Load environment variables
load_dotenv()

class Config:
    """Central configuration for the HR & Regional B2B Lead Generation system."""

    # --- API Keys ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    # --- LLM Settings ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # Default to openai, ensure lowercase

    # Specific model names per provider
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4-turbo")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL","anthropic/claude-3-5-haiku-20241022")     #  "claude-3-sonnet-20240229")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-flash") # Use flash as a reasonable default
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral/mistral-large-lates")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.2")


    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # --- Agent Settings ---
    RESEARCH_AGENT_MAX_ITER = int(os.getenv("RESEARCH_AGENT_MAX_ITER", "10"))
    ANALYSIS_AGENT_MAX_ITER = int(os.getenv("ANALYSIS_AGENT_MAX_ITER", "10")) # Reviewer uses this too

    # --- URLs Processing ---
    MAX_URLS_TO_PROCESS = int(os.getenv("MAX_URLS_TO_PROCESS", "10")) # Keep at 10 as decided

    # --- Output Settings ---
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", os.path.join(os.path.dirname(__file__), "output.csv"))

    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Ensure uppercase for logging levels

    # --- API Rate Limits (Optional - not directly used by factory but good practice) ---
    OPENAI_RATE_LIMIT_RETRY = int(os.getenv("OPENAI_RATE_LIMIT_RETRY", "3"))
    SERPER_RATE_LIMIT_RETRY = int(os.getenv("SERPER_RATE_LIMIT_RETRY", "3"))

    # --- Request Retry Settings (Optional - relevant for scraper/requests) ---
    API_RETRY_DELAY = int(os.getenv("API_RETRY_DELAY", "2"))
    API_RETRY_BACKOFF = int(os.getenv("API_RETRY_BACKOFF", "2"))

    # --- Scraper Settings ---
    SCRAPER_REQUEST_TIMEOUT = int(os.getenv("SCRAPER_REQUEST_TIMEOUT", "20"))

    # --- Company Filtering ---
    GENERIC_COMPANY_NAMES = [
        'company', 'organization', 'the firm', 'client',
        'example', 'test', 'none', 'n/a', 'website', 'url'
    ]

    @classmethod
    def validate(cls):
        """Validate critical configuration settings based on chosen provider."""
        missing_keys = []

        supported_providers_requiring_keys = ["openai", "anthropic", "google", "mistralai"]
        all_supported_providers = supported_providers_requiring_keys + ["ollama"]

        # Always check Serper
        if not cls.SERPER_API_KEY:
            missing_keys.append("SERPER_API_KEY")

        # Check provider-specific key if required
        provider = cls.LLM_PROVIDER # Read the provider set in config (from .env)

        # Check provider-specific key
        if provider == "openai" and not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY (for selected provider 'openai')")
        elif provider == "anthropic" and not cls.ANTHROPIC_API_KEY:
            missing_keys.append("ANTHROPIC_API_KEY (for selected provider 'anthropic')")
        elif provider == "google" and not cls.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY (for selected provider 'google')")
        elif provider == "mistralai" and not cls.MISTRAL_API_KEY:
            missing_keys.append("MISTRAL_API_KEY (for provider '{provider}')")
        elif provider == "ollama":
            pass # no api key needed for ollama
        elif cls.LLM_PROVIDER not in all_supported_providers:
             missing_keys.append(f"LLM_PROVIDER '{provider}' is not recognized/supported by config validation. Supported: {all_supported_providers}")

        return missing_keys

    @classmethod
    def configure_logging(cls):
        """Configure logging based on settings."""
        # Moved setup_logging import here to avoid potential circular dependency if utils imports config
        from utils.logging_utils import setup_logging
        setup_logging(cls.LOG_LEVEL)
