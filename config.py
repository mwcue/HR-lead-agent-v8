# config.py

import os
from dotenv import load_dotenv
# Moved setup_logging import inside configure_logging to avoid potential circular dependency

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
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3") # Default to llama3, change as needed
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

    # ADDED: Ollama base URL
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # --- Agent Settings ---
    RESEARCH_AGENT_MAX_ITER = int(os.getenv("RESEARCH_AGENT_MAX_ITER", "10"))
    ANALYSIS_AGENT_MAX_ITER = int(os.getenv("ANALYSIS_AGENT_MAX_ITER", "10"))

    # --- URLs Processing ---
    MAX_URLS_TO_PROCESS = int(os.getenv("MAX_URLS_TO_PROCESS", "10"))

    # --- Output Settings ---
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", os.path.join(os.path.dirname(__file__), "output.csv"))

    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- API Rate Limits (Optional) ---
    OPENAI_RATE_LIMIT_RETRY = int(os.getenv("OPENAI_RATE_LIMIT_RETRY", "3"))
    SERPER_RATE_LIMIT_RETRY = int(os.getenv("SERPER_RATE_LIMIT_RETRY", "3"))

    # --- Request Retry Settings (Optional) ---
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
        supported_providers = ["openai", "anthropic", "google", "ollama", "mistralai"] # Update list

        # Always check Serper
        if not cls.SERPER_API_KEY:
            missing_keys.append("SERPER_API_KEY")

        # Check provider-specific key only if required
        provider = cls.LLM_PROVIDER
        if provider == "openai" and not cls.OPENAI_API_KEY:
            missing_keys.append(f"OPENAI_API_KEY (for selected provider '{provider}')")
        elif provider == "anthropic" and not cls.ANTHROPIC_API_KEY:
            missing_keys.append(f"ANTHROPIC_API_KEY (for selected provider '{provider}')")
        elif provider == "google" and not cls.GOOGLE_API_KEY:
            missing_keys.append(f"GOOGLE_API_KEY (for selected provider '{provider}')")
        elif provider == "mistralai" and not cls.MISTRAL_API_KEY:
            missing_keys.append(f"MISTRAL_API_KEY (for selected provider '{provider}')")
        elif provider == "ollama":
            pass # No API key needed for Ollama
        elif provider not in supported_providers:
             missing_keys.append(f"LLM_PROVIDER '{provider}' is not recognized/supported by config validation.")

        return missing_keys

    @classmethod
    def configure_logging(cls):
        """Configure logging based on settings."""
        from utils.logging_utils import setup_logging # Import here
        setup_logging(cls.LOG_LEVEL)
