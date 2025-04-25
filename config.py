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
    # ADDED: Keys for other providers
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # --- LLM Settings ---
    # ADDED: LLM_PROVIDER setting
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # Default to openai, ensure lowercase

    # Specific model names per provider
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    # ADDED: Models for other providers
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") # Use flash as a reasonable default

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

        # Always check Serper
        if not cls.SERPER_API_KEY:
            missing_keys.append("SERPER_API_KEY")

        # Check provider-specific key
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY (for selected provider 'openai')")
        elif cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            missing_keys.append("ANTHROPIC_API_KEY (for selected provider 'anthropic')")
        elif cls.LLM_PROVIDER == "google" and not cls.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY (for selected provider 'google')")
        elif cls.LLM_PROVIDER not in ["openai", "anthropic", "google"]: # Add other supported providers here
             missing_keys.append(f"LLM_PROVIDER '{cls.LLM_PROVIDER}' is not recognized/supported by config validation.")

        return missing_keys

    @classmethod
    def configure_logging(cls):
        """Configure logging based on settings."""
        # Moved setup_logging import here to avoid potential circular dependency if utils imports config
        from utils.logging_utils import setup_logging
        setup_logging(cls.LOG_LEVEL)
