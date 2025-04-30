# utils/llm_factory.py
"""
Factory function to create and return a LangChain BaseLanguageModel instance
based on configuration settings. This allows switching LLM providers easily.
"""

import logging
from config import Config # Import configuration

# Configure logger
logger = logging.getLogger(__name__)

# Supported providers - mapping config string to library requirements
SUPPORTED_PROVIDERS = {
    "openai": "langchain_openai",
    "anthropic": "langchain_anthropic",
    "google": "langchain_google_genai",
    # Add other providers here as needed (e.g., "mistralai": "langchain_mistralai")
}

def get_llm_instance():
    """
    Creates and returns a LangChain LLM instance based on Config settings.

    Reads Config.LLM_PROVIDER and instantiates the corresponding
    LangChain chat model (e.g., ChatOpenAI, ChatAnthropic).

    Returns:
        An instance of a LangChain BaseLanguageModel (e.g., ChatOpenAI)
        or None if initialization fails.
    """
    provider = Config.LLM_PROVIDER.lower()
    temperature = Config.LLM_TEMPERATURE
    llm_instance = None

    logger.info(f"Attempting to initialize LLM for provider: '{provider}'")

    if provider not in SUPPORTED_PROVIDERS:
        logger.error(f"Unsupported LLM provider configured: '{provider}'. Supported: {list(SUPPORTED_PROVIDERS.keys())}")
        return None

    try:
        # --- Define a helper function to extract plain model name ---
        def get_plain_model_name(full_name):
            if "/" in full_name:
                plain_name = full_name.split('/', 1)[1]
                logger.debug(f"Extracted plain model name '{plain_name}' from '{full_name}'")
                return plain_name
            logger.debug(f"Using model name as is (no prefix found): '{full_name}'")
            return full_name
        # --- End helper function ---

        if provider == "openai":
            # --- OpenAI ---
            api_key = Config.OPENAI_API_KEY
            full_model_name = Config.OPENAI_MODEL
            base_url = Config.OPENAI_API_BASe
            plain_model_name = get_plain_model_name(full_model_name)

            if not api_key:
                logger.error("OpenAI API key (OPENAI_API_KEY) not found in configuration.")
                return None
            try:
                from langchain_openai import ChatOpenAI
                llm_instance = ChatOpenAI(
                    model=plain_model_name,
                    temperature=temperature,
                    api_key=api_key # Explicitly pass key for clarity
                )
                logger.info(f"Initialized ChatOpenAI with model: {model_name}")
            except ImportError:
                logger.error("Failed to import ChatOpenAI. Install 'langchain-openai'.")
                return None

        elif provider == "anthropic":
            # --- Anthropic ---
            api_key = Config.ANTHROPIC_API_KEY # Assumes this exists in Config
            full_model_name = Config.ANTHROPIC_MODEL # Assumes this exists in Config
            plain_model_name = get_plain_model_name(full_model_name)
            if not api_key:
                logger.error("Anthropic API key (ANTHROPIC_API_KEY) not found in configuration.")
                return None
            try:
                from langchain_anthropic import ChatAnthropic
                llm_instance = ChatAnthropic(
                    model=plain_model_name,
                    temperature=temperature,
                    api_key=api_key,
                    # Add any other required Anthropic parameters here
                    # max_tokens_to_sample=1024 # Example parameter
                )
                logger.info(f"Initialized ChatAnthropic with model: {plain_model_name}")
            except ImportError:
                logger.error("Failed to import ChatAnthropic. Install 'langchain-anthropic'.")
                return None

        elif provider == "google":
            # --- Google Gemini ---
            api_key = Config.GOOGLE_API_KEY # Assumes this exists in Config
            full_model_name = Config.GEMINI_MODEL # Assumes this exists in Config
            plain_model_name = get_plain_model_name(full_model_name)
            if not api_key:
                logger.error("Google API key (GOOGLE_API_KEY) not found in configuration.")
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Note: Temperature might be handled differently or have specific ranges
                llm_instance = ChatGoogleGenerativeAI(
                    model=plain_model_name,
                    google_api_key=api_key,
                    temperature=temperature,
                    convert_system_message_to_human=True # Often helpful for Gemini
                )
                logger.info(f"Initialized ChatGoogleGenerativeAI with model: {plain_model_name}")
            except ImportError:
                logger.error("Failed to import ChatGoogleGenerativeAI. Install 'langchain-google-genai'.")
                return None

        elif provider == "mistralai":
            api_key = Config.MISTRAL_API_KEY
            full_model_name = Config.MISTRAL_MODEL
            plain_model_name = get_plain_model_name(full_model_name)

            if not api_key: 
                logger.error("Mistral API key (MISTRAL_API_KEY) not found.")
                return None
            try:
                from langchain_mistralai.chat_models import ChatMistralAI
                llm_instance = ChatMistralAI(
                    model=plain_model_name, 
                    mistral_api_key=api_key, 
                    temperature=temperature)
                logger.info(f"Initialized ChatMistralAI with model: {plain_model_name}")
            except ImportError: 
                logger.error("Failed to import ChatMistralAI. Install 'langchain-mistralai'.")
                return None

        elif provider == "ollama":
            full_model_name = Config.OLLAMA_MODEL
            base_url = Config.OLLAMA_BASE_URL
            plain_model_name = get_plain_model_name(full_model_name)

            logger.info(f"Attempting Ollama connection: model='{plain_model_name}', ...")
            logger.warning(...)
            try:
                from langchain_ollama import ChatOllama
                llm_instance = ChatOllama(
                    model=plain_model_name,
                    base_url = base_url,
                    temperature= temperature
                    ) 

                logger.info(f"Initialized ChatOllama with model: {plain_model_name} at {base_url}")
                # REMOVED Attribute Override Block
            except ImportError: logger.error(...); return None
            except Exception as ollama_err: logger.error(...); return None

        # --- Add blocks for other providers (ex:) here ---

        else:
            # This case should technically be caught by the initial check, but added for safety
            logger.error(f"Provider '{provider}' logic not implemented in factory.")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize LLM instance for provider '{provider}': {e}", exc_info=True)
        return None

    return llm_instance
