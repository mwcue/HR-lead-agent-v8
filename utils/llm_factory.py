# utils/llm_factory.py
"""
Factory function to create and return a LangChain BaseLanguageModel instance
based on configuration settings. This allows switching LLM providers easily.
"""

import logging
from config import Config # Import configuration

# Configure logger
logger = logging.getLogger(__name__)

# ADDED: Ollama and MistralAI to supported providers
SUPPORTED_PROVIDERS = {
    "openai": "langchain_openai",
    "anthropic": "langchain_anthropic",
    "google": "langchain_google_genai",
    "ollama": "langchain_community",
    "mistralai": "langchain_mistralai",
}

def get_llm_instance():
    """
    Creates and returns a LangChain LLM instance based on Config settings.

    Reads Config.LLM_PROVIDER and instantiates the corresponding
    LangChain chat model (e.g., ChatOpenAI, ChatAnthropic, ChatOllama, ChatMistralAI).

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
        if provider == "openai":
            # --- OpenAI ---
            api_key = Config.OPENAI_API_KEY
            model_name = Config.OPENAI_MODEL
            if not api_key:
                logger.error("OpenAI API key (OPENAI_API_KEY) not found in configuration.")
                return None
            try:
                from langchain_openai import ChatOpenAI
                llm_instance = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key
                )
                logger.info(f"Initialized ChatOpenAI with model: {model_name}")
            except ImportError:
                logger.error("Failed to import ChatOpenAI. Install 'langchain-openai'.")
                return None

        elif provider == "anthropic":
            # --- Anthropic ---
            api_key = Config.ANTHROPIC_API_KEY
            model_name = Config.ANTHROPIC_MODEL
            if not api_key:
                logger.error("Anthropic API key (ANTHROPIC_API_KEY) not found in configuration.")
                return None
            try:
                from langchain_anthropic import ChatAnthropic
                llm_instance = ChatAnthropic(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key,
                )
                logger.info(f"Initialized ChatAnthropic with model: {model_name}")
            except ImportError:
                logger.error("Failed to import ChatAnthropic. Install 'langchain-anthropic'.")
                return None

        elif provider == "google":
            # --- Google Gemini ---
            api_key = Config.GOOGLE_API_KEY
            model_name = Config.GEMINI_MODEL
            if not api_key:
                logger.error("Google API key (GOOGLE_API_KEY) not found in configuration.")
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm_instance = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=temperature,
                    convert_system_message_to_human=True
                )
                logger.info(f"Initialized ChatGoogleGenerativeAI with model: {model_name}")
            except ImportError:
                logger.error("Failed to import ChatGoogleGenerativeAI. Install 'langchain-google-genai'.")
                return None

        # --- ADDED: Ollama ---
        elif provider == "ollama":
            model_name = Config.OLLAMA_MODEL # Assumes this exists in Config
            base_url = Config.OLLAMA_BASE_URL # Assumes this exists in Config
            logger.info(f"Attempting Ollama connection: model='{model_name}', base_url='{base_url}'")
            logger.warning("Ensure the Ollama application is running and the model is downloaded locally.")
            try:
                # Ollama doesn't use temperature in the same way during init usually
                # It might be passed during invoke or set via model parameters if supported
                from langchain_community.chat_models import ChatOllama
                llm_instance = ChatOllama(
                    model=model_name,
                    base_url=base_url,
                    temperature=temperature # Pass temperature if supported by the Langchain integration version
                )
                # You might want a check here to see if the Ollama server is reachable
                # llm_instance.invoke("Hi") # Example test - remove in production
                logger.info(f"Initialized ChatOllama with model: {model_name} at {base_url}")
            except ImportError:
                logger.error("Failed to import ChatOllama. Install 'langchain-community'.")
                return None
            except Exception as ollama_err:
                 logger.error(f"Failed to initialize Ollama. Is the server running and model available? Error: {ollama_err}", exc_info=True)
                 return None

        # --- ADDED: MistralAI ---
        elif provider == "mistralai":
            api_key = Config.MISTRAL_API_KEY # Assumes this exists in Config
            model_name = Config.MISTRAL_MODEL # Assumes this exists in Config
            if not api_key:
                logger.error("Mistral API key (MISTRAL_API_KEY) not found in configuration.")
                return None
            try:
                from langchain_mistralai.chat_models import ChatMistralAI
                llm_instance = ChatMistralAI(
                    model=model_name,
                    mistral_api_key=api_key,
                    temperature=temperature
                )
                logger.info(f"Initialized ChatMistralAI with model: {model_name}")
            except ImportError:
                logger.error("Failed to import ChatMistralAI. Install 'langchain-mistralai'.")
                return None

        else:
            logger.error(f"Provider '{provider}' logic not implemented in factory.")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize LLM instance for provider '{provider}': {e}", exc_info=True)
        return None

    # Final check
    if llm_instance is None:
         logger.error(f"LLM instance is None after attempting initialization for provider '{provider}'.")

    return llm_instance
