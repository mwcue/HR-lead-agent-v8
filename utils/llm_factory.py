# utils/llm_factory.py
"""
Factory function to create and return a LangChain BaseLanguageModel instance
based on configuration settings. This allows switching LLM providers easily.
"""

import logging
from config import Config

# Configure logger
logger = logging.getLogger(__name__)

# Supported providers - mapping config string to library requirements
SUPPORTED_PROVIDERS = {
    "openai": "langchain_openai",
    "anthropic": "langchain_anthropic", 
    "google": "langchain_google_genai",
    "mistralai": "langchain_mistralai",
    "ollama": "langchain_community",
}

def get_llm_instance():
    """
    Creates and returns a LangChain LLM instance based on Config settings.

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
        # Helper function to extract plain model name (remove provider prefix)
        def get_plain_model_name(full_name):
            if "/" in full_name:
                plain_name = full_name.split('/', 1)[1]
                logger.debug(f"Extracted plain model name '{plain_name}' from '{full_name}'")
                return plain_name
            logger.debug(f"Using model name as is (no prefix found): '{full_name}'")
            return full_name

        if provider == "openai":
            # OpenAI
            api_key = Config.OPENAI_API_KEY
            full_model_name = Config.OPENAI_MODEL
            plain_model_name = get_plain_model_name(full_model_name)

            if not api_key:
                logger.error("OpenAI API key (OPENAI_API_KEY) not found in configuration.")
                return None
                
            try:
                from langchain_openai import ChatOpenAI
                llm_instance = ChatOpenAI(
                    model=plain_model_name,
                    temperature=temperature,
                    api_key=api_key
                )
                logger.info(f"Initialized ChatOpenAI with model: {plain_model_name}")
            except ImportError:
                logger.error("Failed to import ChatOpenAI. Install 'langchain-openai'.")
                return None

        elif provider == "anthropic":
            # Anthropic
            api_key = Config.ANTHROPIC_API_KEY
            full_model_name = Config.ANTHROPIC_MODEL
            plain_model_name = get_plain_model_name(full_model_name)
            
            if not api_key:
                logger.error("Anthropic API key (ANTHROPIC_API_KEY) not found in configuration.")
                return None
                
            try:
                from langchain_anthropic import ChatAnthropic
                llm_instance = ChatAnthropic(
                    model=plain_model_name,
                    temperature=temperature,
                    api_key=api_key
                )
                logger.info(f"Initialized ChatAnthropic with model: {plain_model_name}")
            except ImportError:
                logger.error("Failed to import ChatAnthropic. Install 'langchain-anthropic'.")
                return None

        elif provider == "google":
            # Google Gemini
            api_key = Config.GOOGLE_API_KEY
            full_model_name = Config.GEMINI_MODEL
            plain_model_name = get_plain_model_name(full_model_name)
            
            if not api_key:
                logger.error("Google API key (GOOGLE_API_KEY) not found in configuration.")
                return None
                
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm_instance = ChatGoogleGenerativeAI(
                    model=plain_model_name,
                    google_api_key=api_key,
                    temperature=temperature,
                    convert_system_message_to_human=True
                )
                logger.info(f"Initialized ChatGoogleGenerativeAI with model: {plain_model_name}")
            except ImportError:
                logger.error("Failed to import ChatGoogleGenerativeAI. Install 'langchain-google-genai'.")
                return None

        elif provider == "mistralai":
            # Mistral AI
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
                    temperature=temperature
                )
                logger.info(f"Initialized ChatMistralAI with model: {plain_model_name}")
            except ImportError:
                logger.error("Failed to import ChatMistralAI. Install 'langchain-mistralai'.")
                return None

        elif provider == "ollama":
            # Ollama
            full_model_name = Config.OLLAMA_MODEL
            plain_model_name = get_plain_model_name(full_model_name)

            try:
                from langchain_community.llms import Ollama
                llm_instance = Ollama(
                    model=plain_model_name,
                    temperature=temperature
                )
                logger.info(f"Initialized Ollama with model: {plain_model_name}")
            except ImportError:
                logger.error("Failed to import Ollama. Install 'langchain-community'.")
                return None

        else:
            logger.error(f"Provider '{provider}' logic not implemented in factory.")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize LLM instance for provider '{provider}': {e}", exc_info=True)
        return None

    return llm_instance
