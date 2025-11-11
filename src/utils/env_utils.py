"""
Environment variable utilities for loading .env files and setting up paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger()


def load_environment_variables(env_file: str = ".env") -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file
    """
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from: {env_file}")
    else:
        logger.warning(f".env file not found at: {env_file}. Using system environment variables.")


def get_env_variable(var_name: str, default: str = None) -> str:
    """
    Get an environment variable or return default.
    
    Args:
        var_name: Name of environment variable
        default: Default value if variable not found
    
    Returns:
        Environment variable value or default
    """
    value = os.getenv(var_name, default)
    if value is None:
        logger.warning(f"Environment variable '{var_name}' not found and no default provided.")
    return value


def setup_cache_directories() -> None:
    """
    Set up cache directories for models and transformers.
    Uses environment variables if set, otherwise uses defaults.
    """
    # Get cache paths from environment or use defaults
    ollama_models = get_env_variable('OLLAMA_MODELS')
    transformers_cache = get_env_variable('TRANSFORMERS_CACHE')
    hf_home = get_env_variable('HF_HOME')
    
    # Set them if they exist
    if ollama_models:
        os.environ['OLLAMA_MODELS'] = ollama_models
        logger.info(f"Set OLLAMA_MODELS to: {ollama_models}")
    
    if transformers_cache:
        os.environ['TRANSFORMERS_CACHE'] = transformers_cache
        logger.info(f"Set TRANSFORMERS_CACHE to: {transformers_cache}")
    
    if hf_home:
        os.environ['HF_HOME'] = hf_home
        logger.info(f"Set HF_HOME to: {hf_home}")


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to project root
    """
    # Get the directory containing this file, then go up to project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root