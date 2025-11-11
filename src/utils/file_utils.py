"""
File utility functions for handling configs, directories, and file operations.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger()


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        Dict containing configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded JSON config from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dict containing configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded YAML config from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise


def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Path to output file
    """
    # Ensure parent directory exists
    ensure_directory_exists(os.path.dirname(output_path))
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {output_path}: {e}")
        raise


def load_text_file(file_path: str) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to text file
    
    Returns:
        Text content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Loaded text from: {file_path}")
        return text
    except FileNotFoundError:
        logger.error(f"Text file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        raise


def save_text_file(text: str, output_path: str) -> None:
    """
    Save text to a file.
    
    Args:
        text: Text content to save
        output_path: Path to output file
    """
    # Ensure parent directory exists
    ensure_directory_exists(os.path.dirname(output_path))
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved text to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save text to {output_path}: {e}")
        raise


def get_all_files_in_directory(directory: str, extension: str = None) -> list:
    """
    Get all files in a directory, optionally filtered by extension.
    
    Args:
        directory: Directory path
        extension: File extension to filter (e.g., '.pdf', '.txt')
    
    Returns:
        List of file paths
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if extension is None or file.endswith(extension):
                files.append(file_path)
    
    logger.info(f"Found {len(files)} files in {directory}")
    return files


def get_file_name_without_extension(file_path: str) -> str:
    """
    Get filename without extension from a file path.
    
    Args:
        file_path: Full file path
    
    Returns:
        Filename without extension
    """
    return Path(file_path).stem