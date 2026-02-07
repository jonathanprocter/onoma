import os
from typing import Any

import toml

DEFAULT_CONFIG = {
    "default_provider": "openai",
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "openai_model": "gpt-4o",
    "azure_openai_endpoint": "",
    "azure_openai_api_key": "",
    "azure_openai_api_version": "2024-02-01",
    "azure_openai_deployment": "",
    "use_azure_openai": False,
    "anthropic_api_key": "",
    "anthropic_base_url": "https://api.anthropic.com",
    "anthropic_model": "claude-3-5-sonnet-20241022",
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "llama3.2:latest",
    "google_api_key": "",
    "naming_convention": "snake_case",
    "llm_model": "gpt-4o",
    "min_filename_words": 5,
    "max_filename_words": 15,
    "system_prompt": "",
    "user_prompt": "",
    "image_prompt": "",
    "enforce_title_case": True,
    "apply_titlecase_all": False,
    "subtitle_separator": " - ",
    "acronyms_path": "",
    "strip_metadata": True,
    "markitdown": {
        "enable_plugins": False,
        "docintel_endpoint": "",
    },
    "report_enabled": True,
    "report_format": "jsonl",
    "report_dir": "",
    "last_report_path": "",
    "batch_rules": [],
    "duplicates_dir": "duplicates",
    "fuzzy_duplicate_threshold": 0.9,
    "rename_folders": False,
}


def get_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from the given config_path or from ~/.onomarc if not specified.
    Returns the config as a dict, or DEFAULT_CONFIG if loading fails.
    """
    if config_path is None:
        config_path = os.path.expanduser("~/.onomarc")
    else:
        config_path = os.path.expanduser(config_path)
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                return merge_config(toml.load(f))
        except Exception:
            pass
    return DEFAULT_CONFIG


def merge_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge a user config with defaults, preserving nested sections."""
    merged = DEFAULT_CONFIG.copy()
    merged_markitdown = DEFAULT_CONFIG.get("markitdown", {}).copy()
    if isinstance(config.get("markitdown"), dict):
        merged_markitdown.update(config["markitdown"])
    merged.update(config)
    merged["markitdown"] = merged_markitdown
    return merged
