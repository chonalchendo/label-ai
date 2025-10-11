import typing as T
from pathlib import Path

import omegaconf as oc

type Config = oc.ListConfig | oc.DictConfig
type Configs = T.Sequence[Config]


def parse_file(path: str) -> Config:
    """Parse a config file from a path.

    Args:
        path (str): path to local config.

    Returns:
        Config: representation of the config file.
    """
    return oc.OmegaConf.load(path)


def parse_string(string: str) -> Config:
    """Parse the given config string.

    Args:
        string (str): content of config string.

    Returns:
        Config: representation of the config string.
    """
    return oc.OmegaConf.create(string)


def merge_configs(configs: Configs) -> Config:
    """Merge a list of config into a single config.

    Args:
        configs (T.Sequence[Config]): list of configs.

    Returns:
        Config: representation of the merged config objects.
    """
    return oc.OmegaConf.merge(*configs)


def to_object(config: Config, resolve: bool = True) -> object:
    """Convert a config object to a python object.

    Args:
        config (Config): representation of the config.
        resolve (bool): resolve variables. Defaults to True.

    Returns:
        object: conversion of the config to a python object.
    """
    return oc.OmegaConf.to_container(config, resolve=resolve)


def load_and_merge_configs(
    config_files: T.Sequence[Path],
    extras: T.Sequence[str] | None = None,
) -> oc.DictConfig:
    """Load configuration files and merge with extra parameters."""
    # Parse config files
    file_configs = [parse_file(str(file)) for file in config_files]

    # Parse extra strings
    string_configs = []
    if extras:
        string_configs = [parse_string(string) for string in extras]

    # Merge all configs
    config = merge_configs([*file_configs, *string_configs])

    if not isinstance(config, oc.DictConfig):
        raise RuntimeError("Config is not a dictionary")

    return config
