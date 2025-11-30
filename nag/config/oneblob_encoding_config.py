from dataclasses import dataclass, field
from typing import Literal, Union
from tools.serialization.json_convertible import JsonConvertible
from nag.config.encoding_config import EncodingConfig


@dataclass
class OneBlobEncodingConfig(EncodingConfig):
    """Encoding configuration for tinycudann encodings."""

    otype: str = field(default="OneBlob")

    n_bins: int = field(default=16)

    @classmethod
    def available_presets(cls) -> list[str]:
        """Get the available presets for this encoding configuration.

        Returns
        -------
        list[str]
            The available preset names.
        """
        return ["oneblob-large", "oneblob-medium", "oneblob-small", "oneblob-tiny"]

    @classmethod
    def preset(
        cls, name: Literal["oneblob-large", "oneblob-medium", "oneblob-small", "oneblob-tiny"]
    ) -> "OneBlobEncodingConfig":
        """Get a preset encoding configuration.

        Parameters
        ----------
        name : Literal["oneblob-large", "oneblob-medium", "oneblob-small", "oneblob-tiny"]
            The name of the preset.

        Returns
        -------
        OneBlobEncodingConfig
            The preset encoding configuration.

        Raises
        ------
        ValueError
            If the preset name is unknown.
        """
        if name == "oneblob-large":
            return cls(
                n_bins=64,
            )
        elif name == "oneblob-medium":
            return cls(
                n_bins=32,
            )
        elif name == "oneblob-small":
            return cls(
                n_bins=16,
            )
        elif name == "oneblob-tiny":
            return cls(
                n_bins=8,
            )
        else:
            raise ValueError(f"Unknown preset: {name}")
