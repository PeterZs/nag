from dataclasses import dataclass, field
from typing import Literal, Union
from tools.serialization.json_convertible import JsonConvertible
from nag.config.encoding_config import EncodingConfig


@dataclass
class HashGridEncodingConfig(EncodingConfig):
    """Encoding configuration for tinycudann encodings."""

    otype: str = field(default="HashGrid")

    n_levels: int = field(default=16)

    n_features_per_level: int = field(default=4)

    log2_hashmap_size: int = field(default=17)

    base_resolution: int = field(default=4)

    per_level_scale: float = field(default=1.61)

    interpolation: str = field(default="Linear")

    @classmethod
    def available_presets(cls) -> list[str]:
        """Get the available presets for this encoding configuration.

        Returns
        -------
        list[str]
            The available preset names.
        """
        return ["large", "medium", "small", "tiny"]

    @classmethod
    def preset(
        cls, name: Literal["large", "medium", "small", "tiny"]
    ) -> "HashGridEncodingConfig":
        """Get a preset encoding configuration.

        Parameters
        ----------
        name : Literal[&quot;large&quot;, &quot;medium&quot;, &quot;small&quot;, &quot;tiny&quot;]
            The name of the preset.

        Returns
        -------
        EncodingConfig
            The preset encoding configuration.

        Raises
        ------
        ValueError
            If the preset name is unknown.
        """
        if name == "large":
            return cls(
                n_levels=16,
                n_features_per_level=4,
                log2_hashmap_size=17,
                base_resolution=4,
                per_level_scale=1.61,
                interpolation="Linear"
            )
        elif name == "medium":
            return cls(
                n_levels=12,
                n_features_per_level=4,
                log2_hashmap_size=15,
                base_resolution=4,
                per_level_scale=1.61,
                interpolation="Linear"
            )
        elif name == "small":
            return cls(
                n_levels=8,
                n_features_per_level=4,
                log2_hashmap_size=13,
                base_resolution=4,
                per_level_scale=1.61,
                interpolation="Linear"
            )
        elif name == "tiny":
            return cls(
                n_levels=6,
                n_features_per_level=4,
                log2_hashmap_size=12,
                base_resolution=4,
                per_level_scale=1.61,
                interpolation="Linear"
            )
        else:
            raise ValueError(f"Unknown preset: {name}")
