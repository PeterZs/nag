from dataclasses import dataclass, field
from typing import Literal, Union, Dict, Type, List
from tools.serialization.json_convertible import JsonConvertible

SUBCLASSES = None


def check_subclasses():
    """Imports subclasses so they can be used in presets."""
    global SUBCLASSES
    if SUBCLASSES is None:
        from nag.config.hash_grid_encoding_config import HashGridEncodingConfig
        from nag.config.oneblob_encoding_config import OneBlobEncodingConfig
        SUBCLASSES = [HashGridEncodingConfig, OneBlobEncodingConfig]
    else:
        pass


@dataclass
class EncodingConfig(JsonConvertible):
    """Encoding configuration for tinycudann encodings."""

    otype: str = field(default=None)

    @classmethod
    def available_presets(cls) -> list[str]:
        """Get the available presets for this encoding configuration.

        Returns
        -------
        list[str]
            The available preset names.
        """
        presets = set()
        presets = list(presets.union(
            *[set(x.available_presets()) for x in cls.__subclasses__()]))
        return presets

    @classmethod
    def get_presets_per_type(cls) -> Dict[Type["EncodingConfig"], List[str]]:
        """Get the available presets for this encoding configuration.

        Returns
        -------
        dict[str, list[str]]
            The available preset names.
        """
        presets_per_type = {}
        for subclass in cls.__subclasses__():
            presets_per_type[subclass] = subclass.available_presets()
        return presets_per_type

    @classmethod
    def parse(
        cls, obj: Union["EncodingConfig", str]
    ) -> "EncodingConfig":
        """Parse an encoding configuration.

        Parameters
        ----------
        obj : EncodingConfig | str
            The object or a preset name to parse.

        Returns
        -------
        EncodingConfig
            The parsed encoding configuration.

        Raises
        ------
        ValueError
            If the object is not a valid encoding configuration.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.preset(obj)
        raise ValueError(f"Invalid encoding configuration: {obj}")

    @classmethod
    def preset(
        cls, name: Literal["large", "medium", "small", "tiny"]
    ) -> "EncodingConfig":
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
        check_subclasses()
        presets_per_type = cls.get_presets_per_type()
        for encoding_type, presets in presets_per_type.items():
            if name in presets:
                return encoding_type.preset(name)
        raise ValueError(f"Unknown preset: {name}")
