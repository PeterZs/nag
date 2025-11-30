from dataclasses import dataclass, field
from typing import Literal, Union
from tools.serialization.json_convertible import JsonConvertible


@dataclass
class NetworkConfig(JsonConvertible):
    """Network configuration for tinycudann models."""

    otype: str = field(default="FullyFusedMLP")

    activation: str = field(default="ReLU")

    output_activation: str = field(default="None")

    n_neurons: int = field(default=64)

    n_hidden_layers: int = field(default=5)

    @classmethod
    def parse(
        cls, obj: Union["NetworkConfig", str]
    ) -> "NetworkConfig":
        """Parse a network configuration.

        Parameters
        ----------
        obj : NetworkConfig | str
            The object or a preset name to parse.

        Returns
        -------
        NetworkConfig
            The parsed network configuration.

        Raises
        ------
        ValueError
            If the object is not a valid network configuration.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.preset(obj)
        raise ValueError(f"Invalid network configuration: {obj}")

    @classmethod
    def preset(
        cls, name: Literal["large", "medium", "small", "tiny"]
    ) -> "NetworkConfig":
        """Get a preset network configuration.

        Parameters
        ----------
        name : Literal[&quot;large&quot;, &quot;medium&quot;, &quot;small&quot;]
            The name of the preset.

        Returns
        -------
        NetworkConfig
            The preset network configuration.

        Raises
        ------
        ValueError
            If the preset name is unknown.
        """
        if name == "large":
            return cls(
                otype="FullyFusedMLP",
                activation="ReLU",
                output_activation="None",
                n_neurons=64,
                n_hidden_layers=5
            )
        elif name == "medium":
            return cls(
                otype="FullyFusedMLP",
                activation="ReLU",
                output_activation="None",
                n_neurons=64,
                n_hidden_layers=4
            )
        elif name == "small":
            return cls(
                otype="FullyFusedMLP",
                activation="ReLU",
                output_activation="None",
                n_neurons=64,
                n_hidden_layers=3
            )
        elif name == "tiny":
            return cls(
                otype="FullyFusedMLP",
                activation="ReLU",
                output_activation="None",
                n_neurons=32,
                n_hidden_layers=3
            )
        else:
            raise ValueError(f"Unknown preset: {name}")
