# NAG Training

[Back to README](../README.md) | [Back (Datasets)](./datasets.md) | [Next (Reproducibility)](reproducibility.md)

In this section, we provide instructions on how to train Neural Atlas Graphs (NAG) models using our implementation.

The main entry point for training a NAG model is the `run_nag.py` script located in the `nag/scripts/` directory. This script allows you to configure and initiate the training process.


The training process can be started using the following command:

```bash
python nag/scripts/run_nag.py --config-path [path-to-config]
```

Which will start the training using the specified configuration file. In the [config/](config/) directory, you can find all configuration files used for our experiments, including those for the Waymo and Davis datasets.

Our training is fully configurable via a YAML config file whose contents should match our [config class](nag/config/nag_config.py). You can also pass configuration arguments as command line arguments to override specific settings in the config file. To do so, replace underscores (`_`) in the config field names with hyphens (`-`) when passing them as command line arguments. E.g. if you want to change the `learn_resolution_factor` field (to train on lower resolution), you would pass `--learn-resolution-factor [value]` as a command line argument.

During the training process, the model is evaluated to produce outputs for all frames, calculating metrics, as well as scene decompositions for every object. You can monitor the training progress and results using TensorBoard or Weights & Biases, depending on your logging configuration. The default logger defined in our configuration is "tensorboard", this can be changed by altering the "experiment_logger" property in the NAGConfig (This field is inherited from the ExperimentOutputConfig class). For our experiments, we where using Weights & Biases (wandb), such that this value corresponds to wandb for all our training configs. Yet, if you does not have wandb installed, it will automatically fall back to tensorboard.
If you want to use Weights & Biases, make sure to set the `WANDB_API_KEY` in your `.env` file.

Since you know now how the training works, you maybe want to check our [Reproducibility](./reproducibility.md) notes
or directly jump to our [working_with_nag.ipnyb](../notebooks/working_with_nags.ipynb) notebook, where we provide a pre-trained checkpoint of the NAG, showcase how to decompose scenes and texture edit them.


