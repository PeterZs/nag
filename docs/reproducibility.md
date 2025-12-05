# Reproducibility

[Back to README](../README.md) | [Back (Training)](./training.md) | [Next (Working With NAGs Notebook)](../notebooks/working_with_nags.ipynb)


This document will briefly describe how to reproduce all our experiments which we presented in the paper.

For the waymo experiments, we included all training configurations within the [waymo](../config/waymo/) config directory, while all configurations for davis are in the [davis](../config/davis/) directory:

- [waymo/final](../config/waymo/final/) containes the waymo configurations to run the experiments mentioned in our main manuscript. The name follows the waymo segment name, the used frame indices, model size and the used undistorted images. 
- [davis](../config/davis/) containes the davis configurations to run the experiments mentioned in our main manuscript. The name corresponds to the davis scene.
- [large-ego](../config/waymo/large-ego/) contains the configurations for the waymo large-ego motion experiments in our supplementary material.
- [ablations](../config/waymo/lablations/) contains the configurations for the waymo ablation experiments in our supplementary material. The contains the the first 3 digits of the used segments, start / end indices, model size and a further name specifing the differences. If further documentation is needed, feel free to drop a Github issue.

After training with the usual training script, the models are evaluated automatically according our stated metrics. All metrics are saved in our Tracker objects, which is serialized to disk, producing CSV-files, and also logged to the respective experiment logger. 

If you interested in evaluating the NAG, decompose scenes, or texture edit them you may want to look into our [working_with_nag.ipnyb](../notebooks/working_with_nags.ipynb) notebook.

> **Note:** We did our best to give reproducible code by the use of deterministic algorithms and seeding, yet we cannot necessary guarantee that on different machines, with different library versions or CUDA implementations, or due to errors during cleanup of the code the results are exactly the same. If something is significantly different then expected and stated in our paper, please open a Github issue, so we can investigate the problem.