# Flexiv Rizon True VLA

This repository contains the training and evaluation pipeline for a True VLA policy in MuJoCo.

## Prerequisite

First install and run the data-collection project:

https://github.com/Alchemist77/mujoco-vla-data-collection

That project generates the demonstration dataset required for training this VLA model.

## Workflow

1. Install the data-collection repository.
2. Collect demonstration data.
3. Copy the generated `.npz` dataset into this repository.
4. Train the True VLA model.
5. Test the trained checkpoint and save evaluation videos.
