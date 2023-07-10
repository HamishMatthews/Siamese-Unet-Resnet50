# Siamese UNet with TorchGeo Pre-trained ResNet50 for Sentinel-2 L2A Binary Classification

This repository contains an implementation of a Siamese UNet model with a pre-trained ResNet50 backbone from TorchGeo for binary classification of Sentinel-2 L2A satellite imagery. The model is designed to accurately classify pixels as either fire-affected or background. The Siamese architecture enhances feature extraction through shared weights between two UNet encoders, improving classification accuracy.

## Motivation

Satellite missions enable large-scale data acquisitions at high resolution in a short amount of time, paving the way for exciting scenarios and applications in the computer vision domain. The capacity to collect continental-scale information represents an extremely valuable resource to researchers and authorities in different contexts.

In Earth Observation, crisis response and disaster management represent relevant topics. Data availability plays a vital role in timely responses and continuous monitoring of areas affected by catastrophic events, such as earthquakes, floodings and forest fires.

In this context, ChaBuD challenge proposes a computer vision task, leveraging raster bi-temporal geospatial data collected from Sentinel-2 L2A satellite imagery. The goal is to identify areas previously affected by forest wildfires over the state of California to support local authorities in monitoring the affected areas and planning the restoration process.

## Task Description

The challenge proposes a binary image segmentation task on forest fires monitored over California.
The goal is to predict whether or not a region was affected by a forest fire, given a pre-fire and post-fire satellite acquisition.

Participants are encouraged to leverage on temporal information to develop change detection models.

## Evaluation

IoU is used for final evaluation. Evaluation score is computed the validation set (public leaderboard) and hidden test set (private leaderboard).

The final evaluation (private leaderboard) admits only two submissions for the entire challenge.

## Repository Information

This repository contains my implementation of the Siamese UNet model with a pre-trained ResNet50 backbone for the ChaBuD challenge. The code includes data preprocessing, model implementation, training, and evaluation scripts.

## Downloading Training Data
The hdf5 training data can either be downloaded from the competition website or using the script located in the `datasets` folder. 

## Competition Information

https://huggingface.co/spaces/competitions/ChaBuD-ECML-PKDD2023

## Contact

hamish.e.matthews@gmail.com

