# Medical Image Analysis Laboratory

Welcome to the medical image analysis laboratory (MIALab).
This repository contains all code you will need to recreate the results presented in our report, "Almost Perfect Segmentation Without Better Models: The Impact of Evaluation Metrics" by Emma Chambers, Camila Silva Delgado and Esther Su Wee Tan.

----
To run, use the following in your command line or script:

```
python pipeline.py [OPTIONS]

--result_dir PATH
    Directory for results.
    Default: ./mia-result (relative to script directory)

--data_atlas_dir PATH
    Directory with atlas data.
    Default: mialab/data/atlas

--data_train_dir PATH
    Directory with training data.
    Default: mialab/data/train/

--data_test_dir PATH
    Directory with testing data.
    Default: mialab/data/test/

--RF {Standard,GridSearch,Balanced}
    Specify how to run the Random Forest.
    Default: Standard

--run_metric_expts
    Run metric manipulation experiments (largest CC, shrink,
    distance trimming) at multiple stages of the pipeline.
    Default: off

--load_model
    Load model weights if available.
    Default: off

--save_model_weights
    Save model weights after training.
    Default: off

```

----

Environment prerequisites:
- python~=3.10 (prevent wheel building error with SimpleITK)

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.
