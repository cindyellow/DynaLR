# Dynamic Learning Rate Scheduling

## 1. About this project

This is completed as part of the final project for [CSC413](https://uoft-csc413.github.io/2023/) at the University of Toronto. Contributors to this code are Shih-Ting (Cindy) Huang ([@cindyellow](https://github.com/cindyellow)) and Warren Zhu ([@CoderWarren](https://github.com/CoderWarren))

## 2. Usage
`dynaopt.py` specifies how our customized optimizer works - searching through potential learning rates for each step. To run the CNN training example with this optimizer, check out `main.ipynb`, which uses ConvNet to classify images from CIFAR10. `NLM.ipynb` provides another example of our algorithm implemented in a neural language model for masked word prediction task.