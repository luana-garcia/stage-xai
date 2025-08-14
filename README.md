# Research Project on XAI and Fairness

This internship is a collaboration between:
- INRIA (Institut national de recherche en sciences et technologies du numérique) Paris, and
- University of Toulouse - IMT (Institut de Mathéematiques de Toulouse).

Work made in the scope of the Regalia Laboratory (INRIA Paris).

We use the Folktables dataset and library to explore the bias in the Census data of the USA (year 2018).

## Requirements

You must have Python installed to run it.

## Installation

For running our project, you can first create a virtual enviroment, we recommend the [Anaconda](https://www.anaconda.com/):

```sh
conda create --name project_env
conda activate project_env
```

After you can install all the requirements through the file `requirements.txt`:
```sh
pip install -r requirements.txt
```

or if you're testing the library, you can use in the editable mode:

```sh
pip install -e .
```

## Usage

In the `test` folder you can see some examples of usage.

You can use it to train different models and see the performance on each subdataset. Our `DataLoader`class works with the folktables dataset, but it's easily adaptable.

It can also be used to run Anchors and SHAP on your dataset and do a PCA analysis after.