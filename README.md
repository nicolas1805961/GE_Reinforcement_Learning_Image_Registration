# Registration 2D / 2D with a Deep Q Learning approach - An implementation

## Context

This project is an end of studies joint project between
[École Pour l’Informatique et les Techniques Avancées (*EPITA*)](https://www.epita.fr/)
[IMAGE specialization](https://www.epita.fr/nos-formations/diplome-ingenieur/cycle-ingenieur/les-majeures/#majeure-IMAGE)
and [GE Healthcare](https://www.gehealthcare.fr/).

###### EPITA students
Geoffrey Jount <geoffrey.jount@epita.fr>\
Nicolas Portal <nicolas.portal@epita.fr>\
Raphaël Dias--Monteiro <raphael.dias-monteiro@epita.fr>

###### GE Healthcare supervisors
Maxime Taron <maxime.taron@ge.com>\
Thomas Benseghir <thomas.benseghir@ge.com>


## Previous works

This project is based on [Liao et al., *An Artificial Agent for Robust Image Registration*, 2016, arXiv:**1611.10336**](https://arxiv.org/abs/1611.10336).

## Documentation

The provided documentation is not aimed at presenting the problem, describing the reasoning behind the developed
framework or providing results and comparisons over other possible solutions.

## Architecture

### `agent` module

###### Files
- `registration.py`

The `agent` module contains the `RegistrationAgent` object that features the principal methods to:
- fit training data,
- evaluate registration with ground truth data
- infer registration transformations
- visualize the agent registration steps

### `datasets` module

###### Files
- `dataset.py`

The `datasets` module contains the several dataset interfaces that are used in this project.\
The base class `DQNDataset` inherits from `Pytorch Dataset` object and acts as an abstract class that should not be used
by itself. This dataset owns reference images and floating images.\
From this class derive three useful classes:
- `TrainDQNDataset` - owns the q-values used in training
- `RegisterDQNDataset` - owns the big images and the centers of rotations
- `TestDQNDataset` - derives from `RegisterDQNDataset`, owns the ground truth transformations  

### `deepqnet` module

###### Files
- `deepqnet.py`

The `deepqnet` module contains the Convolutional Neural Network model that is used to learn a reward policy for the
registration agent.

This model is built on top of `Pytorch` neural network modules and layers. The `fit` method requires a generator built
from a `TrainDQNDataset`.

### `qvalues` module

###### Files
- `qtable.py`
- `qvalues.ipynb`

The `qvalues` module contains the actions of the agent following the three degrees of freedom, the `Q` object that
computes the reward for a given transformation and the `Q_table` object that stores an array of `Q` objects.

The q values once computed are saved to a folder to be used when fitting the DeepQNet model.

The `qvalues.ipynb` notebook is the interface to specify the data to be loaded and from which to compute the q values
accordingly.

### Root

###### Files
- `utils.py`


- `evaluation.ipynb`
- `training.ipynb`
- `visualization.ipynb`

The `utils.py` file exposes several functions that are used to manipulate the images when applying transformations or
to visualize registration steps.

The different notebooks do exactly what their names are.

## Pipeline

![](doc/pipeline.png)

## Setting up and running

### Google Colab

If you are missing some technical requirements, [Google Colab](https://colab.research.google.com/) provides a good
alternative to use a full Python environment with access to GPU computing power, useful for the training of the
Deep Q Network.

Because the Pipeline is splitted into several modules and the Google Colab instances are stateless, the outputs and
generated files can not be passed from one notebook to another.

##### How to run and store the notebooks on a Colab instance ?

In order to have access to the different modules of the Pipeline, a notebook must be opened with the other folders
of the repository at the same location.
This means that in practice, the whole repository must be cloned in order to use the training, inference or
visualization parts.  
Doing so from the notebook itself does not enable to run other notebooks nor to save the data, models and other.

The following is the workflow that we have found to be the least burdensome:
- Clone the `scripts` repository into a Google Drive,
- Open the notebook of interest or any notebook that is inside the repository from the Google Drive interface,
- In the first cells of the notebook, execute the mounting of the Drive, this will allow the notebook to access
the Python scripts and to load and save data from the Drive,
- Another important cell to execute is the one performing a directory change, the directory destination can be
modified depending on the situation but be careful to update accordingly the imports of the modules to the new
location of the current working directory
- Input and output data can then be loaded and saved to the Drive directly, thus allowing them to be reused by
later scripts or notebook without having to download and upload them manually.

### Localhost

All Python dependencies are listed in the `requirements.txt` file. To install them, create a virtual env and use the
Python package installer `pip`.

```sh
42sh$ python -m venv env
42sh$ . env/bin/activate
(env) 42sh$ pip install -r requirements.txt
```

With a local execution, the mounting of the Drive is not useful for the matter of passing data from one notebook to
another.

The modification of the current working directory can also not be an issue anymore depending on the situation.
