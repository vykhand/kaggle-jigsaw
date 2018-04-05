# Sample solution for Kaggle Jigsaw competition

Pipeline author: [Andrey Vykhodtsev](https://www.linkedin.com/in/vykhand/)

## Attribution

My work here is focused on tooling, monitoring, execution, clean pipeline and integration with Azure services. Other code was adopted from many public kernels and github repositories that can be [found here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/kernels) and [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion). I tried to keep the references to original kernels or githubs in the code comments, when appropritate. I apologize if I missed some references or quotes, and if so, please let me know.

## About this solution

This is the pipeline I developed during the [Kaggle Jigsaw Toxic comment classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). In this competition, I focused more on tools rather than on winning or developing novel models.

The goal of developing and publishing this is to share reusable code that will help other people run Kaggle competitions on Azure ML Services.

This is the list of features:

 * Everything is a hyperparameter approach. Config files control everything from model parameters to feature generation and cv.
 * Run experiments locally or remotely using Azure ML Workbench
 * Run experiments on GPU-based [Azure DSVM](https://aka.ms/dsvm) machines
 * Easily start and stop VMs
 * Add and remove VMs to and from the VM fleet
 * Zero manual data download for new VMs - files are downloaded from Azure Storage on demand by code.
 * Caching feature transformations in Azure Storage
 * Shutting down the VMs at the end of experiment
 * Time and memory usage logging
 * Keras/Tensorboard integration
 * Usage of [Sacred](https://pypi.python.org/pypi/sacred) library and logging experiment reporducibility data to CosmosDB
 * Integration with Telegram Bot and Telegram notifications
 * Examples of running LightGBM, sklearn models, Keras/Tensorflow


## Presentations and blog posts

This solution is accompanying a few presentations that I gave on [PyData meetup in Ljubljana](https://www.meetup.com/PyData-Slovenia-Meetup/) and [AI meetup in Zagreb](https://www.meetup.com/Artificial-Intelligence-Future-Meetup/). Links to the slides:

 * [PyData Ljubljana #5](https://www.slideshare.net/andreyvykhodtsev/20180328-av-kagglejigsawwithamlwb-92229518)
 * AI Future Zagreb

I am also writing a blog post which is going to published [here](https://www.meetup.com/Artificial-Intelligence-Future-Meetup/).

## Technical overview of the solution


### Pipeline

![Pipeline: code structure](img/pipeline.png)

### Architecture

### Experiment flow

## Deploying solution

## Data downloads

Data for this competition is not distributed with this repository. You need to download it from the competition page located [here](). There are also multiple other external files that I used for this competition:

 * [Google's Full list of banned words]()
 * [Glove embedding vectors]()
 * [FastText embedding vectors]()

## About Azure and Azure CLI



## About Azure ML Services

Welcome to your new Azure Machine Learning Project.

For more information go to <http://aka.ms/AzureMLGettingStarted>

Configure your favorite IDE and open this project using the **File menu**.

Add and prepare data sources using the **Data** tab.

Add and explore notebooks using the **Notebook** tab.

Explore past runs and access project outputs using the **Run History** tab.


## Standard operating procedures

### Running experiments with AMLWB
### Accessing experiment files

#### Submitting to Kaggle

### Adding new compute
### Deallocating machines
### Running series of experiments
### Monitoring GPU usage
### Monitoring Tensorflow using Tensorboard
### Retrieving experiment results from CosmosDB
### Accessing experiment storage directly

## Code structure

## Future work and TODOs
