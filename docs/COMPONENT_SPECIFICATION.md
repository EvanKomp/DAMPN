# DAMPN Component Specification
Updated 02.22 EK
***
***

This package contains minimal units to accomplish a full ML pipleline from data on file to predictions and evaluation.
External dependancies are reduced to well established packages that have very targeted componentss. Simple
components that can be produced in an easy to understand format are created from scratch as opposed to relying on a 3rd party.

### Data preparation
Converting from structures on file to machine ready dataset
***

`dampn.data.augmentor.Augmentor`
Partitions and augements series of labeled reactant, product, and ts structures into a set of structures to be loaded.
Will optionally augment the dataset with non-TS structures between the TS and the reactant or product.
- _Inputs_: Files to consider, portion TS, R, P, augment to keep
- _Primary Outputs_: New file structure

`dampn.data.featurizer.Featurizer`
Call on a file containing molecule information and get back a dgl graph
- _Inputs_: File to consider, which atom and edge features, other parameters eg basis
- _Primary Outputs_: dgl graph

`dampn.data.loader.DataLoader`
Call on a set of molecular structure files, return a dataset object.
- _Inputs_: featurizing method, where to store ml-ready objects
- _Primary Outputs_: Dataset object

### Data pipeline
Preparing dataset for ML
***

`dampn.data.dataset.Dataset`
Contains a set of dgl graphs and targets
- _Inputs_: Processed files on disk
- _Primary Outputs_: self 
__Methods__: Batch graphs into dgl batch to be fed to ML


`dampn.data.splitter.Splitter`
Splits dataset
- _Inputs_: Dataset, how to split eg. random or by atoms
- _Primary Outputs_: multiple dataset

### Modeling
Machine learning model
***

`dampn.model.dgl_modules.DAMPLayer`
Distance attentive message passing layer. Pytorch module
- _Inputs_: Hyperparameters eg. feature and hidden sizes, whether to consider risiduals
- _Primary Outputs_: self
__Methods__: forward pass, updated node and edge states

`dampn.model.model.DAMPModel`
Fully structured model.
- _Inputs_: Hyperparameters:
> Number of layers, where to use residuals, DAMPLayer hyperparameters
- _Primary Outputs_: self
__Methods__: train, evaluate, predict

