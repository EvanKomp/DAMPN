"""Trainable model class."""
from typing import List, Union

import numpy
import sklearn.metrics
import torch
import torch.nn as nn

import dampn.data.dataset
import dampn.data.transformer


class Callback:
    """Store values over epochs.
    
    Should not be instantiated directly. Just a fancy dict.
    
    Example
    -------
    >>>def MAE(a, b):
    >>>    return np.mean(np.abs(a - b))
    >>>model.train(dataset, val_data=val_dataset, epochs=20, log_every_n_epochs=5, metrics=[MAE])
    >>>callback = model.history
    >>>callback
    ['train_loss', 'val_loss', 'train_MAE', 'val_MAE']
    >>>callback['val_MAE']
    [5, 10, 15, 20], [.8, .65, .53, .51]
    """
    def __init__(self):
        self.metrics = {}
        return
    
    def __repr__(self):
        return f"Callback contains metrics: {tuple(self.metrics.keys())}"
    
    def __getitem__(self, metric_name: str):
        if metric_name not in list(self.metrics.keys()):
            raise AttributeError("Calback does not contain values for this metric.")

        return self.metrics[metric_name]
    
    def append(self, epoch: int, metrics_dict):
        if type(epoch) != int:
            raise ValueError('`epoch` must be int')
        for metric_name, matric_value in metric_dict.items():
            if metric_name not in list(self.metrics.keys()):
                self.metrics[metric_name] = {'epochs': [], 'values': []}
            
            if len(self.metrics[metric_name]['epochs']) == 0:
                pass
            elif epoch <= self.metrics[metric_name]['epochs'][-1]:
                raise ValueError('Time travel! `epoch` must be greater than what has already been recorded.') 
            else:
                self.metrics[metric_name]['epochs'].append(epoch)
                self.metrics[metric_name]['values'].append(matric_value)
        return
                

class Model():
    
    def __init__(self):
        """Prepare attributes and construct the model.
        
        Must be supered in child.
        """
        if not hasattr(self, 'model'):
            raise AttributeError('Constructor did not assign the `model` attribute.')
        if not hasattr(self, 'optimizer'):
            raise AttributeError('Constructor did not assign the `optimizer` attribute.')
        if not hasattr(self, 'loss_fn'):
            raise AttributeError('Constructor did not assign the `loss_fn` attribute.')
        self.history = Callback()
        self.epoch = 0
        return
    
    def train(
        self,
        dataset: dampn.data.dataset.Dataset,
        epochs: int = 10,
        batch_size: 12,
        log_every_n_epochs: int = 5
        metrics: List[Union[str, callable]] = None,
        val_dataset: dampn.data.dataset.Dataset = None
    ):
        """Train the model on the training data.
        
        
        """
        # prepare the metrics to use
        metric_callables = {}
        for metric in metrics:
            if type(metric) == str:
                metric_callables[metric] = getattr(sklearn.metrics, metric)
            else:
                metric_callables[metric.__name__] = metric
        
        
        begin_epoch = int(self.epoch)
        for e in range(epochs):
            batches = dataset.batch_generator(batch_size)
            for X_batch, y_batch in batches:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()
                
                self.epoch += 1
                
                if self.epoch - begin_epoch % log_every_n_epochs == 0:
                    matric_values = {'train_loss': float(loss)}
                    
                    # TODO metrics and val loss
        return
                    
    def predict(
        self,
        dataset: dampn.data.dataset.Dataset,
        transformers: List[dampn.data.transformer.Transformer] = None,
        return_true: bool = False
    ):
        """Predict on dataset.


        """
        batches = dataset.batch_generator(batch_size)
        y_pred = []
        y_true = []
        for X_batch, y_batch in batches:
            y_pred.append(self.model(X_batch).detach().numpy())
            y_true.append(y_batch)
        y_pred = numpy.concatenate(y_pred, axis=0)
        y_true = numpy.concatenate(y_true, axis=0)

        if transformers is not None:
            raise NotImplemented()
            # have not created transformers yet
            for transformer in transformer:
                if transformer.which == 'y':
                    y_pred = transformer.untransform(y_pred)
                    y_true = transformer.untransform(y_true)
        if return_true:
            return y_pred, y_true
        else:
            return y_pred

    def evaluate(
        self,
        dataset: dampn.data.dataset.Dataset,
        metric: callable,
        transformers: List[dampn.data.transformer.Transformer] = None
    ):
        """Score model on dataset
        
        
        """
        y_pred, y_true = self.predict(dataset=dataset, transformers=transformers, return_true=True)
        return metric(y_true, y_pred)

                
class DAMPNModel(Model):
    
    def __init__(
        self,
        n_dampn_layers: int,
        node_feature_size: int,
        edge_feature_size: int,
        output_size: int,
        node_hidden_sizes: Union[int, List[int]] = 28,
        edge_hidden_sizes: Union[int, List[int]] = 28,
        context_sizes: Union[int, List[int]] = 28,
        classification: bool = False,
        system_features_size: int = None,
        update_edges: bool = False,
        readout: "AttentiveFP",
        dropouts: Union[float, List[float]] = 0.0,
        loss_fn_regr = nn.MSELoss,
        loss_fn_class = nn.CrossEntropyLoss,
        optimizer = torch.optim.Adam,
        learning_rate = .01,
        **kwargs
    ):
        if classification:
            self.loss_fn = loss_fn_class()
        else:
            self.loss_fn = loss_fn_regr()
        
        self.model = dampn.model.modules.DAMPNModule(
        
        )
            
        
        super().__init__()
        return