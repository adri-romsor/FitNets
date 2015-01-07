"""
A module containing different learning rules for use with the SGD training
algorithm.
"""
import numpy as np
import warnings

from theano import config
from pylearn2.monitor import Monitor
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import sharedX



class TeacherDecayOverEpoch(TrainExtension):
    """
    Scales the teacher weight linearly on each epochs

    Parameters
    ----------
    start : int
        The epoch on which to start shrinking the learning rate
    saturate : int
        The epoch to saturate the shrinkage
    final_lambda_teach : float
        The teacher weight to use at the end of learning.
    """

    def __init__(self, start, saturate, final_lambda_teach):
        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0
        assert isinstance(start, (py_integer_types, py_float_types))
        assert isinstance(saturate, (py_integer_types, py_float_types))
        assert saturate > start
        assert start >= 0
        assert saturate >= start

    def setup(self, model, dataset, algorithm):
        """
        Initializes the decay schedule based on epochs_seen.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model to which the training algorithm is applied.
        dataset : pylearn2.datasets.Dataset
            The dataset to which the model is applied.
        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            Describes how gradients should be updated.
        """
        monitor = Monitor.get_monitor(model)
        self._count = monitor.get_epochs_seen()
        self._apply_lambda_teach(algorithm)

    def on_monitor(self, model, dataset, algorithm):
        """
        Updates the teacher weight based on the linear decay schedule.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        algorithm : WRITEME
        """
        self._count += 1
        self._apply_lambda_teach(algorithm)

    def _apply_lambda_teach(self, algorithm): 
        """Updates the teacher weight on algorithm based on the epochs elapsed."""
        if not self._initialized:
            self._init_lambda_teach = algorithm.cost.lambda_teach.get_value()
            self._step = ((self._init_lambda_teach - self.final_lambda_teach) /
                          (self.saturate - self.start + 1))
            self._initialized = True
        algorithm.cost.lambda_teach.set_value(np.cast[config.floatX](
            self.current_lambda_teach()))

    def current_lambda_teach(self):
        """
        Returns the teacher weight currently desired by the decay schedule.
        """
        if self._count >= self.start:
            if self._count < self.saturate:
                new_lambda_teach = self._init_lambda_teach - self._step * (self._count
                        - self.start + 1)
            else:
                new_lambda_teach = self.final_lambda_teach
        else:
            new_lambda_teach = self._init_lambda_teach
            
        if new_lambda_teach < 0:
	  new_lambda_teach = 0

	return new_lambda_teach
        
