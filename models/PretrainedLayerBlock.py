import math
import operator
import sys
import warnings

from pylearn2.utils import wraps
from pylearn2.compat import OrderedDict
from pylearn2.models.mlp import Layer


class PretrainedLayerBlock(Layer):

    """
    A block of layers whose weights are initialized, and optionally fixed,
    based on prior training.

    Parameters
    ----------
    block_name : string
      Name of the layers block
    model_content : Model 
    block_output_layer : integer
	Specifies the last layer of the model to be used to initialize the block
    freeze_params: bool
        If True, regard layer_conent's parameters as fixed
        If False, they become parameters of this layer and can be
        fine-tuned to optimize the MLP's cost function.
    """

    def __init__(self, block_name, model_content, block_output_layer, freeze_params=False):
               
        # Check that block_output_layer is within the proper range
        assert (block_output_layer >= 0)
        assert(block_output_layer < len(model_content.layers))
        
        # Select pretrained layer block
        del model_content.layers[block_output_layer+1:]
        
        # Rename block layers
        for i in range(len(model_content.layers)):
	  model_content.layers[i].layer_name = block_name + str(i)
	 
	layer_name = block_name
	layer_content = model_content

        super(PretrainedLayerBlock, self).__init__()
        self.__dict__.update(locals())
        del self.self


    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        assert self.get_input_space() == space

    @wraps(Layer.get_params)
    def get_params(self):

        if self.freeze_params:
            return []
        return self.layer_content.get_params()

    @wraps(Layer.get_input_space)
    def get_input_space(self):

        return self.layer_content.get_input_space()

    @wraps(Layer.get_output_space)
    def get_output_space(self):

        return self.layer_content.get_output_space()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        return OrderedDict([])      

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):

        if not hasattr(self.layer_content, "input_space"):
            raise AttributeError("Input space has not been provided.")

        rval = self.layer_content.layers[0].fprop(state_below)

        rlist = [rval]

        for layer in self.layer_content.layers[1:]:
            rval = layer.fprop(rval)
            rlist.append(rval)

        if return_all:
            return rlist
        return rval
