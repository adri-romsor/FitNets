import theano.tensor as T
import cPickle as pkl
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import TanhConvNonlinearity, SigmoidConvNonlinearity
from pylearn2.models.mlp import ConvElemwise as ConvElemwisePL2

class TeacherHintRegressionCost(DefaultDataSpecsMixin, Cost):
    """
    Represents an objective function to be minimized by some
    `TrainingAlgorithm`.
    """
    
    # If True, the data argument to expr and get_gradients must be a
    # (X, Y) pair, and Y cannot be None.
    supervised = False
    
    def __init__(self, teacher, hintlayer):  
    
      # Load teacher network.
      if isinstance(teacher, str):
	fo = open(teacher, 'r')
	teacher_model = pkl.load(fo)
	fo.close()
      else:
	teacher_model = teacher
	
      del teacher_model.layers[hintlayer+1:]

      self.teacher = teacher_model
      self.hintlayer = hintlayer

    def expr(self, model, data, ** kwargs):
        """
        Returns a theano expression for the cost function.
        
        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments. Not used by the base class.
        """
        
        space, sources = self.get_data_specs(model)
        space.validate(data)
        x = data
        
        axes = model.input_space.axes
	                    
        # Compute student output
        student_output = model.fprop(x)
        
        # Compute teacher output
        hint = x
        for l in range(self.hintlayer+1):
	  hint = self.teacher.layers[l].fprop(hint)
        
        # Change teacher format if non-convolutional regressor
	if hasattr(model.layers[-1].get_output_space(),'dim'):
	  hint = hint.reshape(shape=(hint.shape[axes.index('b')],
				      hint.shape[axes.index('c')]*
				      hint.shape[axes.index(0)]*
				      hint.shape[axes.index(1)]),ndim=2)
				      
				      
	# Transform output if necessary (only in tanh/sigmoid cases to use ce error instead of mse)
	if (isinstance(self.teacher.layers[self.hintlayer], ConvElemwisePL2)) and isinstance(self.teacher.layers[self.hintlayer].nonlinearity,TanhConvNonlinearity):
	  hint = (hint + 1) / float(2)
	  cost = -T.log(student_output) * hint
	  cost = T.sum(cost,axis=1)
	elif (isinstance(self.teacher.layers[self.hintlayer], ConvElemwisePL2)) and isinstance(self.teacher.layers[self.hintlayer].nonlinearity,SigmoidConvNonlinearity):
	  cost = -T.log(student_output) * hint
          cost = T.sum(cost,axis=1)
	else:
	  # Compute cost
	  cost = 0.5*(hint - student_output)**2
	  cost = T.sum(cost,axis=1)
        
        return T.mean(cost)
        
    def get_monitoring_channels(self, model, data, **kwargs):
        """
        .. todo::

            WRITEME

        .. todo::

            how do you do prereqs in this setup? (I think PL changed
            it, not sure if there still is a way in this context)

        Returns a dictionary mapping channel names to expressions for
        channel values.

        Parameters
        ----------
        model : Model
            the model to use to compute the monitoring channels
        data : batch
            (a member of self.get_data_specs()[0])
            symbolic expressions for the monitoring data
        kwargs : dict
            used so that custom algorithms can use extra variables
            for monitoring.

        Returns
        -------
        rval : dict
            Maps channels names to expressions for channel values.
        """
               	
	rval = OrderedDict()
			
        value_cost_wrt_teacher = self.expr(model,data)

        if value_cost_wrt_teacher is not None:
	   name = 'cost_wrt_teacher'
	   rval[name] = value_cost_wrt_teacher
	   
        return rval        



        


