from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import theano.tensor as T
from theano import function
import time


# Parse arguments
_, model_path, batch_size = sys.argv
batch_size = int(batch_size)


# Set variables
#batch_size = 100

# Load teacher model
model = serial.load(model_path)
model.set_batch_size(batch_size)


# Load dataset
src = model.dataset_yaml_src
assert src.find('train') != -1
test = yaml_parse.load(src)
test = test.get_test_set()
test.X = test.X.astype('float32')

assert test.X.shape[0] % batch_size == 0


def compute_test_accuracy(model):
    test_acc = []
    
    Xb = model.get_input_space().make_batch_theano()
    Xb.name = 'Xb'
    yb = model.get_output_space().make_batch_theano()
    yb.name = 'yb'
    
    y_model = model.fprop(Xb)
    label = T.argmax(yb,axis=1)
    prediction = T.argmax(y_model,axis=1)
    acc_model = 1.-T.neq(label , prediction).mean()
    
    batch_acc = function([Xb,yb],[acc_model])
        
    iterator = test.iterator(mode = 'even_sequential',
                            batch_size = batch_size,
                            data_specs = model.cost_from_X_data_specs())
    start_time_model = time.time()                       
    for item in iterator:
        x_arg, y_arg = item	  
        test_acc.append(batch_acc(x_arg, y_arg)[0])
    elapsed_time = time.time() - start_time_model
    
    return [sum(test_acc) / float(len(test_acc)), elapsed_time]

# Evaluate teacher

[acc_model, elapsed_time] = compute_test_accuracy(model)

error = 1. - acc_model 


# Print results
print 'Model achieved %f error in %fs' % (error, elapsed_time)

