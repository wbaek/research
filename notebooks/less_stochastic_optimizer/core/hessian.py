import tensorflow as tf
from functools import reduce
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import array_ops, tensor_array_ops, control_flow_ops

def hessians_highrank(ys, xs, gradients=None, name="hessians", colocate_gradients_with_ops=False,
            gate_gradients=False, aggregation_method=None):
  """Constructs the Hessian (one or more rank matrix) of sum of `ys` with respect to `x` in `xs`.
  `hessians_highrank()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`. This function currently
  only supports evaluating the Hessian with respect to (a list of) one-
  dimensional tensors.
  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.
  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.
  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
  xs = gradients_impl._AsList(xs)
  kwargs = {
    'colocate_gradients_with_ops': colocate_gradients_with_ops,
    'gate_gradients': gate_gradients,
    'aggregation_method': aggregation_method
  }
  # Compute first-order derivatives and iterate for each x in xs.
  hessians = []
  _gradients = tf.gradients(ys, xs, **kwargs) if gradients is None else gradients
  for i, _gradient, x in zip(range(len(xs)), _gradients, xs):
    shape = x.shape
    _gradient = tf.reshape(_gradient, [-1])
    
    n = tf.size(x)
    loop_vars = [
      array_ops.constant(0, tf.int32),
      tensor_array_ops.TensorArray(x.dtype, n)
    ]
    _, hessian = control_flow_ops.while_loop(
      lambda j, _: j < n,
      lambda j, result: (j + 1, result.write(j, tf.gradients(_gradient[j], x, **kwargs)[0])),
      loop_vars
    )
    hessians.append(hessian.stack())
  return hessians

def diagonal_inverse_hessians_highrank(ys, xs, gradients=None, name="hessians", colocate_gradients_with_ops=False,
            gate_gradients=False, aggregation_method=None):
  """Constructs the Hessian (one or more rank matrix) of sum of `ys` with respect to `x` in `xs`.
  `hessians_highrank()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`. This function currently
  only supports evaluating the Hessian with respect to (a list of) one-
  dimensional tensors.
  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.
  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.
  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
  xs = gradients_impl._AsList(xs)
  kwargs = {
    'colocate_gradients_with_ops': colocate_gradients_with_ops,
    'gate_gradients': gate_gradients,
    'aggregation_method': aggregation_method
  }
  # Compute first-order derivatives and iterate for each x in xs.
  hessians = []
  _gradients = tf.gradients(ys, xs, **kwargs) if gradients is None else gradients
  for i, _gradient, x in zip(range(len(xs)), _gradients, xs):
    shape = x.shape
    _gradient = tf.reshape(_gradient, [-1])
    
    n = tf.size(x)
    g = tf.gradients(_gradient, x, **kwargs)[0]
    hessian = tf.clip_by_value( 1.0 / tf.reshape(g, [-1]), -1e1, 1e1 )
    hessians.append(hessian)
  return hessians

