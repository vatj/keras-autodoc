from tensorflow.keras import Model
from keras_tuner import HyperParameters
from keras_autodoc.get_signatures import get_function_signature, get_signature_end


def test_signature():
    excpected = ('Model.compile(\n'
                 '    optimizer="rmsprop",\n'
                 '    loss=None,\n'
                 '    metrics=None,\n'
                 '    loss_weights=None,\n'
                 '    weighted_metrics=None,\n'
                 '    run_eagerly=None,\n'
                 '    steps_per_execution=None,\n'
                 '    jit_compile=None,\n'
                 '    **kwargs\n'
                 ')')
    computed = get_function_signature(Model.compile)
    assert computed == excpected


def test_wrapping_signature():
    expected = '(parent_name, parent_values)'
    computed = get_signature_end(HyperParameters.conditional_scope)
    assert computed == expected
