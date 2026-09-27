is_simple_core = False

if is_simple_core:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import Parameter
    from dezero.models import Model
    from dezero.dataloaders import DataLoader
    from dezero.core import test_mode
else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import Parameter
    from dezero.models import Model
    from dezero.dataloaders import DataLoader
    from dezero.core import test_mode
setup_variable()