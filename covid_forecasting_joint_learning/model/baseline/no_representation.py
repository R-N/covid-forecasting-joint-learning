from ..general import SourcePick, SharedMode, make_objective as __make_objective, eval as __eval

def make_objective(*args, **kwargs):
    return __make_objective(
        *args,
        use_representation_past=False,
        **kwargs
    )

def eval(*args, **kwargs):
    return __eval(
        *args,
        use_representation_past=False,
        **kwargs
    )
