from ..general import SourcePick, SharedMode, make_objective as __make_objective, eval as __eval

def make_objective(*args, **kwargs):
    return __make_objective(
        *args,
        source_pick=SourcePick.LONGEST,
        **kwargs
    )

def eval(*args, **kwargs):
    return __eval(
        *args,
        source_pick=SourcePick.LONGEST,
        **kwargs
    )
