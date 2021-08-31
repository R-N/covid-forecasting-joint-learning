from ..general import SourcePick, SharedMode, make_objective as __make_objective

def make_objective(*args, **kwargs):
    return __make_objective(
        *args,
        source_pick=SourcePick.LONGEST
    )
