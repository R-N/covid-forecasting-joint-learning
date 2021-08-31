from ..general import SourcePick, SharedMode, make_objective as __make_objective

def make_objective(*args, **kwargs):
    return __make_objective(
        *args,
        shared_mode=SharedMode.PRIVATE,
        joint_learning=False
    )
