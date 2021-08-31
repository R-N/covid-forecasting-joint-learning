from ..general import SourcePick, SharedMode, make_objective as __make_objective

def make_objective(*args, **kwargs):
    return __make_objective(
        *args,
        use_representation_past=False,
        use_shared=False,
        joint_learning=False,
        **kwargs
    )
