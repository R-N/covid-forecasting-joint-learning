import torch
from captum.attr import IntegratedGradients

def filter_args(args, tf=True, exo=True, seed=True):
    i = iter(args)
    return (
        next(i),
        next(i) if seed else None,
        next(i) if exo and seed else None, 
        next(i) if tf else None,
        next(i) if exo else None
    )


LABELS = ["past", "past_seed", "past_exo", "future", "future_exo"]


def get_result_label(*f_args, **f_kwargs):
    return filter_args(LABELS, *f_args, **f_kwargs)


def wrap_params(model, *f_args, **f_kwargs):
    def model_param(*args):
        return model(*filter_args(args, *f_args, **f_kwargs))
    return model_param

def wrap_sum(model):
    def sum_model(*inputs):
        output = model(*inputs)
        return torch.sum(output, dim=0)
    return sum_model

def detach_tuple(tup):
    return tuple(x.detach() for x in tup)

def select_tuple(tup, indices=(0, 3)):
    if indices is None:
        return tup
    return tuple(tup[i] for i in indices)

def prepare_batch(batch):
    for t in batch:
        t.grad = None
        t.requires_grad_()
    return batch

def calc_input_importance(
    model,
    batch,
    method=IntegratedGradients,
    tf=True,
    exo=True,
    seed=True,
    single=True,
    out_dim=3
):
    if not (tf and exo and seed):
        model = wrap_params(model, tf=tf, exo=exo, seed=seed)

    model = wrap_sum(model)
    if single:
        model = wrap_sum(model)
    ig = method(model)

    batch = filter_args(batch, tf=tf, exo=exo, seed=seed)
    labels = get_result_label(tf=tf, exo=exo, seed=seed)
    if single:
        attr = detach_tuple(ig.attribute(prepare_batch(batch)))
    else:
        attr = [detach_tuple(ig.attribute(prepare_batch(batch), target=i)) for i in range(out_dim)]
        attr = list(zip(*attr))
    return dict(zip(labels, attr))
