import torch
from captum.attr import Saliency, LayerGradCam

def filter_args(args, teacher_forcing=True, use_exo=True, use_seed=True, none=True):
    i = iter(args)
    ret = (
        next(i),
        next(i) if use_seed else None,
        next(i) if use_exo and use_seed else None, 
        next(i) if teacher_forcing else None,
        next(i) if use_exo else None
    )
    ret = ret if none else filter_none(ret)
    return ret

def filter_batch(batch, teacher_forcing=True, use_exo=True, use_seed=True, none=False):
    ret = (
        batch[0],
        batch[1] if use_seed else None,
        batch[2] if use_exo and use_seed else None, 
        batch[3] if teacher_forcing else None,
        batch[4] if use_exo else None
    )
    ret = ret if none else filter_none(ret)
    return ret

def filter_none(tup):
    return tuple(x for x in tup if x is not None)


LABELS = ["past", "past_use_seed", "past_exo", "future", "future_exo"]


def get_result_label(*args, **kwargs):
    return filter_batch(LABELS, *args, **kwargs)


def wrap_params(model, *f_args, **f_kwargs):
    def model_param(*args):
        return model(*filter_args(args, *f_args, none=True, **f_kwargs))
    return model_param

def wrap_sum(model):
    def sum_model(*inputs):
        output = model(*inputs)
        return torch.sum(output, dim=0)
    return sum_model

def detach_tuple(tup):
    return tuple(x.detach() for x in tup)

def postprocess_result(tup):
    ret = detach_tuple(tup)
    # ret = tuple(t[0] for t in ret)
    while ret[0].dim() > 1:
        ret = tuple(torch.sum(t, dim=0) / t.size(0) for t in ret)
    return ret

def select_tuple(tup, indices=(0, 3)):
    if indices is None:
        return tup
    return tuple(tup[i] for i in indices)

def prepare_batch(batch):
    for t in batch:
        if t is None:
            continue
        t.grad = None
        t.requires_grad_()
    return batch

def single_batch(t):
    return torch.stack([t[0]])

def __prepare_model(
    model,
    # teacher_forcing=True,
    # use_exo=True,
    # use_seed=True,
    single=True,
    **kwargs
):
    # if not (teacher_forcing and use_exo and use_seed):
    #     model = wrap_params(model, teacher_forcing=teacher_forcing, use_exo=use_exo, use_seed=use_seed)

    model = wrap_sum(model)
    if single:
        model = wrap_sum(model)
    return model

def __calc_weight(
    method,
    batch,
    teacher_forcing=True,
    use_exo=True,
    use_seed=True,
    single=True,
    out_dim=3
):
    batch = filter_batch(batch, teacher_forcing=teacher_forcing, use_exo=use_exo, use_seed=use_seed, none=True)
    # batch = tuple(single_batch(t) for t in batch)
    if single:
        attr = postprocess_result(method.attribute(prepare_batch(batch)))
    else:
        attr = [postprocess_result(method.attribute(prepare_batch(batch), target=i)) for i in range(out_dim)]
        attr = list(zip(*attr))
    return attr

def calc_input_weight(
    model,
    batch,
    method=Saliency,
    teacher_forcing=True,
    use_exo=True,
    use_seed=True,
    single=True,
    out_dim=3
):
    model = __prepare_model(
        model,
        teacher_forcing=teacher_forcing,
        use_exo=use_exo,
        use_seed=use_seed,
        single=single
    )
    method = method(model)
    attr = __calc_weight(
        method,
        batch,
        teacher_forcing=teacher_forcing,
        use_exo=use_exo,
        use_seed=use_seed,
        single=single,
        out_dim=3
    )
    labels = get_result_label(teacher_forcing=teacher_forcing, use_exo=use_exo, use_seed=use_seed, none=True)
    return dict(zip(labels, attr))


def calc_layer_weight(
    model,
    layer,
    batch,
    method=LayerGradCam,
    teacher_forcing=True,
    use_exo=True,
    use_seed=True,
    single=True,
    out_dim=3
):
    model = __prepare_model(
        model,
        teacher_forcing=teacher_forcing,
        use_exo=use_exo,
        use_seed=use_seed,
        single=single
    )
    method = method(model, layer)
    return __calc_weight(
        method,
        batch,
        teacher_forcing=teacher_forcing,
        use_exo=use_exo,
        use_seed=use_seed,
        single=single,
        out_dim=3
    )

