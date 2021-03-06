import torch
from torch import nn
import contextlib
from .loss import MSSELoss
from .loss_common import rmsse, reduce
from .util import LINE_PROFILER
import numpy as np
from ..pipeline import sird


dummy_context = contextlib.nullcontext()


def dummy():
    pass


def __eval(
    samples,
    loss_fn,
    weights,
    target_weights,
    optimizer=None,
    train=False,
    clip_grad_norm=None,
    grad_scaler=None,
    lr=None
):
    if train:
        optimizer.zero_grad(set_to_none=True)

    loss = 0
    target_loss = 0
    target_losses = {}

    context = torch.cuda.amp.autocast() if train and grad_scaler else dummy_context

    with context:
        for sample in samples:
            sample, kabko = sample[:-1], sample[-1]
            pred = kabko.model(*sample[:5])
            loss_s = loss_fn(sample[1], sample[3], pred)
            weight = kabko.weight
            loss += weight * loss_s

            if kabko.is_target:
                target_loss += loss_s
                target_losses[kabko.is_target] = loss_s

        loss /= weights
        target_loss /= target_weights

    if train:
        if grad_scaler:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()

        if clip_grad_norm:
            clip_grad_norm(lr=lr)

    target_losses = [target_losses[i + 1].detach().item() for i in range(len(target_losses))]
    return loss, target_loss, target_losses

def prepare_kabkos(sources, targets, source_weight=1.0, train=False):
    weights = 0
    target_weights = 0

    for source in sources:
        source.is_target = 0
        source.weight = source_weight
        weights += source.weight
        if train:
            source.model.train()
        else:
            source.model.eval()

    i = 0
    for target in targets:
        i += 1
        target.is_target = i
        target.weight = 1.0
        weights += target.weight
        target_weights += target.weight
        if train:
            target.model.train()
        else:
            target.model.eval()

    return weights, target_weights

def eval(
    sources,
    targets,
    train=True,
    optimizer=None,
    scheduler=None,
    loss_fn=MSSELoss(),
    source_weight=1.0,
    key=lambda k: k.dataloaders[0],
    clip_grad_norm=None,
    grad_scaler=None
):
    members = sources + targets
    shortest = min(members, key=lambda k: len(key(k).dataset))
    size = len(key(shortest).dataset)

    weights, target_weights = prepare_kabkos(sources, targets, source_weight, train=train)

    avg_loss = 0
    avg_target_loss = 0
    avg_target_losses = [0 for i in range(len(targets))]

    joint_dataloader_enum = list(zip(*[key(k) for k in members]))

    stepped = False

    lr = None
    if scheduler:
        lr = scheduler.get_last_lr()[0]
    context = dummy_context if train else torch.no_grad()
    with context:
        for batch_id, samples in enumerate(joint_dataloader_enum):
            loss_s, target_loss_s, target_losses = __eval(
                samples,
                loss_fn,
                weights,
                target_weights,
                train=train,
                optimizer=optimizer,
                clip_grad_norm=clip_grad_norm,
                grad_scaler=grad_scaler,
                lr=lr
            )

            avg_loss += loss_s
            avg_target_loss += target_loss_s
            avg_target_losses = [sum(x) for x in zip(avg_target_losses, target_losses)]

            if train:
                if grad_scaler:
                    scale = grad_scaler.get_scale()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    stepped = stepped or scale == grad_scaler.get_scale()
                else:
                    optimizer.step()
                    stepped = True

                optimizer.zero_grad(set_to_none=True)

    if train and scheduler and stepped:
        scheduler.step()

    avg_loss /= size
    avg_target_loss /= size
    avg_target_losses = [x / size for x in avg_target_losses]
    return avg_loss, avg_target_loss, avg_target_losses

def train(*args, **kwargs):
    return eval(*args, train=True, **kwargs)

def val(*args, **kwargs):
    return eval(*args, train=False, **kwargs)

def test(
    target,
    loss_fn=rmsse,
    reduction="mean",
    key=lambda k: k.dataloaders[-1]
):
    target.model.eval()
    dataloader = key(target)
    target_loss = 0
    n = target.population
    for batch_id, sample in enumerate(dataloader):
        # sample, kabko = sample[:-1], sample[-1]
        pred_vars = target.model(*sample[:5]).detach().numpy()
        prev, final = sample[5], sample[6]
        if isinstance(prev, torch.Tensor):
            prev, final = prev.numpy(), final.numpy()
        pred_final = target.model.rebuild(pred_vars, prev, n, sird.rebuild, scaler=target.scaler_2)
        losses = [loss_fn(
            prev[i][:, 1:],
            final[i],
            pred_final[i]
        ) for i in range(len(pred_final))]
        losses = np.stack(losses)

        target_loss += reduce(losses, reduction=reduction, reduce_feature=False)
    return target_loss


def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# https://github.com/pseeth/autoclip
def autoclip_gradient(model, grad_history, clip_percentile=10, min_clip=1, max_clip=10):
    obs_grad_norm = _get_grad_norm(model)
    grad_history.append(obs_grad_norm)
    clip_value = np.percentile(grad_history, clip_percentile)
    # clip_value_0 = clip_value
    if max_clip:
        clip_value = min(max_clip, clip_value)
    if min_clip:
        clip_value = max(min_clip, clip_value)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
