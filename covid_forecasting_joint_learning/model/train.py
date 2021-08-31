import torch
from torch import nn
import contextlib
from .loss import MSSELoss
from .util import LINE_PROFILER


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
            loss_s = loss_fn(sample[3], pred)
            weight = kabko.weight
            loss += weight * loss_s

            if kabko.is_target:
                print("is_target", kabko.is_target)
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
            clip_grad_norm()

    print("target_losses", target_losses)
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
    """
    shortest = min(members, key=lambda k: len(key(k).dataset))
    size = len(key(shortest).dataset)
    """

    weights, target_weights = prepare_kabkos(sources, targets, source_weight, train=train)

    avg_loss = 0
    avg_target_loss = 0
    avg_target_losses = [0 for i in range(len(targets))]

    joint_dataloader_enum = list(zip(*[key(k) for k in members]))
    size = len(joint_dataloader_enum)
    # assert len(set([len(samples) for samples in joint_dataloader_enum])) == 1


    stepped = False

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
                grad_scaler=grad_scaler
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

def test(*args, **kwargs):
    return eval(*args, train=False, **kwargs)
