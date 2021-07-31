import torch
from torch import nn
from .modules.main import SingleModel


def __train(samples, loss_fn, optimizer, clip_grad_norm=None, grad_scaler=None):
    optimizer.zero_grad(set_to_none=True)
    loss = 0

    weights = 0
    
    with torch.cuda.amp.autocast():
        for sample in samples:
            pred = sample["kabko"].model(sample)
            loss_s = loss_fn(sample["future"], pred)
            weight = sample["kabko"].weight
            loss += weight * loss_s
            weights += weight

            if sample["kabko"].is_target:
                target_loss = loss_s

    loss /= weights

    if grad_scaler:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
    else:
        loss.backward()

    if clip_grad_norm:
        clip_grad_norm()

    return loss, target_loss


def train(
    sources,
    target,
    optimizer,
    scheduler=None,
    loss_fn=nn.MSELoss(),
    source_weight=1.0,
    key=lambda k: k.dataloaders[0],
    clip_grad_norm=None
):
    members = [*sources, target]
    shortest = min(members, key=lambda k: len(key(k).dataset))
    size = len(key(shortest).dataset)

    for source in sources:
        source.is_target = False
        source.weight = source_weight
        source.model.train()

    target.is_target = True
    target.weight = 1.0
    target.model.train()

    avg_loss = 0
    avg_target_loss = 0

    joint_dataloader_enum = zip(*[key(k) for k in members])

    grad_scaler = torch.cuda.amp.GradScaler()

    for batch_id, samples in enumerate(joint_dataloader_enum):
        loss = 0
        target_loss = 0

        loss_s, target_loss_s = __train(samples, loss_fn, optimizer, clip_grad_norm, grad_scaler)
        loss += loss_s
        target_loss += target_loss_s

        if grad_scaler:
            grad_scaler.step(optimizer).step()
        else:
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)

        grad_scaler.update()

        avg_loss += loss
        avg_target_loss += target_loss

    if scheduler:
        scheduler.step()
        
    avg_loss /= size
    avg_target_loss /= size
    return avg_loss, avg_target_loss


def test(
    sources,
    target,
    loss_fn=nn.MSELoss(),
    source_weight=1.0,
    key=lambda k: k.dataloaders[1]
):
    members = [*sources, target]
    shortest = min(members, key=lambda k: len(key(k).dataset))
    size = len(key(shortest).dataset)

    for source in sources:
        source.is_target = False
        source.weight = source_weight
        source.model.eval()

    target.is_target = True
    target.weight = 1.0
    target.model.eval()

    avg_loss = 0
    avg_target_loss = 0

    with torch.no_grad():
        joint_dataloader_enum = zip(*[key(k) for k in members])

        for batch_id, samples in enumerate(joint_dataloader_enum):
            loss = 0
            
            for sample in samples:
                pred = sample["kabko"].model(sample)
                loss_s = loss_fn(sample["future"], pred)
                loss += sample["kabko"].weight * loss_s

                if sample["kabko"].is_target:
                    target_loss = loss_s

            loss /= 1 + ((len(samples)-1) * source_weight)

            avg_loss += loss
            avg_target_loss += target_loss

    avg_loss /= size
    avg_target_loss /= size
    return avg_loss, avg_target_loss
