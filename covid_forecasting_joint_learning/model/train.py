import torch
from torch import nn
from .modules.main import SingleModel


def __train(samples, loss_fn, optimizer):
    optimizer.zero_grad()
    loss = 0
    
    for sample in samples:
        pred = sample["kabko"].model(sample)
        loss_s = loss_fn(sample["future"], pred)
        loss += sample["kabko"].weight * loss_s

        if sample["kabko"].is_target:
            target_loss = loss_s

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, target_loss


def train(
    sources,
    target,
    optimizer,
    loss_fn=nn.MSELoss(),
    source_weight=1.0,
    device="cpu",
    key=lambda k: k.dataloaders[0]
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

    for batch_id, samples in enumerate(joint_dataloader_enum):

        for sample in samples:
            sample["kabko"].model.freeze_private(True)
            sample["kabko"].model.freeze_shared(False)

        loss = 0
        target_loss = 0

        loss_s, target_loss_s = __train(samples, loss_fn, optimizer)
        loss += loss_s
        target_loss += target_loss_s

        for sample in samples:
            sample["kabko"].model.freeze_private(False)
            sample["kabko"].model.freeze_shared(True)

        loss_s, target_loss_s = __train(samples, loss_fn, optimizer)
        loss += loss_s
        target_loss += target_loss_s

        loss /= 2.0
        loss /= 1 + ((len(samples)-1) * source_weight)
        target_loss /= 2.0

        avg_loss += loss
        avg_target_loss += target_loss

    avg_loss /= size
    avg_target_loss /= size
    return avg_loss, avg_target_loss


def test(
    sources,
    target,
    loss_fn=nn.MSELoss(),
    source_weight=1.0,
    device="cpu",
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
