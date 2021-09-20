from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR

class OneCycleLR:
    def __init__(self, optimizer, lr, steps_per_epoch, epochs):
        self.optimizer = optimizer
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = epochs
        self.epochs = 0
        self.scheduler = None
        self.create()

    def create(self):
        self.scheduler = _OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs)
        return self.scheduler

    def step(self, step=1):
        if self.epochs + step > self.max_epochs:
            self.create()
        return self.scheduler.step()
