import torch
from torch import nn
from .. import util as ModelUtil


# Learnable Init LSTM
class LILSTMCell(nn.Module):

    def __init__(
        self,
        input_size,
        state_size
    ):
        super(LILSTMCell, self).__init__()

        self.cell = nn.LSTMCell(input_size, state_size)

        self.hx_0 = nn.Parameter(ModelUtil.learnable_xavier((state_size,)))
        self.cx_0 = nn.Parameter(ModelUtil.learnable_xavier((state_size,)))

    def forward(self, x, hx=None):
        # Input of dimension (Batch, Channel)
        batch_size = x.size(0)
        hx, cx = (None, None) if hx is None else hx

        hx = ModelUtil.repeat_batch(self.hx_0, batch_size) if hx is None else hx
        cx = ModelUtil.repeat_batch(self.cx_0, batch_size) if cx is None else cx
        
        hx, cx = self.cell(x, (hx, cx))

        return hx, cx


class PastHead(nn.Module):
    def __init__(
        self,
        input_size,
        state_size,
        use_last_past=False
    ):
        super(PastHead, self).__init__()

        self.cell = LILSTMCell(input_size, state_size)

        self.use_last_past = use_last_past
        self.state_size = state_size

    def forward(self, x):

        hx, cx = None, None

        past_length = x.size(0)
        past_length = past_length if self.use_last_past else past_length - 1

        for i in range(past_length):
            hx, cx = self.cell(x[i], (hx, cx))

        return hx, cx


class FutureSingle:
    def __init__(self, code, obj):
        self.code = code
        self.obj = obj

class FutureHead(nn.Module):
    PRE_FUTURE = 0
    POST_FUTURE = 1
    DONE = 2

    def __init__(
        self,
        input_size,
        state_size,
        future_length=14
    ):
        super(FutureHead, self).__init__()

        self.cell = LILSTMCell(input_size, state_size)

        self.state_size = state_size

    def forward(self, x, hx, cx=None):
        batch_size = x.size(0)
        # Eh, whatever. I'm not using this
        x = ModelUtil.to_sequential_tensor(x)

        outputs = []
        future_seed = input["future_seed"]
        for i in range(self.future_length):
            wrapper = FutureSingle(FutureHead.PRE_FUTURE, ModelUtil.to_batch_tensor(cx))
            yield wrapper
            cx = ModelUtil.to_sequential_tensor(wrapper.obj)
            hx, cx = self.future_cell(
                x,
                (hx, cx)
            )
            wrapper = FutureSingle(FutureHead.POST_FUTURE, ModelUtil.to_batch_tensor(cx))
            yield wrapper
            cx = ModelUtil.to_sequential_tensor(wrapper.obj)
            outputs.append(cx)

        ret = ModelUtil.to_batch_tensor(torch.stack(outputs))
        yield FutureSingle(FutureHead.DONE, ret)


class Head(nn.Module):
    def __init__(
        self,
        past_input_size,
        future_input_size,
        state_size,
        future_length=14,
        use_last_past=False,
        teacher_forcing=False
    ):
        super(Head, self).__init__()

        self.past_block = PastHead(
            past_input_size,
            state_size,
            use_last_past=use_last_past
        )
        self.future_block = FutureHead(
            future_input_size,
            state_size,
            future_length=future_length
        )
        self.teacher_forcing = teacher_forcing

    def forward(self, x_past, x_future):
        hx, cx = self.past_block(input)
        cx = None
        input["hx"] = hx
        input["cx"] = cx

        future_iter = self.future_cell(input)
        ret = None
        while True:
            try:
                ret = next(future_iter)
            except StopIteration as ex:
                break
            if ret.code == FutureHead.PRE_FUTURE:
                pass
            elif ret.code == FutureHead.POST_FUTURE:
                pass
            if ret.code == FutureHead.DONE:
                break

        return FutureSingle(FutureHead.DONE, obj)
