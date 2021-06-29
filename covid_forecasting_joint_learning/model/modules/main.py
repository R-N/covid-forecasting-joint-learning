import torch
from torch import nn
from .residual import ResidualFC
from .representation import RepresentationBlock
from .head import PastHead, LILSTMCell
from .combine import CombineRepresentation, CombineHead
from .. import util as ModelUtil


class RepresentationModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        private_representation={},
        pre_shared_representation={},
        shared_representation=None,
        combine_representation={},
    ):
        super(RepresentationModel, self).__init__()

        self.use_shared_representation = False
        if shared_representation is not None\
            or combine_representation is not None\
            or pre_shared_representation is not None:
            assert shared_representation is not None\
                and combine_representation is not None  # \
                # and pre_shared_representation is not None
            self.use_shared_representation = True


        if isinstance(private_representation, dict):
            private_representation = RepresentationBlock(
                input_size,
                hidden_size,
                **private_representation
            )
        self.private_representation = private_representation

        if isinstance(pre_shared_representation, dict):
            pre_shared_representation = RepresentationBlock(
                input_size,
                hidden_size,
                **pre_shared_representation
            )
        self.pre_shared_representation = pre_shared_representation

        if isinstance(shared_representation, dict):
            shared_representation = RepresentationBlock(
                hidden_size,
                hidden_size,
                **shared_representation
            )
        self.shared_representation = shared_representation

        if isinstance(combine_representation, dict):
            combine_representation = CombineRepresentation(
                hidden_size,
                **combine_representation
            )
        self.combine_representation = combine_representation

    def forward(self, x):
        x_private = self.private_representation(x)
        x_shared = None
        if self.use_shared_representation:
            x_shared = x if self.pre_shared_representation is None else self.pre_shared_representation(x)
            x_shared = self.shared_representation(x_shared)
            x_private = self.combine_representation(x_private, x_shared)
        return x_private, x_shared

    def freeze_shared(self, freeze=True):
        if self.use_shared_representation:
            self.shared_representation.requires_grad_(not freeze)

    def freeze_private(self, freeze=True):
        self.private_representation.requires_grad_(not freeze)
        if self.use_shared_representation:
            self.pre_shared_representation.requires_grad_(not freeze)
            self.combine_representation.requires_grad_(not freeze)


class PastModel(nn.Module):
    def __init__(
        self,
        input_size_past,
        hidden_size_past,
        private_state_size,
        shared_state_size,
        representation_model={},
        private_head={},
        shared_head=None
    ):
        super(PastModel, self).__init__()
        
        use_representation = False
        if hidden_size_past\
            or representation_model is not None:
            assert hidden_size_past\
                and representation_model is not None
            use_representation = True
        self.use_representation = use_representation

        use_shared_head = False
        if shared_state_size\
            or shared_head is not None:
            assert shared_state_size\
                and shared_head is not None
            use_shared_head = True
        self.use_shared_head = use_shared_head


        self.use_shared_representation = False
        self.representation_model = None
        if use_representation:
            self.representation_model = RepresentationModel(
                input_size_past,
                hidden_size_past,
                **representation_model
            )
            self.use_shared_representation = self.representation_model.use_shared_representation

        if isinstance(private_head, dict):
            private_head = PastHead(
                input_size_past if not use_representation else (hidden_size_past * (2 if self.use_shared_representation is not None else 1)),
                private_state_size,
                **private_head
            )
        self.private_head = private_head

        if isinstance(shared_head, dict):
            shared_head = PastHead(
                input_size_past if not use_representation else hidden_size_past,
                shared_state_size,
                **shared_head
            )
        self.shared_head = shared_head

    def forward(self, x):
        if self.use_representation:
            x_private, x_shared = self.representation_model(x)
        else:
            x_private, x_shared = x, x
        hx_private = self.private_head(x_private)
        if self.use_shared_head:
            hx_shared = self.shared_head(x_shared)
        else:
            hx_shared = None
        return hx_private, hx_shared

    def freeze_shared(self, freeze=True):
        if self.use_representation:
            self.representation_model.freeze_shared(freeze)
        if self.use_shared_head:
            self.shared_head.requires_grad_(not freeze)

    def freeze_private(self, freeze=True):
        if self.use_representation:
            self.representation_model.freeze_private(freeze)
        self.private_head.requires_grad_(not freeze)


class PostFutureModel(nn.Module):
    def __init__(
        self,
        private_state_size,
        shared_state_size,
        output_size,
        private_head_future={},
        shared_head_future=None,
        combine_head={}
    ):
        super(PostFutureModel, self).__init__()
        
        use_shared_head = False
        if shared_state_size\
            or shared_head_future is not None\
            or combine_head is not None:
            assert shared_state_size\
                and shared_head_future is not None\
                and combine_head is not None
            use_shared_head = True
        self.use_shared_head = use_shared_head

        if isinstance(private_head_future, dict):
            private_head_future = ResidualFC(
                private_state_size,
                output_size,
                **private_head_future
            )
        self.private_head_future = private_head_future

        if isinstance(shared_head_future, dict):
            shared_head_future = ResidualFC(
                shared_state_size,
                output_size,
                **shared_head_future
            )
        self.shared_head_future = shared_head_future

        if isinstance(combine_head, dict):
            combine_head = CombineRepresentation(
                output_size,
                **combine_head
            )
        self.combine_head = combine_head

    def forward(self, x_private, x_shared=None):
        if self.use_shared_head or x_shared is not None:
            assert self.use_shared_head and x_shared is not None
        x_private = self.private_head_future(x_private)
        if self.use_shared_head:
            x_shared = self.shared_head_future(x_shared)
            x = self.combine_head(x_private, x_shared)
        else:
            x = x_private
        return x

    def freeze_shared(self, freeze=True):
        if self.use_shared_head:
            self.shared_head_future.requires_grad_(not freeze)

    def freeze_private(self, freeze=True):
        if self.use_shared_head:
            self.combine_head.requires_grad_(not freeze)
        self.private_head_future.requires_grad_(not freeze)


class SingleModel(nn.Module):
    # Blocks have to be kwargs, None, or the actual block
    def __init__(
        self,
        input_size_past,
        hidden_size_past,
        input_size_future,
        hidden_size_future,
        private_state_size,
        shared_state_size,
        output_size,
        future_length=14,
        past_model={},
        representation_future_model={},
        private_head_future_cell={},
        shared_head_future_cell=None,
        post_future_model={},
        teacher_forcing=False,
        use_exo=True
    ):
        super(SingleModel, self).__init__()
        
        use_representation_future = False
        if hidden_size_future\
            or representation_future_model is not None:
            assert hidden_size_future\
                and representation_future_model is not None
            use_representation_future = True
        self.use_representation_future = use_representation_future

        self.past_model = PastModel(
            input_size_past,
            hidden_size_past,
            private_state_size,
            shared_state_size,
            **past_model
        )

        if isinstance(representation_future_model, dict):
            representation_future_model = RepresentationModel(
                input_size_future,
                hidden_size_future,
                **representation_future_model
            )
        self.representation_future_model = representation_future_model

        use_shared_representation_future = use_representation_future\
            and representation_future_model.use_shared_representation
        self.use_shared_representation_future = use_shared_representation_future

        if isinstance(private_head_future_cell, dict):
            private_head_future_cell = LILSTMCell(
                input_size_future if not use_representation_future else (hidden_size_past * (2 if use_shared_representation_future is not None else 1)),
                private_state_size,
                **private_head_future_cell
            )
        self.private_head_future_cell = private_head_future_cell

        if isinstance(shared_head_future_cell, dict):
            shared_head_future_cell = LILSTMCell(
                input_size_future if not use_representation_future else hidden_size_future,
                shared_state_size,
                **shared_head_future_cell
            )
        self.shared_head_future_cell = shared_head_future_cell


        use_shared_head = False
        if shared_state_size\
            or shared_head_future_cell is not None\
            or self.past_model.use_shared_head:
            assert shared_state_size\
                and shared_head_future_cell is not None\
                and self.past_model.use_shared_head
            use_shared_head = True
        self.use_shared_head = use_shared_head

        self.post_future_model = CombineHead(
            private_state_size,
            shared_state_size,
            output_size,
            **post_future_model
        )

        self.future_length = future_length
        self.teacher_forcing = teacher_forcing
        self.use_exo = use_exo

    def prepare_seed(self, past_seed_full, o=None, o_exo=None):
        if o is not None:
            o = o.detach()
            if o_exo is not None:
                o = torch.cat(o, o_exo, dim=o.dim()-1)
            past_seed_full = torch.cat([*past_seed_full, o])
        if self.use_representation_future:
            past_seed_full = ModelUtil.to_batch_tensor(past_seed_full)
            x_private, x_shared = self.representation_future_model(past_seed_full)
            past_seed_full = ModelUtil.to_sequential_tensor(past_seed_full)
            x_private = ModelUtil.to_sequential_tensor(x_private)
            x_shared = ModelUtil.to_sequential_tensor(x_shared)
        else:
            x_private, x_shared = past_seed_full, past_seed_full
        return past_seed_full, x_private[-1], x_shared[-1]

    def forward(self, input):
        x_past = input["past"]
        # if self.use_exo:
        #     x_past = torch.cat(x_past, input["past_exo"])
        hx_private, hx_shared = self.past_model(x_past)

        x_future = None
        teacher_forcing = self.teacher_forcing and self.training
        if teacher_forcing:
            x_future = ModelUtil.to_sequential_tensor(input["future"])

        if self.use_exo:
            x_future_exo = ModelUtil.to_sequential_tensor(input["future_exo"])

        hx_private, hx_shared = hx_private[0], hx_shared[0]

        if teacher_forcing or x_future is not None:
            assert teacher_forcing and x_future is not None

        past_seed = ModelUtil.to_sequential_tensor(input["past_seed"])
        if self.use_exo:
            past_exo = ModelUtil.to_sequential_tensor(input["past_exo"])
            past_seed_full = torch.cat([past_seed, past_exo], dim=past_seed.dim()-1)
        else:
            past_seed_full = past_seed

        outputs = []
        cx_private, cx_shared, o, o_exo = None, None, None, None
        for i in range(self.future_length):
            past_seed_full, x_private, x_shared = self.prepare_seed(
                past_seed_full,
                o,
                o_exo
            )

            hx_private, cx_private = self.private_head_future_cell(
                x_private,
                (hx_private, cx_private)
            )
            if self.use_shared_head:
                hx_shared, cx_shared = self.shared_head_future_cell(
                    x_shared,
                    (hx_shared, cx_shared)
                )

            o = self.post_future_model(cx_private, cx_shared)
            outputs.append(o)

            if teacher_forcing:
                o = x_future[i]
            if self.use_exo:
                o_exo = x_future_exo[i]

        ret = ModelUtil.to_batch_tensor(torch.stack(outputs))
        return ret

    def freeze_shared(self, freeze=True):
        self.past_model.freeze_shared(freeze)
        if self.use_representation_future:
            self.representation_future_model.freeze_shared(freeze)
        if self.use_shared_head:
            self.shared_head_future_cell.requires_grad_(not freeze)
        self.post_future_model.freeze_shared(freeze)

    def freeze_private(self, freeze=True):
        self.past_model.freeze_private(freeze)
        if self.use_representation_future:
            self.representation_future_model.freeze_private(freeze)
        self.private_head_future_cell.requires_grad_(not freeze)
        self.post_future_model.freeze_shared(freeze)
