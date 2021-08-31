import torch
from torch import nn
from .residual import ResidualFC
from .representation import RepresentationBlock
from .head import PastHead2, LILSTMCell2
from .combine import CombineRepresentation, CombineHead
from .. import util as ModelUtil
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from .. import attr as Attribution
from contextlib import suppress
from ..util import LINE_PROFILER


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
                hidden_size if pre_shared_representation is not None else input_size,
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
        # Input is of linear shape (Batch, Length, Channel)
        x = ModelUtil.linear_to_conv1d_tensor(x)

        x_private = self.private_representation(x)
        x_private = ModelUtil.conv1d_to_linear_tensor(x_private)

        x_shared = None
        if self.use_shared_representation:
            x_shared = x
            if self.pre_shared_representation is not None:
                x_shared = self.pre_shared_representation(x_shared)
            x_shared = self.shared_representation(x_shared)
            x_shared = ModelUtil.conv1d_to_linear_tensor(x_shared)

            x_private = self.combine_representation(x_private, x_shared)
            return x_private, x_shared
        else:
            return x_private

    def freeze_shared(self, freeze=True):
        if self.use_shared_representation:
            self.shared_representation.requires_grad_(not freeze)

    def freeze_private(self, freeze=True):
        self.private_representation.requires_grad_(not freeze)
        if self.use_shared_representation:
            if self.pre_shared_representation is not None:
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
            private_head = PastHead2(
                input_size_past if not use_representation else (hidden_size_past * (2 if self.use_shared_representation is not None else 1)),
                private_state_size,
                **private_head
            )
        self.private_head = private_head

        if isinstance(shared_head, dict):
            shared_head = PastHead2(
                input_size_past if not use_representation else hidden_size_past,
                shared_state_size,
                **shared_head
            )
        self.shared_head = shared_head

    def forward(self, x, return_cx=False):
        x_private, x_shared = x, x
        if self.use_representation:
            if self.use_shared_representation:
                x_private, x_shared = self.representation_model(x)
            else:
                x_private = self.representation_model(x)
        x_private = ModelUtil.linear_to_sequential_tensor(x_private)

        hx_private = self.private_head(x_private, return_cx=return_cx)
        if self.use_shared_head:
            x_shared = ModelUtil.linear_to_sequential_tensor(x_shared)
            hx_shared = self.shared_head(x_shared, return_cx=return_cx)
        else:
            hx_shared = (None, None) if return_cx else None

        # hx stays in sequential shape
        if return_cx:
            if self.use_shared_head:
                return hx_private[0], hx_private[1], hx_shared[0], hx_shared[1]
            else:
                return hx_private[0], hx_private[1]
        else:
            if self.use_shared_head:
                return hx_private, hx_shared
            return hx_private


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
        seed_length=30,
        future_length=14,
        past_model={},
        representation_future_model={},
        private_head_future_cell={},
        shared_head_future_cell=None,
        post_future_model={},
        teacher_forcing=True,
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
            private_head_future_cell = LILSTMCell2(
                input_size_future if not use_representation_future else (hidden_size_future * (2 if use_shared_representation_future is not None else 1)),
                private_state_size,
                **private_head_future_cell
            )
        self.private_head_future_cell = private_head_future_cell

        if isinstance(shared_head_future_cell, dict):
            shared_head_future_cell = LILSTMCell2(
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

        self.seed_length = seed_length
        self.future_length = future_length
        self.teacher_forcing = teacher_forcing
        self.use_exo = use_exo

    def prepare_seed(self, past_seed_full, o=None, o_exo=None, seed_length=None):
        # past_seed_full is of sequential shape (Length, Batch, Channel)
        # o and o_exo is of sequential item shape (Batch, Channel)
        if o is not None:
            # o = o.detach()
            if o_exo is not None:
                o = torch.cat([o, o_exo], dim=o.dim() - 1)
            past_seed_full = torch.cat([past_seed_full, torch.stack([o])], dim=0)
        seed_length = seed_length or self.seed_length
        past_seed_full = past_seed_full[:seed_length]
        x_private, x_shared = past_seed_full, past_seed_full
        if self.use_representation_future:
            # print("prepare_seed", "past_seed_full", past_seed_full.size())
            past_seed_full = ModelUtil.sequential_to_linear_tensor(past_seed_full)
            if self.use_shared_representation_future:
                x_private, x_shared = self.representation_future_model(past_seed_full)
            else:
                x_private = self.representation_future_model(past_seed_full)
            past_seed_full = ModelUtil.linear_to_sequential_tensor(past_seed_full)
            x_private = ModelUtil.linear_to_sequential_tensor(x_private)
            x_shared = ModelUtil.linear_to_sequential_tensor(x_shared)
        return past_seed_full, x_private[-1], x_shared[-1]

    def forward(self, past, past_seed, past_exo=None, future=None, future_exo=None):
        # if self.use_exo:
        #     x_past = torch.cat(x_past, input["past_exo"])
        # hx_private, cx_private, hx_shared, cx_shared = self.past_model(past)
        hx_private, hx_shared = None, None
        if self.past_model.use_shared_head:
            hx_private, hx_shared = self.past_model(past)
        else:
            hx_private = self.past_model(past)

        teacher_forcing = self.teacher_forcing and self.training
        future = ModelUtil.linear_to_sequential_tensor(future) if teacher_forcing else None

        if self.use_exo:
            future_exo = ModelUtil.linear_to_sequential_tensor(future_exo)

        if teacher_forcing or future is not None:
            assert teacher_forcing and future is not None

        past_seed = ModelUtil.linear_to_sequential_tensor(past_seed)
        if self.use_exo:
            past_exo = ModelUtil.linear_to_sequential_tensor(past_exo)
            past_seed_full = torch.cat([past_seed, past_exo], dim=past_seed.dim() - 1)
        else:
            past_seed_full = past_seed

        outputs = []
        cx_private, cx_shared, o, o_exo = None, None, None, None
        last = self.future_length - 1
        for i in range(self.future_length):
            # print("for", "past_seed_full", past_seed_full.size())
            past_seed_full, x_private, x_shared = self.prepare_seed(
                past_seed_full,
                o,
                o_exo
            )

            """
            if cx_private is not None:
                print(x_private.size(), hx_private.size(), cx_private.size())
            else:
                print(x_private.size(), hx_private.size())
            """

            if i < last:
                cx_private, hx_private = self.private_head_future_cell(
                    x_private,
                    hx_private, cx_private,
                    return_reversed=True
                )
                if self.use_shared_head:
                    cx_shared, hx_shared = self.shared_head_future_cell(
                        x_shared,
                        hx_shared, cx_shared,
                        return_reversed=True
                    )
            else:
                cx_private = self.private_head_future_cell(
                    x_private,
                    hx_private, cx_private,
                    return_hx=False
                )
                if self.use_shared_head:
                    cx_shared = self.shared_head_future_cell(
                        x_shared,
                        hx_shared, cx_shared,
                        return_hx=False
                    )

            o = self.post_future_model(cx_private, cx_shared)
            outputs.append(o)

            if teacher_forcing:
                o = future[i]
            else:
                o = o.detach()
            if self.use_exo:
                o_exo = future_exo[i]

        ret = ModelUtil.sequential_to_linear_tensor(torch.stack(outputs))
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

    def get_summary(self, sample):
        return summary(self, input_data=sample)

    def write_graph(self, path, sample):
        self.summary_writer = SummaryWriter(path)
        self.eval()
        self.summary_writer.add_graph(self, input_to_model=sample)
        self.summary_writer.close()

    def __weight_kwargs_default(self, kwargs):
        if "teacher_forcing" not in kwargs:
            kwargs["teacher_forcing"] = self.teacher_forcing
        if "use_exo" not in kwargs:
            kwargs["use_exo"] = self.use_exo
        if "use_seed" not in kwargs:
            kwargs["use_seed"] = True
        return kwargs

    def get_input_attr(self, sample, *args, **kwargs):
        kwargs = self.__weight_kwargs_default(kwargs)
        return Attribution.calc_input_attr(self, sample, *args, **kwargs)

    def get_layer_attr(self, layer, sample, *args, **kwargs):
        if not layer:
            return None
        kwargs = self.__weight_kwargs_default(kwargs)
        return Attribution.calc_layer_attr(self, layer, sample, *args, **kwargs)

    def get_aggregate_layer_attr(self, sample):
        layer_attrs = {}
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["past_model.private_representation"] = self.get_layer_attr(self.past_model.representation_model.private_representation, sample)
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["past_model.shared_representation"] = self.get_layer_attr(self.past_model.representation_model.shared_representation, sample)
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["past_model.private_head"] = self.get_layer_attr(self.past_model.private_head, sample, labels=["hx", "cx"])
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["past_model.shared_head"] = self.get_layer_attr(self.past_model.shared_head, sample, labels=["hx", "cx"])
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["past_model"] = self.get_layer_attr(self.past_model, sample)
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["future_model.private_representation"] = self.get_layer_attr(self.representation_future_model.private_representation, sample)
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["future_model.shared_representation"] = self.get_layer_attr(self.representation_future_model.shared_representation, sample)
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["future_model.private_head"] = self.get_layer_attr(self.private_head_future_cell, sample, labels=["cx", "hx"])
        with suppress(KeyError, TypeError, AttributeError):
            layer_attrs["future_model.shared_head"] = self.get_layer_attr(self.shared_head_future_cell, sample, labels=["cx", "hx"])
        print([k for k, v in layer_attrs.items() if v is None])
        layer_attrs = Attribution.aggregate_layer_attr(layer_attrs)
        return layer_attrs
