"""
copy from facebookresearch fvcore
"""
import logging
import warnings
import typing
from collections import Counter, OrderedDict, defaultdict
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from typing import Dict, Set, Tuple, TypeVar, Iterable, Iterator, Counter, DefaultDict
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.jit import _get_trace_graph, TracerWarning
from dataclasses import dataclass
from copy import copy

try:
    from math import prod
except ImportError:
    from numpy import prod
    
Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]


def get_shape(val: Any) -> Optional[List[int]]:
    """
    Get the shapes from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None


"""
Below are flop/activation counters for various ops. Every counter has the following signature:

Args:
    inputs (list(torch._C.Value)): The inputs of the op in the form of a list of jit object.
    outputs (list(torch._C.Value)): The outputs of the op in the form of a list of jit object.

Returns:
    number: The number of flops/activations for the operation.
    or Counter[str]
"""


def generic_activation_jit(op_name: Optional[str] = None) -> Handle:
    """
    This method return a handle that counts the number of activation from the
    output shape for the specified operation.

    Args:
        op_name (str): The name of the operation. If given, the handle will
            return a counter using this name.

    Returns:
        Callable: An activation handle for the given operation.
    """

    def _generic_activation_jit(
        i: Any, outputs: List[Any]
    ) -> Union[typing.Counter[str], Number]:
        """
        This is a generic jit handle that counts the number of activations for any
        operation given the output shape.
        """
        out_shape = get_shape(outputs[0])
        ac_count = prod(out_shape)
        if op_name is None:
            return ac_count
        else:
            return Counter({op_name: ac_count})

    return _generic_activation_jit


def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


def linear_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the aten::linear operator.
    """
    # Inputs is a list of length 3; unlike aten::addmm, it is the first
    # two elements that are relevant.
    input_shapes = [get_shape(v) for v in inputs[0:2]]
    # input_shapes[0]: [dim0, dim1, ..., input_feature_dim]
    # input_shapes[1]: [output_feature_dim, input_feature_dim]
    assert input_shapes[0][-1] == input_shapes[1][-1]
    flops = prod(input_shapes[0]) * input_shapes[1][0]
    return flops


def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.

    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).

    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop


def conv_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    Count flops for convolution.
    """
    # Inputs of Convolution should be a list of length 12 or 13. They represent:
    # 0) input tensor, 1) convolution filter, 2) bias, 3) stride, 4) padding,
    # 5) dilation, 6) transposed, 7) out_pad, 8) groups, 9) benchmark_cudnn,
    # 10) deterministic_cudnn and 11) user_enabled_cudnn.
    # starting with #40737 it will be 12) user_enabled_tf32
    assert len(inputs) == 12 or len(inputs) == 13, len(inputs)
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6].toIValue()

    # use a custom name instead of "_convolution"
    return Counter(
        {"conv": conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)}
    )


def einsum_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the einsum operation.
    """
    # Inputs of einsum should be a list of length 2+.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    # Inputs[2] optionally stores the optimized path of contraction.
    assert len(inputs) >= 2, len(inputs)
    equation = inputs[0].toIValue()
    # Get rid of white space in the equation string.
    equation = equation.replace(" ", "")
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        return flop

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        return flop
    else:
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
        raise NotImplementedError("Unsupported einsum operation.")


def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop


def norm_flop_counter(affine_arg_index: int) -> Handle:
    """
    Args:
        affine_arg_index: index of the affine argument in inputs
    """

    def norm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for norm layers.
        """
        # Inputs[0] contains the shape of the input.
        input_shape = get_shape(inputs[0])
        has_affine = get_shape(inputs[affine_arg_index]) is not None
        assert 2 <= len(input_shape) <= 5, input_shape
        # 5 is just a rough estimate
        flop = prod(input_shape) * (5 if has_affine else 4)
        return flop

    return norm_flop_jit


def batchnorm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    training = inputs[5].toIValue()
    assert isinstance(training, bool), "Signature of aten::batch_norm has changed!"
    if training:
        return norm_flop_counter(1)(inputs, outputs)  # pyre-ignore
    has_affine = get_shape(inputs[1]) is not None
    input_shape = prod(get_shape(inputs[0]))
    return input_shape * (2 if has_affine else 1)


def elementwise_flop_counter(input_scale: float = 1, output_scale: float = 0) -> Handle:
    """
    Count flops by
        input_tensor.numel() * input_scale + output_tensor.numel() * output_scale

    Args:
        input_scale: scale of the input tensor (first argument)
        output_scale: scale of the output tensor (first element in outputs)
    """

    def elementwise_flop(inputs: List[Any], outputs: List[Any]) -> Number:
        ret = 0
        if input_scale != 0:
            shape = get_shape(inputs[0])
            ret += input_scale * prod(shape)
        if output_scale != 0:
            shape = get_shape(outputs[0])
            ret += output_scale * prod(shape)
        return ret

    return elementwise_flop

T = TypeVar("T", bound="JitModelAnalysis")

_IGNORED_OPS: Set[str] = {
    "aten::Int",
    "aten::ScalarImplicit",
    "aten::__and__",
    "aten::arange",
    "aten::bitwise_not",
    "aten::cat",
    "aten::chunk",
    "aten::clamp",
    "aten::clamp_",
    "aten::constant_pad_nd",
    "aten::contiguous",
    "aten::copy_",
    "aten::detach",
    "aten::dropout",
    "aten::empty",
    "aten::eq",
    "aten::expand",
    "aten::flatten",
    "aten::floor",
    "aten::floor_divide",
    "aten::full",
    "aten::full_like",
    "aten::gather",
    "aten::ge",
    "aten::gt",
    "aten::index",
    "aten::index_put_",
    "aten::masked_fill",
    "aten::max",
    "aten::narrow",
    "aten::new_empty",
    "aten::new_full",
    "aten::new_zeros",
    "aten::nonzero",
    "aten::ones",
    "aten::permute",
    "aten::relu",
    "aten::relu_",
    "aten::remainder",
    "aten::reshape",
    "aten::roll",
    "aten::select",
    "aten::size",
    "aten::slice",
    "aten::split",
    "aten::split_with_sizes",
    "aten::squeeze",
    "aten::stack",
    "aten::t",
    "aten::to",
    "aten::transpose",
    "aten::type_as",
    "aten::unbind",
    "aten::unsqueeze",
    "aten::unsqueeze_",
    "aten::view",
    "aten::zeros",
    "aten::zeros_like",
}

@dataclass
class Statistics:
    """
    For keeping track of the various model statistics recorded during
    analysis.
    """

    counts: "Dict[str, Counter[str]]"
    unsupported_ops: "Dict[str, Counter[str]]"
    uncalled_mods: "Set[str]"

def _named_modules_with_dup(
    model: nn.Module, prefix: str = ""
) -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)
        
def _named_modules_without_dup(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """
    Like .named_modules(), but the results are slightly different for
    some wrapped models.
    """
    seen = set()
    for name, mod in _named_modules_with_dup(model):
        if mod not in seen:
            seen.add(mod)
            yield name, mod
def _get_scoped_trace_graph(
    module: nn.Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    aliases: Dict[Union[str, nn.Module], str],
) -> torch._C.Graph:
    """
    Traces the provided module using torch.jit._get_trace_graph, but adds
    submodule scope information to each graph node. The resulting graph
    is in-lined and has all model parameters treated as inputs. The input
    model has the scope name '', while its descendants have names of the
    form 'child.grandchild.grandgrandchild...'.

    Args:
        model (nn.Module) : The module to trace
        inputs (tuple) : Inputs used during the trace of the model
        aliases (dict(str or nn.Module, str) : maps modules and module
            names to the canonical name to be used as the scope for
            that module.

    Returns:
        graph (torch._C.Graph) : The pytorch JIT trace of the model
    """

    class ScopePushHook:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, module: nn.Module, inputs: Any) -> Any:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.push_scope(self.name)
            return inputs

    class ScopePopHook:
        def __call__(self, module: nn.Module, inputs: Any, outputs: Any) -> Any:
            tracing_state = torch._C._get_tracing_state()
            if tracing_state:
                tracing_state.pop_scope()
            return outputs

    hook_handles: List[Any] = []

    def register_hooks(mod: nn.Module, name: str) -> None:
        prehook = mod.register_forward_pre_hook(ScopePushHook(name))
        posthook = mod.register_forward_hook(ScopePopHook())
        hook_handles.append(prehook)
        hook_handles.append(posthook)

    # Unwrap DDP, but correct the scope names for the root module.
    if isinstance(
        module, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
    ):
        # Since DataParallel just wraps the model, add an extra set of hooks
        # to the model it wraps to account for the wrapper. Then trace it.
        root_name = aliases[module]
        module = module.module
        register_hooks(module, root_name)

    for name, mod in _named_modules_without_dup(module):
        name = aliases[mod]
        register_hooks(mod, name)

    graph, _ = _get_trace_graph(module, inputs)

    for handle in hook_handles:
        handle.remove()

    return graph

class JitModelAnalysis:
    """
    Provides access to per-submodule model statistics obtained by
    tracing a model with pytorch's jit tracing functionality. Calculates
    a statistic on a per-operator basis using the provided set of functions
    that acts on the inputs and outputs to the operator, then aggregates
    this over modules in the model. Can return the aggregate statistic for
    any submodule in the model. Is lazily evaluated, and will perform the
    trace when a statistic is first requested. Changing the operator handles
    will cause the trace to be rerun on the next request.

    Submodules may be referred to using the module's name. The input model has
    name "", while its descendants have names of the form
    "child.grandchild.grandgrandchild...".

    An operator is treated as within the scope of a module if calling that
    module directly resulted in that operator being run. In particular,
    this means that calls to other functions owned by a module or explicit
    calls to module.forward(...) will not register resulting operators as
    contributing statistics to that module.
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        """
        Args:
            model: The model to analyze
            inputs: The inputs to the model for analysis.

        We will trace the execution of `model.forward(inputs)`. This means
        inputs have to be tensors or tuple of tensors (see
        https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace).
        In order to trace other methods or unsupported input types, you may need
        to implement a wrapper module.
        """
        self._model = model
        self._inputs = inputs
        self._op_handles: Dict[str, Handle] = {}
        # Mapping from names to submodules
        self._named_modules: Dict[str, nn.Module] = dict(_named_modules_with_dup(model))
        # Mapping from submodules and their aliases to the canonical name of each submodule
        self._aliases: Dict[Union[nn.Module, str], str] = self._get_aliases(model)
        self._stats: Optional[Statistics] = None

        self._ignored_ops: Set[str] = copy(_IGNORED_OPS)
        self.unsupported_ops_warnings(True)
        self.uncalled_modules_warnings(True)
        self.tracer_warnings("no_tracer_warning")
        self.ancestor_mode("owner")

    def total(self, module_name: str = "") -> int:
        """
        Returns the total aggregated statistic across all operators
        for the requested module.

        Args:
            module_name (str) : The submodule to get data for. Defaults to
                the entire model.
        Returns:
            int : The aggregated statistic.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        total_count = sum(stats.counts[module_name].values())
        return total_count

    def by_operator(self, module_name: str = "") -> typing.Counter[str]:
        """
        Returns the statistics for a requested module, grouped by operator
        type. The operator handle determines the name associated with each
        operator type.

        Args:
            module_name (str) : The submodule to get data for. Defaults
                to the entire model.
        Returns:
            Counter(str) : The statistics for each operator.
        """
        stats = self._analyze()
        module_name = self.canonical_module_name(module_name)
        return stats.counts[module_name]

    def by_module_and_operator(self) -> Dict[str, typing.Counter[str]]:
        """
        Returns the statistics for all submodules, separated out by
        operator type for each submodule. The operator handle determines
        the name associated with each operator type.

        Returns:
            dict(str, Counter(str)):
                The statistics for each submodule and each operator.
                Grouped by submodule names, then by operator name.
        """
        stats = self._analyze()
        return stats.counts

    def by_module(self) -> typing.Counter[str]:
        """
        Returns the statistics for all submodules, aggregated over
        all operators.

        Returns:
            Counter(str): statistics counter grouped by submodule names
        """
        stats = self._analyze()
        summed_counts = Counter()
        for mod, results in stats.counts.items():
            summed_counts[mod] = sum(results.values())
        return summed_counts

    def unsupported_ops(self, module_name: str = "") -> typing.Counter[str]:
        """
        Lists the number of operators that were encountered but unsupported
        because no operator handle is available for them. Does not include
        operators that are explicitly ignored.

        Args:
            module_name (str) : The submodule to list unsupported ops.
                Defaults to the entire model.

        Returns:
            Counter(str) : The number of occurences each unsupported operator.
        """
        if self._stats is None:
            raise RuntimeError(
                "Analysis results should be computed "
                "before calling unsupported_ops()"
            )
        module_name = self.canonical_module_name(module_name)
        return self._stats.unsupported_ops[module_name]  # pyre-fixme

    def uncalled_modules(self) -> Set[str]:
        """
        Returns a set of submodules that were never called during the
        trace of the graph. This may be because they were unused, or
        because they were accessed via direct calls .forward() or with
        other python methods. In the latter case, statistics will not be
        attributed to the submodule, though the statistics will be included
        in the parent module.

        Returns:
            set(str) : The set of submodule names that were never called
                during the trace of the model.
        """
        stats = self._analyze()
        return stats.uncalled_mods

    def set_op_handle(self, *args, **kwargs: Optional[Handle]) -> "JitModelAnalysis":
        """
        Sets additional operator handles, or replaces existing ones.

        Args:
            args: (str, Handle) pairs of operator names and handles.
            kwargs: mapping from operator names to handles.

        If a handle is ``None``, the op will be explicitly ignored. Otherwise,
        handle should be a function that calculates the desirable statistic
        from an operator. The function must take two arguments, which are the
        inputs and outputs of the operator, in the form of ``list(torch._C.Value)``.
        The function should return a counter object with per-operator statistics.

        Examples
        ::
            handlers = {"aten::linear": my_handler}
            counter.set_op_handle("aten::matmul", None, "aten::bmm", my_handler2)
                   .set_op_handle(**handlers)
        """
        self._stats = None
        if len(args) % 2 != 0:
            raise TypeError(
                "set_op_handle should be called with pairs of names and handles!"
            )
        for name, handle in zip(args[::2], args[1::2]):
            kwargs[name] = handle
        for name, handle in kwargs.items():
            if handle is None:
                self._ignored_ops.add(name)
            else:
                self._op_handles[name] = handle
        return self

    def clear_op_handles(self) -> "JitModelAnalysis":
        """
        Clears all operator handles currently set.
        """
        self._op_handles = {}
        self._ignored_ops = copy(_IGNORED_OPS)
        self._stats = None
        return self

    def canonical_module_name(self, name: str) -> str:
        """
        Returns the canonical module name of the given ``name``, which might be
        different from the given ``name`` if the module is shared.
        This is the name that will be used as a key when statistics are
        output using .by_module() and .by_module_and_operator().

        Args:
            name (str) : The name of the module to find the canonical name for.
        Returns:
            str : The canonical name of the module.
        """
        # Blocks access by a direct module reference
        assert isinstance(name, str), "Module name must be a string."
        if name in self._aliases:
            return self._aliases[name]
        else:
            raise KeyError(
                "Requested module name is not among "
                "the descendants of the analyzed model."
            )

    def copy(
        self,
        new_model: Optional[nn.Module] = None,
        new_inputs: Union[None, Tensor, Tuple[Tensor, ...]] = None,
    ) -> "JitModelAnalysis":
        """
        Returns a copy of the :class:`JitModelAnalysis` object, keeping all
        settings, but on a new model or new inputs.

        Args:
            new_model (nn.Module or None) : a new model for the new
                JitModelAnalysis. If None, uses the original model.
            new_inputs (typing.Tuple[object, ...] or None) : new inputs
                for the new JitModelAnalysis. If None, uses the original
                inputs.
        Returns:
            JitModelAnalysis : the new model analysis object
        """
        model = self._model if new_model is None else new_model
        inputs = self._inputs if new_inputs is None else new_inputs
        return (
            JitModelAnalysis(model=model, inputs=inputs)
            .set_op_handle(**self._op_handles)
            .unsupported_ops_warnings(self._enable_warn_unsupported_ops)
            .uncalled_modules_warnings(self._enable_warn_uncalled_mods)
            .tracer_warnings(self._warn_trace)
        )

    def tracer_warnings(self: T, mode: str) -> T:
        """
        Sets which warnings to print when tracing the graph to calculate
        statistics. There are three modes. Defaults to 'no_tracer_warning'.
        Allowed values are:

        * 'all' : keeps all warnings raised while tracing
        * 'no_tracer_warning' : suppress torch.jit.TracerWarning only
        * 'none' : suppress all warnings raised while tracing

        Args:
            mode (str) : warning mode in one of the above values.
        """
        if mode not in ["all", "no_tracer_warning", "none"]:
            raise ValueError(f"Unrecognized tracer warning mode {mode}.")
        self._warn_trace = mode
        return self

    def ancestor_mode(self: T, mode: str) -> T:
        """
        Sets how to determine the ancestor modules of an operator. Must be one of
        "owner" or "caller".

        * "caller": an operator belongs to all modules that is currently executing
          `forward()` at the time the operator is called.
        * "owner": an operator belongs to the last module that's executing
          `forward()` at the time the operator is called, plus this module's recursive
          parents.  If an module has multiple parents (e.g. a shared module), only one
          will be picked.

        For most cases, a module only calls submodules it owns, so both options would
        work identically. In certain edge cases, this option will affect the hierarchy
        of results, but won't affect the total count.
        """
        if mode not in ["owner", "caller"]:
            raise ValueError(f"Unrecognized ancestor mode: {mode}")
        self._ancestor_mode = mode
        return self

    def unsupported_ops_warnings(self: T, enabled: bool) -> T:
        """
        Sets if warnings for unsupported operators are shown. Defaults
        to True. Counts of unsupported operators may be obtained from
        :meth:`unsupported_ops` regardless of this setting.

        Args:
            enabled (bool) : Set to 'True' to show unsupported operator
                warnings.
        """
        self._enable_warn_unsupported_ops = enabled
        return self

    def uncalled_modules_warnings(self: T, enabled: bool) -> T:
        """
        Sets if warnings from uncalled submodules are shown. Defaults to true.
        A submodule is considered "uncalled" if it is never called during
        tracing. This may be because it is actually unused, or because it is
        accessed via calls to ``.forward()`` or other methods of the module.
        The set of uncalled modules may be obtained from
        :meth:`uncalled_modules` regardless of this setting.

        Args:
            enabled (bool) : Set to 'True' to show warnings.
        """
        self._enable_warn_uncalled_mods = enabled
        return self

    def _warn_unsupported_ops(self, ops: typing.Counter[str]) -> None:
        if not self._enable_warn_unsupported_ops:
            return
        logger = logging.getLogger(__name__)
        for op, freq in ops.items():
            logger.warning(
                "Unsupported operator {} encountered {} time(s)".format(op, freq)
            )

    def _warn_uncalled_mods(self, uncalled_mods: Set[str]) -> None:
        if not self._enable_warn_uncalled_mods:
            return
        uncalled_mods = {x for x in uncalled_mods if self._has_forward(x)}
        if len(uncalled_mods) == 0:
            return

        logger = logging.getLogger(__name__)
        logger.warning(
            "The following submodules of the model were never "
            "called during the trace of the graph. They may be "
            "unused, or they were accessed by direct calls to "
            ".forward() or via other python methods. In the latter "
            "case they will have zeros for statistics, though their "
            "statistics will still contribute to their parent calling "
            "module.\n" + ", ".join(sorted(uncalled_mods))
        )

    def _get_aliases(self, model: nn.Module) -> Dict[Union[str, nn.Module], str]:
        aliases = {}
        for name, module in _named_modules_with_dup(model):
            if module not in aliases:
                aliases[module] = name
            aliases[name] = aliases[module]
            if "/" in name:
                sub_name = name.split("/")[-1]
                aliases[sub_name] = aliases[module]
        return aliases

    def _get_all_ancestors(self, module_name: str) -> Set[str]:
        """
        Get all ancestors of the given module, defined by ownership.
        If the given module has multiple owners, use its canonical name.
        """
        parts = self.canonical_module_name(module_name).split(".")
        res = {""}
        for k in range(len(parts) + 1):
            res.add(".".join(parts[:k]))
        return res

    def _analyze(self) -> "Statistics":
        # Don't calculate if results are already stored.
        stats = self._stats
        if stats is not None:
            return stats

        with warnings.catch_warnings():
            if self._warn_trace == "none":
                warnings.simplefilter("ignore")
            elif self._warn_trace == "no_tracer_warning":
                warnings.filterwarnings("ignore", category=TracerWarning)
            graph = _get_scoped_trace_graph(self._model, self._inputs, self._aliases)

        # Assures even modules not in the trace graph are initialized to zero count
        counts = {}
        unsupported_ops = {}
        # We don't need the duplication here, but self._model.named_modules()
        # gives slightly different results for some wrapped models.
        for _, mod in _named_modules_with_dup(self._model):
            name = self._aliases[mod]
            counts[name] = Counter()
            unsupported_ops[name] = Counter()

        all_seen = set()
        for node in graph.nodes():
            kind = node.kind()
            if kind == "prim::PythonOp":
                # for PythonOp, pyname contains the actual name in Python
                # pyre-fixme[16]: `Node` has no attribute `pyname`.
                kind = kind + "." + node.pyname()
            scope_names = node.scopeName().split("/")
            all_seen.update(scope_names)
            if self._ancestor_mode == "caller":
                ancestors = set(scope_names)
            else:
                ancestors = self._get_all_ancestors(scope_names[-1])
                all_seen.update(ancestors)
            if kind not in self._op_handles:
                if self._should_ignore_node(node):
                    continue
                for name in ancestors:
                    unsupported_ops[name][kind] += 1
            else:
                inputs, outputs = list(node.inputs()), list(node.outputs())
                op_counts = self._op_handles[kind](inputs, outputs)
                if isinstance(op_counts, Number):
                    op_counts = Counter({self._simplify_op_name(kind): op_counts})
                for v in op_counts.values():
                    if not isinstance(v, (int, float, np.float64, np.int64)):
                        raise ValueError(
                            f"Invalid type {type(v)} for the flop count! "
                            "Please use a wider type to avoid overflow."
                        )

                # Assures an op contributes at most once to a module
                for name in ancestors:
                    counts[name] += op_counts

        uncalled_mods = set(self._aliases.values()) - all_seen
        stats = Statistics(
            counts=counts, unsupported_ops=unsupported_ops, uncalled_mods=uncalled_mods
        )
        self._stats = stats
        self._warn_unsupported_ops(unsupported_ops[""])
        self._warn_uncalled_mods(uncalled_mods)
        return stats

    def _simplify_op_name(self, full_op_name: str) -> str:
        """
        Get simplified name of the op without the preceding namespace, e.g.
        aten::batch_norm -> batch_norm
        """
        p = full_op_name.find("::")
        if p != -1:
            return full_op_name[p + 2 :]
        else:
            return full_op_name

    def _has_forward(self, mod_name: str) -> bool:
        # Whether the module has a valid forward method.
        # Modules without forward are not expected to get called
        # and therefore should not produce "uncalled" warnings
        module = self._named_modules.get(mod_name)
        if module is None:
            return False
        module_type = type(module)
        # Containers are not meant to be called anyway (they don't have forward)
        # NOTE: We add nn.Identity as well to silence the uncalled warning, but it's
        # different from other containers: Identity has a forward but the forward does
        # not contain ops, so it appears "uncalled" after tracing. A more proper way
        # may be to use forward hooks (instead of the graph) to decide whether a module
        # has been called.
        no_forward_mods = {nn.ModuleList, nn.ModuleDict, nn.Module, nn.Identity}
        for mod in no_forward_mods:
            if module_type.forward is mod.forward:
                return False
        return True

    def _should_ignore_node(self, node) -> bool:
        kind = node.kind()
        if kind in self._ignored_ops:
            return True
        # Ignore all prim:: operators, with two exceptions:
        # * prim::PythonOp can be a user-implemented `torch.autograd.Function`
        # * prim::CallFunction an be a call to scripted module/function.
        if kind.startswith("prim::PythonOp") or kind.startswith("prim::CallFunction"):
            return False
        if kind.startswith("prim::"):
            return True
        return False
# A dictionary that maps supported operations to their flop count jit handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::addmm": addmm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
    "aten::mm": matmul_flop_jit,
    "aten::linear": linear_flop_jit,
    # You might want to ignore BN flops due to inference-time fusion.
    # Use `set_op_handle("aten::batch_norm", None)
    "aten::batch_norm": batchnorm_flop_jit,
    "aten::group_norm": norm_flop_counter(2),
    "aten::layer_norm": norm_flop_counter(2),
    "aten::instance_norm": norm_flop_counter(1),
    "aten::upsample_nearest2d": elementwise_flop_counter(0, 1),
    "aten::upsample_bilinear2d": elementwise_flop_counter(0, 4),
    "aten::adaptive_avg_pool2d": elementwise_flop_counter(1, 0),
    "aten::grid_sampler": elementwise_flop_counter(0, 4),  # assume bilinear
}
class FlopCountAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule model flop count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard flop counters for a few common operators.
    Note that:

        1. Flop is not a well-defined concept. We just produce our best estimate.
        2. We count one fused multiply-add as one flop.

    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.

    Flop counts can be obtained as:

    * ``.total(module_name="")``: total flop count for the module
    * ``.by_operator(module_name="")``: flop counts for the module, as a Counter
      over different operator types
    * ``.by_module()``: Counter of flop counts for all submodules
    * ``.by_module_and_operator()``: dictionary indexed by descendant of Counters
      over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Example usage:

    >>> import torch.nn as nn
    >>> import torch
    >>> class TestModel(nn.Module):
    ...    def __init__(self):
    ...        super().__init__()
    ...        self.fc = nn.Linear(in_features=1000, out_features=10)
    ...        self.conv = nn.Conv2d(
    ...            in_channels=3, out_channels=10, kernel_size=1
    ...        )
    ...        self.act = nn.ReLU()
    ...    def forward(self, x):
    ...        return self.fc(self.act(self.conv(x)).flatten(1))

    >>> model = TestModel()
    >>> inputs = (torch.randn((1,3,10,10)),)
    >>> flops = FlopCountAnalysis(model, inputs)
    >>> flops.total()
    13000
    >>> flops.total("fc")
    10000
    >>> flops.by_operator()
    Counter({"addmm" : 10000, "conv" : 3000})
    >>> flops.by_module()
    Counter({"" : 13000, "fc" : 10000, "conv" : 3000, "act" : 0})
    >>> flops.by_module_and_operator()
    {"" : Counter({"addmm" : 10000, "conv" : 3000}),
     "fc" : Counter({"addmm" : 10000}),
     "conv" : Counter({"conv" : 3000}),
     "act" : Counter()
    }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__


def flop_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """
    Given a model and an input to the model, compute the per-operator Gflops
    of the given model.

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op. We count
            one Multiply-Add as one FLOP.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            gflops for each operation and a Counter that records the number of
            unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    flop_counter = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops)
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.unsupported_ops()

def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r

class ActivationCountAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule model activation count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard activation counters for convolutional and dot-product
    operators.

    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.

    Activation counts can be obtained as:

    * ``.total(module_name="")``: total activation count for a module
    * ``.by_operator(module_name="")``: activation counts for the module, as a
      Counter over different operator types
    * ``.by_module()``: Counter of activation counts for all submodules
    * ``.by_module_and_operator()``: dictionary indexed by descendant of Counters
      over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Example usage:

    >>> import torch.nn as nn
    >>> import torch
    >>> class TestModel(nn.Module):
    ...     def __init__(self):
    ...        super().__init__()
    ...        self.fc = nn.Linear(in_features=1000, out_features=10)
    ...        self.conv = nn.Conv2d(
    ...            in_channels=3, out_channels=10, kernel_size=1
    ...        )
    ...        self.act = nn.ReLU()
    ...    def forward(self, x):
    ...        return self.fc(self.act(self.conv(x)).flatten(1))

    >>> model = TestModel()
    >>> inputs = (torch.randn((1,3,10,10)),)
    >>> acts = ActivationCountAnalysis(model, inputs)
    >>> acts.total()
    1010
    >>> acts.total("fc")
    10
    >>> acts.by_operator()
    Counter({"conv" : 1000, "addmm" : 10})
    >>> acts.by_module()
    Counter({"" : 1010, "fc" : 10, "conv" : 1000, "act" : 0})
    >>> acts.by_module_and_operator()
    {"" : Counter({"conv" : 1000, "addmm" : 10}),
     "fc" : Counter({"addmm" : 10}),
     "conv" : Counter({"conv" : 1000}),
     "act" : Counter()
    }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__

def _fill_missing_statistics(
    model: nn.Module, statistics: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    """
    If, for a given submodule name in the model, a statistic is missing
    from statistics, fills it in with zero. This visually uniformizes
    the reporting of statistics.

    Args:
        model (nn.Module) : the model whose submodule names will be
            used to fill statistics
        statistics (dict(str, dict(str, int))) : the statistics to
            fill in missing values for. Organized as a dictionary
            over statistics, which are each a dictionary over submodules'
            names. The statistics are assumed to be formatted already
            to the desired string format for printing.

    Returns:
        dict(str, dict(str, int)) : the input statistics with missing
            values filled with zero.
    """
    out_stats = {name: stat.copy() for name, stat in statistics.items()}
    for mod_name, _ in model.named_modules():
        for stat in out_stats.values():
            if mod_name not in stat:
                stat[mod_name] = 0
    return out_stats
def _group_by_module(
    statistics: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Converts statistics organized first by statistic type and then by module
    to statistics organized first by module and then by statistic type.

    Args:
        statistics (dict(str, dict(str, any))) : the statistics to convert

    Returns:
        dict(str, dict(str, any)) : the reorganized statistics
    """
    out_stats = defaultdict(dict)
    for stat_name, stat in statistics.items():
        for mod, val in stat.items():
            out_stats[mod][stat_name] = val
    return dict(out_stats)
def _remove_zero_statistics(
    statistics: Dict[str, Dict[str, int]],
    force_keep: Optional[Set[str]] = None,
    require_trivial_children: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Any module that has zero for all available statistics is removed from the
    set of statistics. This can help declutter the reporting of statistics
    if many submodules have zero statistics. Assumes the statistics have
    a model hierarchy starting with a root that has name ''.

    Args:
        statistics (dict(str, dict(str, int))) : the statistics to
            remove zeros from. Organized as a dictionary over modules,
            which are each a dictionary over statistic types.
        force_keep (set(str) or None) : a set of modules to always keep, even
            if they are all zero.
        require_trivial_children (bool) : If True, a statistic will only
            be deleted if all its children are also deleted. Defaults to
            False.

    Returns:
        dict(str, dict(str, int)) : the input statistics dictionary,
            with submodules removed if they have zero for all statistics.
    """
    out_stats: Dict[str, Dict[str, int]] = {}
    _force_keep: Set[str] = force_keep if force_keep else set() | {""}

    def keep_stat(name: str) -> None:
        prefix = name + ("." if name else "")
        trivial_children = True
        for mod in statistics:
            # 'if mod' excludes root = '', which is never a child
            if mod and mod.count(".") == prefix.count(".") and mod.startswith(prefix):
                keep_stat(mod)
                trivial_children &= mod not in out_stats

        if (
            (not all(val == 0 for val in statistics[name].values()))
            or (name in _force_keep)
            or (require_trivial_children and not trivial_children)
        ):
            out_stats[name] = statistics[name].copy()

    keep_stat("")
    return out_stats
def _format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """
    Formats an integer for printing in a table or model representation.
    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int) : The integer to format.
        sig_figs (int) : The number of significant figures to keep
        hide_zero (bool) : If True, x=0 is replaced with an empty string
            instead of '0'.

    Returns:
        str : The formatted string.
    """
    if hide_zero and x == 0:
        return str("")

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return "{{:.{}f}}".format(sig_figs).format(x).rstrip("0").rstrip(".")

    if abs(x) > 1e14:
        return fmt(x / 1e15) + "P"
    if abs(x) > 1e11:
        return fmt(x / 1e12) + "T"
    if abs(x) > 1e8:
        return fmt(x / 1e9) + "G"
    if abs(x) > 1e5:
        return fmt(x / 1e6) + "M"
    if abs(x) > 1e2:
        return fmt(x / 1e3) + "K"
    return str(x)

def _pretty_statistics(
    statistics: Dict[str, Dict[str, int]], sig_figs: int = 3, hide_zero: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Converts numeric statistics to strings with kilo/mega/giga/etc.
    labels.

    Args:
        statistics (dict(str, dict(str, int))) : the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types.
        sig_figs (int) : the number of significant figures for each stat
        hide_zero (bool) : if True, statistics that are zero will be
            written as an empty string. Defaults to False.

    Return:
        dict(str, dict(str, str)) : the input statistics as pretty strings
    """
    out_stats = {}
    for mod, stats in statistics.items():
        out_stats[mod] = {
            s: _format_size(val, sig_figs, hide_zero) for s, val in stats.items()
        }
    return out_stats
def _indicate_uncalled_modules(
    statistics: Dict[str, Dict[str, str]],
    stat_name: str,
    uncalled_modules: Set[str],
    uncalled_indicator: str = "N/A",
) -> Dict[str, Dict[str, str]]:
    """
    If a module is in the set of uncalled modules, replace its statistics
    with the specified indicator, instead of using the existing string.
    Assumes the statistic is already formatting in string form.

    Args:
        statistics (dict(str, dict(str, str))) : the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types. Expects statistics
            have already been converted to strings.
        stat_name (str) : the name of the statistic being modified
        uncalled_modules set(str) : a set of names of uncalled modules.
        indicator (str) : the string that will be used to indicate
            unused modules. Defaults to 'N/A'.

    Returns:
        dict(str, dict(str, str)) : the modified statistics
    """

    stats_out = {mod: stats.copy() for mod, stats in statistics.items()}
    for mod in uncalled_modules:
        if mod not in stats_out:
            stats_out[mod] = {}
        stats_out[mod][stat_name] = uncalled_indicator
    return stats_out
def _model_stats_str(model: nn.Module, statistics: Dict[str, Dict[str, str]]) -> str:
    """
    This produces a representation of the model much like 'str(model)'
    would, except the provided statistics are written out as additional
    information for each submodule.

    Args:
        model (nn.Module) : the model to form a representation of.
        statistics (dict(str, dict(str, str))) : the statistics to
            include in the model representations. Organized as a dictionary
            over module names, which are each a dictionary over statistics.
            The statistics are assumed to be formatted already to the
            desired string format for printing.

    Returns:
        str : the string representation of the model with the statistics
            inserted.
    """

    # Copied from nn.Module._addindent
    def _addindent(s_: str, numSpaces: int) -> str:
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    def print_statistics(name: str) -> str:
        if name not in statistics:
            return ""
        printed_stats = ["{}: {}".format(k, v) for k, v in statistics[name].items()]
        return ", ".join(printed_stats)

    # This comes directly from nn.Module.__repr__ with small changes
    # to include the statistics.
    def repr_with_statistics(module: nn.Module, name: str) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = module.extra_repr()
        printed_stats = print_statistics(name)
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines.extend(extra_repr.split("\n"))
        if printed_stats:
            extra_lines.extend(printed_stats.split("\n"))
        child_lines = []
        for key, submod in module._modules.items():
            submod_name = name + ("." if name else "") + key
            # pyre-fixme[6]: Expected `Module` for 1st param but got
            #  `Optional[nn.modules.module.Module]`.
            submod_str = repr_with_statistics(submod, submod_name)
            submod_str = _addindent(submod_str, 2)
            child_lines.append("(" + key + "): " + submod_str)
        lines = extra_lines + child_lines

        main_str = module._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    return repr_with_statistics(model, "")
def flop_count_str(
    flops: FlopCountAnalysis, activations: Optional[ActivationCountAnalysis] = None
) -> str:
    """
    Calculates the parameters and flops of the model with the given inputs
    and returns a string representation of the model that includes the
    parameters and flops of every submodule. The string is structured
    to be similar that given by str(model), though it is not guaranteed to
    be identical in form if the default string representation of a module has
    been overridden. If a module has zero parameters and flops, statistics
    will not be reported for succinctness.

    The trace can only register the scope of a module if it is called
    directly, which means flops (and activations) arising from explicit
    calls to .forward() or to other python functions of the module will
    not be attributed to that module. Modules that are never called will
    have 'N/A' listed for their flops; this means they are either unused
    or their statistics are missing for this reason. Any such flops are still
    counted towards the parent

    Example:

    >>> import torch
    >>> import torch.nn as nn

    >>> class InnerNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10,10)
    ...         self.fc2 = nn.Linear(10,10)
    ...     def forward(self, x):
    ...         return self.fc1(self.fc2(x))

    >>> class TestNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10,10)
    ...         self.fc2 = nn.Linear(10,10)
    ...         self.inner = InnerNet()
    ...     def forward(self, x):
    ...         return self.fc1(self.fc2(self.inner(x)))

    >>> inputs = torch.randn((1,10))
    >>> print(flop_count_str(FlopCountAnalysis(model, inputs)))
    TestNet(
      #params: 0.44K, #flops: 0.4K
      (fc1): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100
      )
      (fc2): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100
      )
      (inner): InnerNet(
        #params: 0.22K, #flops: 0.2K
        (fc1): Linear(
          in_features=10, out_features=10, bias=True
          #params: 0.11K, #flops: 100
        )
        (fc2): Linear(
          in_features=10, out_features=10, bias=True
          #params: 0.11K, #flops: 100
        )
      )
    )


    Args:
        flops (FlopCountAnalysis): the flop counting object
        activations (bool) : If given, the activations of each layer will
            also be calculated and included in the representation.

    Returns:
        str:
            a string representation of the model with the number of
            parameters and flops included.
    """
    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    model = flops._model
    params = dict(parameter_count(model))

    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings("none")
    stats = {"#params": params, "#flops": flops.by_module()}

    if activations is not None:
        activations.unsupported_ops_warnings(False)
        activations.uncalled_modules_warnings(False)
        activations.tracer_warnings("none")
        stats["#acts"] = activations.by_module()

    all_uncalled = flops.uncalled_modules() | (
        activations.uncalled_modules() if activations is not None else set()
    )
    stats = _fill_missing_statistics(model, stats)
    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(stats, force_keep=all_uncalled)
    stats = _pretty_statistics(stats, sig_figs=2)
    stats = _indicate_uncalled_modules(stats, "#flops", flops.uncalled_modules())
    if activations is not None:
        stats = _indicate_uncalled_modules(
            stats, "#acts", activations.uncalled_modules()
        )

    model_string = ""
    if all_uncalled:
        model_string += (
            "N/A indicates a possibly missing statistic due to how "
            "the module was called. Missing values are still included "
            "in the parent's total.\n"
        )
    model_string += _model_stats_str(model, stats)
    return model_string

import tabulate
def _try_combine(
    stats1: Dict[str, str], stats2: Dict[str, str]
) -> Optional[Dict[str, str]]:
    """
    Try combine two statistics dict to display in one row. If they conflict,
    returns None.
    """
    ret = {}
    if set(stats1.keys()) != set(stats2.keys()):
        return None
    for k, v1 in stats1.items():
        v2 = stats2[k]
        if v1 != v2 and len(v1) and len(v2):
            return None
        ret[k] = v1 if len(v1) else v2
    return ret
def _get_single_child(
    name: str, statistics: Dict[str, Dict[str, str]]
) -> Optional[str]:
    """
    If the given module has only a single child in statistics, return it.
    Otherwise, return None.
    """
    prefix = name + ("." if name else "")
    child = None
    for mod in statistics:
        # 'if mod' excludes root = '', which is never a child
        if mod and mod.count(".") == prefix.count(".") and mod.startswith(prefix):
            if child is None:
                child = mod
            else:
                return None  # We found a second child, so return None
    return child
def _fastforward(
    name: str, statistics: Dict[str, Dict[str, str]]
) -> Tuple[str, Dict[str, str]]:
    """
    If the given module has only a single child and matches statistics
    with that child, merge statistics and their names into one row.
    Then repeat until the condition isn't met.

    Returns:
        str: the new name
        dict: the combined statistics of this row
    """
    single_child = _get_single_child(name, statistics)
    if single_child is None:
        return name, statistics[name]
    combined = _try_combine(statistics[name], statistics[single_child])
    if combined is None:
        return name, statistics[name]
    statistics[single_child] = combined
    return _fastforward(single_child, statistics)

def _model_stats_table(
    statistics: Dict[str, Dict[str, str]],
    max_depth: int = 3,
    stat_columns: Optional[List[str]] = None,
) -> str:
    """
    Formats the statistics obtained from a model in a nice table.

    Args:
        statistics (dict(str, dict(str, str))) : The statistics to print.
            Organized as a dictionary over modules, then as a dictionary
            over statistics in the model. The statistics are assumed to
            already be formatted for printing.
        max_depth (int) : The maximum submodule depth to recurse to.
        stat_columns (list(str)) : Specify the order of the columns to print.
            If None, columns are found automatically from the provided
            statistics.

    Return:
        str : The formatted table.
    """
    if stat_columns is None:
        stat_columns = set()
        for stats in statistics.values():
            stat_columns.update(stats.keys())
        stat_columns = list(stat_columns)

    headers = ["module"] + stat_columns
    table: List[List[str]] = []

    def build_row(name: str, stats: Dict[str, str], indent_lvl: int) -> List[str]:
        indent = " " * indent_lvl
        row = [indent + name]
        for stat_name in stat_columns:  # pyre-ignore[16] Is not None at this point
            row_str = (indent + stats[stat_name]) if stat_name in stats else ""
            row.append(row_str)
        return row

    def fill(indent_lvl: int, prefix: str) -> None:
        if indent_lvl > max_depth:
            return
        for mod_name in statistics:
            # 'if mod' excludes root = '', which is never a child
            if (
                mod_name
                and mod_name.count(".") == prefix.count(".")
                and mod_name.startswith(prefix)
            ):
                mod_name, curr_stats = _fastforward(mod_name, statistics)
                if root_prefix and mod_name.startswith(root_prefix):
                    # Skip the root_prefix shared by all submodules as it carries 0 information
                    pretty_mod_name = mod_name[len(root_prefix) :]
                else:
                    pretty_mod_name = mod_name
                row = build_row(pretty_mod_name, curr_stats, indent_lvl)
                table.append(row)
                fill(indent_lvl + 1, mod_name + ".")

    root_name, curr_stats = _fastforward("", statistics)
    row = build_row(root_name or "model", curr_stats, indent_lvl=0)
    table.append(row)
    root_prefix = root_name + ("." if root_name else "")
    fill(indent_lvl=1, prefix=root_prefix)

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(table, headers=headers, tablefmt="pipe")
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab

def flop_count_table(
    flops: FlopCountAnalysis,
    max_depth: int = 3,
    activations: Optional[ActivationCountAnalysis] = None,
    show_param_shapes: bool = True,
) -> str:
    """
    Format the per-module parameters and flops of a model in a table.
    It looks like this:
    ::
        | model                            | #parameters or shape   | #flops    |
        |:---------------------------------|:-----------------------|:----------|
        | model                            | 34.6M                  | 65.7G     |
        |  s1                              |  15.4K                 |  4.32G    |
        |   s1.pathway0_stem               |   9.54K                |   1.23G   |
        |    s1.pathway0_stem.conv         |    9.41K               |    1.23G  |
        |    s1.pathway0_stem.bn           |    0.128K              |           |
        |   s1.pathway1_stem               |   5.9K                 |   3.08G   |
        |    s1.pathway1_stem.conv         |    5.88K               |    3.08G  |
        |    s1.pathway1_stem.bn           |    16                  |           |
        |  s1_fuse                         |  0.928K                |  29.4M    |
        |   s1_fuse.conv_f2s               |   0.896K               |   29.4M   |
        |    s1_fuse.conv_f2s.weight       |    (16, 8, 7, 1, 1)    |           |
        |   s1_fuse.bn                     |   32                   |           |
        |    s1_fuse.bn.weight             |    (16,)               |           |
        |    s1_fuse.bn.bias               |    (16,)               |           |
        |  s2                              |  0.226M                |  7.73G    |
        |   s2.pathway0_res0               |   80.1K                |   2.58G   |
        |    s2.pathway0_res0.branch1      |    20.5K               |    0.671G |
        |    s2.pathway0_res0.branch1_bn   |    0.512K              |           |
        |    s2.pathway0_res0.branch2      |    59.1K               |    1.91G  |
        |   s2.pathway0_res1.branch2       |   70.4K                |   2.28G   |
        |    s2.pathway0_res1.branch2.a    |    16.4K               |    0.537G |
        |    s2.pathway0_res1.branch2.a_bn |    0.128K              |           |
        |    s2.pathway0_res1.branch2.b    |    36.9K               |    1.21G  |
        |    s2.pathway0_res1.branch2.b_bn |    0.128K              |           |
        |    s2.pathway0_res1.branch2.c    |    16.4K               |    0.537G |
        |    s2.pathway0_res1.branch2.c_bn |    0.512K              |           |
        |   s2.pathway0_res2.branch2       |   70.4K                |   2.28G   |
        |    s2.pathway0_res2.branch2.a    |    16.4K               |    0.537G |
        |    s2.pathway0_res2.branch2.a_bn |    0.128K              |           |
        |    s2.pathway0_res2.branch2.b    |    36.9K               |    1.21G  |
        |    s2.pathway0_res2.branch2.b_bn |    0.128K              |           |
        |    s2.pathway0_res2.branch2.c    |    16.4K               |    0.537G |
        |    s2.pathway0_res2.branch2.c_bn |    0.512K              |           |
        |    ............................. |    ......              |    ...... |

    Args:
        flops (FlopCountAnalysis): the flop counting object
        max_depth (int) : The max depth of submodules to include in the
            table. Defaults to 3.
        activations (ActivationCountAnalysis or None) : If given, include
            activation counts as an additional column in the table.
        show_param_shapes (bool) : If true, shapes for parameters will be
            included in the table. Defaults to True.

    Returns:
        str : The formatted table.

    Examples:
    ::
        print(flop_count_table(FlopCountAnalysis(model, inputs)))
    """
    params_header = "#parameters" + (" or shape" if show_param_shapes else "")
    flops_header, acts_header = "#flops", "#activations"

    model = flops._model
    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    params = dict(parameter_count(model))

    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings("none")

    stats = {params_header: params, flops_header: flops.by_module()}
    stat_columns = [params_header, flops_header]

    if activations is not None:
        activations.unsupported_ops_warnings(False)
        activations.uncalled_modules_warnings(False)
        activations.tracer_warnings("none")
        stats[acts_header] = activations.by_module()
        stat_columns += [acts_header]

    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(stats, require_trivial_children=True)
    stats = _pretty_statistics(stats, hide_zero=False)
    stats = _indicate_uncalled_modules(
        stats,
        flops_header,
        flops.uncalled_modules() & stats.keys(),
        uncalled_indicator="",
    )
    if activations:
        stats = _indicate_uncalled_modules(
            stats,
            acts_header,
            activations.uncalled_modules() & stats.keys(),
            uncalled_indicator="",
        )

    # Swap in shapes for parameters or delete shapes from dict
    param_shapes: Dict[str, Tuple[int, ...]] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }
    to_delete = []
    for mod in stats:
        if mod in param_shapes:
            if show_param_shapes:
                stats[mod][params_header] = str(param_shapes[mod])
            else:
                to_delete.append(mod)
    for mod in to_delete:
        del stats[mod]

    return _model_stats_table(
        statistics=stats,
        max_depth=max_depth,
        stat_columns=stat_columns,
    )

if __name__ == "__main__":
    from networks import net_factory
    model = net_factory('unet',1,1).cuda()
    # from torchvision.models import resnet50, resnet101
    # model = resnet101().cuda()
    
    model.train()
    
    inp = torch.rand(1,1,512,512).cuda()
    inp = inp.to(dtype=torch.float32)
    print(inp.shape)
    print()
    flop = FlopCountAnalysis(model, inp)
    print(flop_count_table(flop, max_depth=1))
    # print(flop_count_str(flop))
    # print(flop.total())
    
    print()
    model.eval()
    flop = FlopCountAnalysis(model, inp)
    print(flop_count_table(flop, max_depth=1))
    
    from time import time
    start_time = time()
    model.eval()
    for i in range(100):
        out = model(inp)
    out = out.detach().cpu().numpy().squeeze()
    print(out.shape)
    end_time = time()
    print('Inference speed :',(end_time - start_time)/100)