from __future__ import annotations
import time, importlib, inspect, functools, pathlib
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Optional, Dict, Callable, cast, Mapping
from tinygrad.helpers import ansilen, prod, DEBUG, getenv, GlobalCounters, DType, colored, merge_dicts
from tinygrad.shape.view import View
from dataclasses import dataclass
if TYPE_CHECKING: from tinygrad.lazy import LazyBuffer

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: rdna3 only has RECIP and not DIV. DIV and POW are on the chopping block
class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto(); AS_STRIDED = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); BUFFER = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[TernaryOps]]

@dataclass(frozen=True)
class MemBuffer:
  idx: int
  dtype: DType
  views: Tuple[View, ...]

@dataclass(frozen=True)
class ConstBuffer:
  val: Any
  dtype: DType
  views: Tuple[View, ...]

class LazyOp:
  __slots__ = "op", "src", "arg", "buffers", "__weakref__"
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]
  arg: Any
  buffers: Tuple[LazyBuffer, ...]
  def __init__(self, op: Op, src: Tuple[Union[LazyOp, LazyBuffer], ...], arg: Any = None):
    self.op, self.src, self.arg, self.buffers = op, src, arg, ()
    try:  # NOTE: the linearizer's key function maps the buffers to ints, and LOCAL_BUFFER is used. we don't care about buffers in these cases
      for x in src: self.buffers += x.buffers
    except AttributeError: self.buffers = ()

  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  def __eq__(self, __value: object) -> bool: return isinstance(__value, LazyOp) and self.op is __value.op and self.src == __value.src and self.arg == __value.arg
  def __hash__(self) -> int: return hash((self.op, self.src, self.arg))
  @property
  def key(self): return (self.op, tuple(map(lambda x: getattr(x, "key", x), self.src)), getattr(self.arg, "key", self.arg))

  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp, str]]) -> LazyOp: return LazyOp(self.op, tuple([y.map_buffers(real_srcs) for y in self.src]), self.arg)
  def get_lazyops(self) -> List[LazyOp]: return [self] + [item for x in self.src for item in x.get_lazyops()]

  def replace_with_movement_ops(self:LazyOp, ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> 'LazyBuffer':
    assert self.op in BinaryOps or self.op in UnaryOps or self.op in TernaryOps
    srcs = [z.replace_with_movement_ops(ops) for z in self.src]
    return srcs[0].e(self.op, *srcs[1:], arg=self.arg)   # type: ignore

  @property
  def st(self): raise NotImplementedError
  @property
  def children(self): raise NotImplementedError
  @property
  def shape(self): raise NotImplementedError
  @property
  def realized(self): raise NotImplementedError
  @property
  def optype(self): raise NotImplementedError
  def realize(self): raise NotImplementedError

  # movement ops
  def reshape(self, _): raise NotImplementedError
  def pad(self, _): raise NotImplementedError
  def expand(self, _): raise NotImplementedError
  def permute(self, _): raise NotImplementedError
  def shrink(self, _): raise NotImplementedError
  def stride(self, _): raise NotImplementedError

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
  def canonicalize(self, device:Optional[str]) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "") if device is not None else self.DEFAULT
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __getitem__(self, x:str) -> Union[Interpreted, Compiled]:
    x = x.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "buffer") and x in self._buffers][0]
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, None)
    if device_from_env: return device_from_env
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()

# **************** for Interpreted Buffers ****************

def apply_shapetracker(fxn_for_op, ret, views):
  for v in views:
    real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
    real_offset = v.offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
    # first, we apply the offset
    # then, we make it the correct shape
    # then, we apply permutations
    # TODO: don't use as_strided
    ret = fxn_for_op[MovementOps.AS_STRIDED](ret, ([s if st != 0 else 1 for s,st in zip(real_shape, v.strides)], v.strides, real_offset))
    # then, we apply pre expand pads
    if v.mask is not None:
      pre_expand_pads = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      post_expand_pads = tuple((x,s-y) if st == 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      if any(x != (0,0) for x in pre_expand_pads):
        ret = fxn_for_op[MovementOps.PAD](ret, pre_expand_pads)
        real_shape = tuple(x+s[0]+s[1] for x,s in zip(real_shape, pre_expand_pads))
    # then, we do any expands
    if any(s != 1 and st == 0 for s,st in zip(real_shape, v.strides)): ret = fxn_for_op[MovementOps.EXPAND](ret, real_shape)
    # lastly, we apply post expand pads
    if v.mask is not None and any(x != (0,0) for x in post_expand_pads): ret = fxn_for_op[MovementOps.PAD](ret, post_expand_pads)
  return ret

class Interpreted:
  def __init__(self, buffer, fxn_for_op: Dict[Op, Callable], to_underlying=lambda x: x._buf, from_underlying=None):
    self.buffer, self.fxn_for_op, self.to_underlying = buffer, fxn_for_op, to_underlying
    self.from_underlying = buffer if from_underlying is None else from_underlying
    self.synchronize = lambda: None
    self.codegen = None

  def exec_ast(self, ast:LazyOp, output=None, inputs=None, context=None, **kwargs):
    if ast.op == LoadOps.BUFFER and LoadOps.BUFFER not in self.fxn_for_op:
      assert inputs[ast.arg.idx-1].dtype == ast.arg.dtype, "dtype mismatch"
      return self.from_underlying(apply_shapetracker(self.fxn_for_op, self.to_underlying(inputs[ast.arg.idx-1]), ast.arg.views))
    if TernaryOps.MULACC in self.fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, ast.src[0].src, ast.arg)
    created_context = context is None
    if context is None: context = dict()
    if not created_context and ast in context: return context[ast]
    srcs = [self.exec_ast(cast(LazyOp, x), inputs=inputs, context=context, **kwargs) for x in ast.src]
    if DEBUG >= 3: st = time.perf_counter()
    ret = self.from_underlying(self.fxn_for_op[ast.op](*([self.to_underlying(x) for x in srcs] + ([ast.arg] if ast.arg is not None else []))))
    if output is not None and ret.dtype != output.dtype and UnaryOps.CAST in self.fxn_for_op: ret = self.from_underlying(self.fxn_for_op[UnaryOps.CAST](self.to_underlying(ret), (output.dtype, False))) # Do manual casting of ret if it does not match the required output dtype.
    if DEBUG >= 5 or (self.buffer != FlopCounter and DEBUG >= 3): print(f"*** {'exec' if created_context else '    '} {GlobalCounters.mem_used/1e9:5.2f} GB {(time.perf_counter()-st)*1e3:7.2f} ms op: {ast.op:20s} out({ret.dtype.name}): {str(ret._buf.shape) if hasattr(ret._buf, 'shape') else str(len(ret._buf)):30s} in({len(srcs)}):", list(set(x._buf.shape if hasattr(x._buf, 'shape') else len(x._buf) for x in srcs)), ast.arg if ast.arg is not None else "")
    if not created_context: context[ast] = ret
    if output is not None and output.output_buffer is not None:
      # TODO: does this check have any meaning anymore?
      # It fails on things like batchnorm initted with zeros
      #assert output.output_buffer.size == ret.size, f"size mismatch, {output.output_buffer.size} != {ret.size}"
      assert output.output_buffer.dtype == ret.dtype
      output.output_buffer._buf = ret._buf
      return output.output_buffer
    return ret

# --teenygrad--

class FlopCounter:
  def __init__(self, tup:Tuple[Tuple[int, ...], DType, int]): self.shape, self.dtype, self.flops, self._buf = *tup, self
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
shape_fxn_for_op: Dict[Op, Callable] = {
  LoadOps.BUFFER: lambda arg: (arg.views[-1].shape, arg.dtype, 0), LoadOps.CONST: lambda arg: (arg.views[-1].shape, arg.dtype, 0),
  UnaryOps.CAST: lambda self,arg: (self.shape, arg[0], self.consume_flops()),   # cast uses no flops
  **{op:lambda self: (self.shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in UnaryOps if op != UnaryOps.CAST},
  **{op:lambda self,y: (self.shape, max(self.dtype, y.dtype), self.consume_flops() + y.consume_flops() + prod(self.shape)) for op in BinaryOps},
  **{op:lambda self,new_shape: (new_shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in ReduceOps},
  TernaryOps.WHERE: lambda self,y,z: (self.shape, self.dtype, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape))}
InterpretedFlopCounter = Interpreted(FlopCounter, shape_fxn_for_op, lambda x: x)
def get_lazyop_info(ast:LazyOp) -> FlopCounter: return InterpretedFlopCounter.exec_ast(ast)

# **************** for Compiled Buffers ****************

from tinygrad.runtime.lib import RawBuffer, RawConst
from tinygrad.shape.symbolic import Variable, sym_infer

class ASTRunner:
  def __init__(self, name, prg, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None, op_estimate=0, mem_estimate=0, display_name:Optional[str]=None, runtime_args:Optional[dict]=None):
    if DEBUG >= 4 and (runtime_args is None or 'binary' not in runtime_args or not runtime_args['binary']): print(prg)
    self.name, self.prg, self.global_size, self.local_size, self.op_estimate, self.mem_estimate, self.display_name, self.runtime_args = name, prg, global_size, local_size, op_estimate, mem_estimate, display_name, runtime_args if runtime_args is not None else {}

  def build(self, runtime):
    self.clprg = runtime(self.name, self.prg, **self.runtime_args)
    return self

  def exec(self, rawbufs, var_vals:Optional[Dict[Variable, int]]=None, force_wait=False, optimizing=False) -> Optional[float]:
    from tinygrad.jit import CacheCollector
    #rawbufs = dedup([x.realized for x in bufs if buf_is_kernel_arg(x)])
    if not optimizing: CacheCollector.add(self, rawbufs, var_vals if var_vals is not None else {})
    return self(rawbufs, var_vals, force_wait=force_wait)

  def __call__(self, rawbufs:List[RawBuffer], var_vals:Optional[Dict[Variable, int]]=None, jit=False, force_wait=False) -> Optional[float]:
    if var_vals is None: var_vals = {}
    global_size = [sym_infer(sz, var_vals) for sz in self.global_size] if self.global_size is not None else self.global_size
    local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else self.local_size
    if et := self.clprg((global_size + [1]*(3-len(global_size))) if global_size is not None else None,
                        (local_size + [1]*(3-len(local_size))) if local_size is not None else None,
                        *rawbufs, *var_vals.values(), wait=force_wait or DEBUG>=1): GlobalCounters.time_sum_s += et
    op_estimate = sym_infer(self.op_estimate, var_vals)
    if DEBUG >= 2:
      print(f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(self.display_name+' '*(33-ansilen(self.display_name))) if self.display_name is not None else self.name:33s} arg {len(rawbufs):3d} sz {str(global_size):18s} {str(local_size):12s} OPs {int(op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB " +
            (str() if et is None else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {self.mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))
    GlobalCounters.kernel_count += 1
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += self.mem_estimate
    return et

class Compiled:
  def __init__(self, buffer: Type[RawBuffer], linearizer_opts, renderer, runtime, synchronize=lambda: None):
    self.buffer, self.linearizer_opts, self.renderer, self.runtime, self.synchronize = buffer, linearizer_opts, renderer, runtime, synchronize
    self.method_cache: Dict[Any, ASTRunner] = {}

  def to_program(self, k):
    k.linearize()
    src = self.renderer(k.function_name, k.uops)
    if len(src) == 3:
      return ASTRunner(k.function_name, src[0], k.global_size, src[1],display_name=k.display_name, runtime_args=src[2]).build(self.runtime)
    return ASTRunner(k.function_name, src, k.global_size, k.local_size,
                     op_estimate=k.info.flops, mem_estimate=k.mem_estimate,
                     display_name=k.display_name, runtime_args={"binary": False}).build(self.runtime)

  def exec_ast(self, ast:LazyOp, output, inputs, **kwargs):
    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # NOTE: this is pretty wrong actually, who knows where else this buffer is used?
    output.realized = output.output_buffer
    if output.realized:
      if output.realized.__class__ is RawConst: output.realized = None  # can't assign to RawConst
      for a in ast.buffers:
        if a.realized == output.realized and not a.st.contiguous:
          output.realized = None
          break

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    if not output.realized: output.realized = self.buffer(prod((s if isinstance(s, int) else s.max for s in output.shape)), output.dtype, **kwargs)
    else:
      from tinygrad.jit import CacheCollector
      CacheCollector._mark_output_buffer(output.output_buffer)
    # update the output var_vals from src
    output.var_vals = dict(sorted(merge_dicts([buf.var_vals for buf in ast.buffers]).items(), key=lambda kv:cast(Variable,kv[0]).key))

    #if DEBUG >= 4:
    #  from extra.utils import print_tree
    #  print_tree(ast)

    from tinygrad.codegen.linearizer import Linearizer
    k = Linearizer(ast, self.linearizer_opts)

    # compilation time
    def get_program():
      from tinygrad.codegen.search import kernel_optimize
      if getenv("KOPT"): kernel_optimize(k, lambda: Linearizer(ast, self.linearizer_opts), self.to_program)
      elif not getenv("NOOPT"): k.hand_coded_optimizations()
      return self.to_program(k)

    if hasattr(k, 'key') and getenv("ENABLE_METHOD_CACHE", 1):
      if k.key not in self.method_cache: self.method_cache[k.key] = get_program()
      prg = self.method_cache[k.key]
    else:
      prg = get_program()

    if prg.name == getenv("PRINT_PRG", ''): print(prg.prg)

    prg.exec([output.realized]+inputs, var_vals=output.var_vals)
    return output.realized
