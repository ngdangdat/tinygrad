from __future__ import annotations
from typing import Tuple, Optional, cast, Union, Any
from tinygrad.ops import LazyOp, UnaryOps, BinaryOps, TernaryOps, LoadOps
from tinygrad.helpers import DType, dtypes, all_int
from tinygrad.runtime.lib import RawBuffer
from tinygrad.runtime.ops_cpu import RawNumpyBuffer
import numpy as np

class LazyBuffer:
  def __init__(self, device:str, shape:Tuple[int, ...], op:Optional[LazyOp], dtype:DType, src:Optional[RawBuffer]=None):
    self.device, self.shape, self.dtype = device, shape, dtype
    if op: self.op = op
    if src: self.realized = src

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer("CPU", x.shape, None, dtypes.from_np(x.dtype), RawNumpyBuffer.fromCPU(x))

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    return LazyBuffer(device, tuple(shape), LazyOp(op, tuple() if src is None else (src,), arg), dtype)

  def contiguous(self:LazyBuffer) -> LazyBuffer:
    if not self.realized and self.op.op == LoadOps.CONTIGUOUS: return self  # two CONTIGUOUS in a row is one
    return self.loadop(LoadOps.CONTIGUOUS, self.shape, self.dtype, src=self)

  def toCPU(self) -> np.ndarray:
    assert self.dtype.np, f"{self.dtype} is not supported in toCPU"
    self_casted = self.e(UnaryOps.CAST, arg=(dtypes.from_np(self.dtype.np), False)) if dtypes.from_np(self.dtype.np) != self.dtype else self
    realized = self_casted.contiguous().realize().realized
    assert all_int(self.shape), f"no toCPU if shape is symbolic, {self.shape=}"
    return cast(RawBuffer, realized).toCPU().reshape(self.shape)

  def e(self:LazyBuffer, op:Union[UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    pass