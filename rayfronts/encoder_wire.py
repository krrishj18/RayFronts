"""Putting tensors on the wire, and taking them off again.

Shared by :mod:`rayfronts.encoder_server` and
:class:`rayfronts.image_encoders.ClientEncoder`; the torch-aware half of
:mod:`rayfronts.encoder_protocol`.

Two transports, for one non-obvious reason:

* **cuda_ipc** rides torch's own ``ForkingPickler`` reductions. A CUDA tensor
  reduces to ``(device, cudaIpcMemHandle bytes, sizes, a ref-count FILENAME,
  an event handle)`` -- all ordinary picklable values -- so it crosses an
  unrelated process boundary fine and the data never leaves the GPU. This is
  the production path.
* **cpu** does NOT use torch's reductions. torch shares a CPU tensor by passing
  a *file descriptor* through ``multiprocessing.resource_sharer``, whose
  handshake is authenticated with the sending process's ``authkey``. Two
  independently launched processes have different authkeys, so that path
  fails (or worse, hangs). CPU tensors therefore travel by VALUE as a numpy
  array inside a :class:`rayfronts.encoder_protocol.TensorPayload`.

Both directions cap the payload so an accidentally huge tensor fails with our
error instead of a multiprocessing "Bad message length".
"""

import torch

from rayfronts import encoder_protocol as proto

# numpy has no bfloat16/complex32; those are sent as float32 with the original
# dtype recorded so the far side can cast back.
_UNSUPPORTED_BY_NUMPY = ("torch.bfloat16", "torch.float8_e4m3fn",
                         "torch.float8_e5m2")


def _check_size(nbytes, what):
  if nbytes > proto.MAX_PAYLOAD_BYTES:
    raise proto.EncoderProtocolError(
      f"{what} is {nbytes / 1024 ** 3:.2f} GiB, over the "
      f"{proto.MAX_PAYLOAD_BYTES / 1024 ** 3:.2f} GiB transport limit. "
      f"Reduce dataset.rgb_resolution or batch_size.")


def to_wire(value, transport, device=None):
  """Convert a value for sending under ``transport``.

  Args:
    value: tensor, or a list/tuple/dict of them, or anything picklable.
    transport: ``"cuda_ipc"`` or ``"cpu"``.
    device: CUDA device to move tensors onto for the cuda_ipc path.
  """
  if isinstance(value, torch.Tensor):
    t = value.detach()
    if transport == proto.TRANSPORT_CPU:
      t = t.cpu().contiguous()
      dtype = str(t.dtype)
      if dtype in _UNSUPPORTED_BY_NUMPY:
        t = t.to(torch.float32)
      arr = t.numpy()
      _check_size(arr.nbytes, "A tensor on the CPU transport")
      return proto.TensorPayload(arr, dtype)
    if not t.is_cuda:
      t = t.to(device if device is not None else "cuda")
    return t.contiguous()
  if isinstance(value, (list, tuple)):
    return type(value)(to_wire(v, transport, device) for v in value)
  if isinstance(value, dict):
    return {k: to_wire(v, transport, device) for k, v in value.items()}
  return value


def from_wire(value, device=None, clone=True):
  """Rebuild a received value.

  ``clone`` matters on the receiving side of a CUDA IPC handle: what arrives is
  a VIEW onto the sender's allocation, which the sender frees as soon as we
  ack. Copying before acking is the whole point of the ack.
  """
  if isinstance(value, proto.TensorPayload):
    t = torch.from_numpy(value.array)
    target_dtype = _DTYPES.get(value.dtype)
    if target_dtype is not None and t.dtype != target_dtype:
      t = t.to(target_dtype)
    if device is not None:
      t = t.to(device)
    return t
  if isinstance(value, torch.Tensor):
    t = value.clone() if clone else value
    if device is not None:
      t = t.to(device)
    return t
  if isinstance(value, (list, tuple)):
    return type(value)(from_wire(v, device, clone) for v in value)
  if isinstance(value, dict):
    return {k: from_wire(v, device, clone) for k, v in value.items()}
  return value


def _build_dtype_table():
  table = dict()
  for name in dir(torch):
    obj = getattr(torch, name)
    if isinstance(obj, torch.dtype):
      table[str(obj)] = obj
  return table


_DTYPES = _build_dtype_table()
