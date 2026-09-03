"""Wire protocol shared by :mod:`rayfronts.encoder_server` and ``ClientEncoder``.

Stdlib only — no torch — so the framing rules can be unit tested on a bare
python.  Tensors themselves ride inside the message dicts and are pickled by
``multiprocessing.connection``'s ``ForkingPickler``; when the *server* process
has done ``import torch.multiprocessing`` those pickles carry **CUDA IPC
handles** rather than data, so a GPU feature map never leaves the GPU.

Message flow (one call)::

    client                                   server
      | ---- {kind:"hello", version}  ------->|
      |<---- {kind:"hello_ack", ...}  --------|
      | ---- {kind:"call", id, method,...} -->|
      |                                       |  run encoder, keep result alive
      |<---- {kind:"result", id, value} ------|
      |  clone() every tensor it received     |
      | ---- {kind:"ack", id} --------------->|
      |                                       |  drop the strong reference
      |                                       |  -> CUDA IPC memory can be freed

The ack is what makes the CUDA-IPC path safe: the producer must keep the
storage alive until the consumer has finished reading it, and the consumer
cannot "finish" without copying, hence *clone before ack*.
"""

from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = 1

KIND_HELLO = "hello"
KIND_HELLO_ACK = "hello_ack"
KIND_CALL = "call"
KIND_RESULT = "result"
KIND_ERROR = "error"
KIND_ACK = "ack"
KIND_BYE = "bye"

TRANSPORT_AUTO = "auto"
TRANSPORT_CUDA_IPC = "cuda_ipc"
TRANSPORT_CPU = "cpu"
VALID_TRANSPORTS = (TRANSPORT_AUTO, TRANSPORT_CUDA_IPC, TRANSPORT_CPU)

# Methods a mapper / mapping server may ask the encoder to perform remotely.
# ClientEncoder refuses anything outside this set so a corrupt or hostile
# message cannot turn the socket into an arbitrary-attribute RPC.
CALLABLE_METHODS = frozenset({
  "encode_image_to_feat_map",
  "encode_image_to_vector",
  "encode_image_to_feat_map_and_vector",
  "encode_labels",
  "encode_prompts",
  "align_spatial_features_with_language",
  "align_global_features_with_language",
  "is_compatible_size",
  "get_nearest_size",
  "insert_labels_into_templates",
})

# Read-only attributes the client may fetch.  Anything not picklable is
# reported as an error by the server rather than crashing it.
GETTABLE_ATTRS = frozenset({
  "device",
  "num_classes",
  "cat_index_to_name",
  "cat_name_to_index",
  "input_resolution",
  "return_radio_features",
  "model_version",
  "predict",
  "eps",
})

KIND_GETATTR = "getattr"

# Generous guard so a mis-sized tensor fails with our error rather than a
# multiprocessing "Bad message length" or an OOM in the pickler.
MAX_PAYLOAD_BYTES = 2 * 1024 ** 3  # 2 GiB


class TensorPayload:
  """A CPU tensor travelling by VALUE (a numpy array), not by handle.

  torch shares a CPU tensor by passing a *file descriptor* through
  ``multiprocessing.resource_sharer``, whose handshake is authenticated with
  the sending process's ``authkey`` -- and two independently launched processes
  do not share one. So the cpu transport pickles the bytes instead. CUDA IPC
  needs no such workaround: its handle and ref-count filename are ordinary
  picklable values, which is why the GPU path stays zero-copy.

  Attributes:
    array: the numpy array carrying the data.
    dtype: str(torch dtype) of the ORIGINAL tensor, so a dtype numpy cannot
      represent (bfloat16, the float8s) can be cast back on arrival.
  """

  __slots__ = ("array", "dtype")

  def __init__(self, array, dtype):
    self.array = array
    self.dtype = str(dtype)

  def __repr__(self):
    return (f"TensorPayload(shape={getattr(self.array, 'shape', None)}, "
            f"dtype={self.dtype})")


class EncoderProtocolError(RuntimeError):
  """Raised on a malformed / out-of-order message."""


class EncoderServerError(RuntimeError):
  """Raised on the client when the server reported an exception."""


def make_hello(transport: str = TRANSPORT_AUTO,
               client_cuda: bool = False) -> Dict[str, Any]:
  if transport not in VALID_TRANSPORTS:
    raise ValueError(f"transport must be one of {VALID_TRANSPORTS}, "
                     f"got {transport!r}")
  return {"kind": KIND_HELLO, "version": PROTOCOL_VERSION,
          "transport": transport, "client_cuda": bool(client_cuda)}


def make_hello_ack(encoder_class: str,
                   device: str,
                   transport: str,
                   feat_dim: Optional[int] = None,
                   resolution=None,
                   methods: Optional[List[str]] = None) -> Dict[str, Any]:
  return {"kind": KIND_HELLO_ACK, "version": PROTOCOL_VERSION,
          "encoder": encoder_class, "device": device, "transport": transport,
          "feat_dim": feat_dim, "resolution": resolution,
          "methods": sorted(methods) if methods else []}


def make_call(call_id: int, method: str, args=(), kwargs=None,
              want_device: str = "same") -> Dict[str, Any]:
  if method not in CALLABLE_METHODS:
    raise EncoderProtocolError(f"Method {method!r} is not proxied. "
                               f"Allowed: {sorted(CALLABLE_METHODS)}")
  return {"kind": KIND_CALL, "id": int(call_id), "method": method,
          "args": list(args), "kwargs": dict(kwargs or {}),
          "want_device": want_device}


def make_getattr(call_id: int, name: str) -> Dict[str, Any]:
  if name not in GETTABLE_ATTRS:
    raise EncoderProtocolError(f"Attribute {name!r} is not proxied. "
                               f"Allowed: {sorted(GETTABLE_ATTRS)}")
  return {"kind": KIND_GETATTR, "id": int(call_id), "name": name}


def make_result(call_id: int, value: Any) -> Dict[str, Any]:
  return {"kind": KIND_RESULT, "id": int(call_id), "value": value}


def make_error(call_id: Optional[int], message: str,
               traceback_str: str = "") -> Dict[str, Any]:
  return {"kind": KIND_ERROR, "id": None if call_id is None else int(call_id),
          "error": str(message), "traceback": traceback_str}


def make_ack(call_id: int) -> Dict[str, Any]:
  return {"kind": KIND_ACK, "id": int(call_id)}


def make_bye() -> Dict[str, Any]:
  return {"kind": KIND_BYE}


def check_response(msg: Any, call_id: int) -> Any:
  """Validate a server response for ``call_id`` and return its value.

  Raises :class:`EncoderServerError` for a server-side exception and
  :class:`EncoderProtocolError` for anything malformed or mismatched.
  """
  if not isinstance(msg, dict) or "kind" not in msg:
    raise EncoderProtocolError(f"Malformed response: {msg!r}")
  kind = msg["kind"]
  if kind == KIND_ERROR:
    raise EncoderServerError(
      f"{msg.get('error', 'unknown error')}\n{msg.get('traceback', '')}".strip())
  if kind != KIND_RESULT:
    raise EncoderProtocolError(f"Expected a {KIND_RESULT!r} message, "
                               f"got {kind!r}")
  if msg.get("id") != call_id:
    raise EncoderProtocolError(
      f"Response id mismatch: expected {call_id}, got {msg.get('id')}")
  return msg["value"]


def check_hello_ack(msg: Any) -> Dict[str, Any]:
  if not isinstance(msg, dict):
    raise EncoderProtocolError(f"Malformed handshake response: {msg!r}")
  if msg.get("kind") == KIND_ERROR:
    raise EncoderServerError(msg.get("error", "unknown handshake error"))
  if msg.get("kind") != KIND_HELLO_ACK:
    raise EncoderProtocolError(
      f"Expected {KIND_HELLO_ACK!r}, got {msg.get('kind')!r}")
  if msg.get("version") != PROTOCOL_VERSION:
    raise EncoderProtocolError(
      f"Encoder server speaks protocol v{msg.get('version')}, "
      f"this client speaks v{PROTOCOL_VERSION}. Rebuild one of them.")
  return msg


def negotiate_transport(requested: str, client_cuda: bool,
                        server_cuda: bool) -> str:
  """Decide the transport both ends will use.

  ``auto``     -> ``cuda_ipc`` when both ends have CUDA, else ``cpu``.
  ``cuda_ipc`` -> error unless both ends have CUDA.
  ``cpu``      -> always ``cpu``.
  """
  if requested not in VALID_TRANSPORTS:
    raise EncoderProtocolError(f"Unknown transport {requested!r}")
  if requested == TRANSPORT_CPU:
    return TRANSPORT_CPU
  if requested == TRANSPORT_CUDA_IPC:
    if not (client_cuda and server_cuda):
      raise EncoderProtocolError(
        "transport=cuda_ipc requested but CUDA is not available on "
        f"{'the client' if not client_cuda else 'the server'}.")
    return TRANSPORT_CUDA_IPC
  return TRANSPORT_CUDA_IPC if (client_cuda and server_cuda) else TRANSPORT_CPU
