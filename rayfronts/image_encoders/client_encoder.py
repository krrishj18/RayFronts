"""An encoder that lives in another process, reached over a Unix socket.

``ClientEncoder`` implements the full ``LangSpatialGlobalImageEncoder``
interface but computes nothing: every call is forwarded to
:mod:`rayfronts.encoder_server`, which owns the real model (RADSeg, NARadio,
...) and the GPU memory it needs.  That lets the mapping server be restarted,
or several mapping servers be run, without paying the model load / VRAM cost
again, and it keeps the encoder's CUDA allocator out of the mapper's process.

Transport
---------
Both processes ``import torch.multiprocessing`` before touching the socket,
which registers torch's reductions with ``multiprocessing``'s ``ForkingPickler``
-- the same pickler ``multiprocessing.connection.Connection.send`` uses.  A CUDA
tensor therefore crosses the socket as a **CUDA IPC handle**: the bytes never
leave the GPU and never hit the socket.  That aliasing is also why the protocol
has an ack (see :mod:`rayfronts.encoder_protocol`): the server must keep the
producing tensor alive until we have copied out of it, so the client
``clone()``s every tensor it receives and only then acks.

``transport: cpu`` is the automatic fallback when either side has no CUDA. It
does NOT use torch's reductions: torch shares a CPU tensor by passing a file
descriptor through ``multiprocessing.resource_sharer``, whose handshake is
authenticated with the sending process's ``authkey``, and two independently
launched processes do not share one. CPU tensors therefore travel by value as
numpy arrays (see :mod:`rayfronts.encoder_wire`).

Thread safety
-------------
The mapping server calls the encoder from two threads: the map loop
(``encode_image_to_feat_map``) and the ROS callback thread
(``encode_prompts`` from ``add_queries``).  Every RPC therefore takes an
``RLock`` around the whole request/response/ack exchange -- a socket carries no
interleaving.
"""

import logging
import os
import threading
import time
from typing import List, Tuple

from typing_extensions import override

logger = logging.getLogger(__name__)

# Registers torch's ForkingPickler reductions (CUDA IPC included). Must happen
# before any Connection.send of a tensor.
import torch
import torch.multiprocessing  # noqa: F401  (import for its side effect)
from multiprocessing import connection as mp_connection

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder
from rayfronts import encoder_protocol as proto
from rayfronts import encoder_wire as wire


class ClientEncoder(LangSpatialGlobalImageEncoder):
  """Proxy to an encoder served by ``python3 -m rayfronts.encoder_server``.

  Attributes:
    device: See base. Where results are MATERIALISED for this process -- the
      mapper's device, not necessarily the server's.
    server_device: Where the served model actually runs.
    socket_path: Unix socket the server listens on.
    transport: Negotiated transport, "cuda_ipc" or "cpu".
    server_encoder: Class name of the encoder on the other end.
  """

  def __init__(self,
               socket: str = "/tmp/rayfronts/encoder.sock",
               transport: str = proto.TRANSPORT_AUTO,
               timeout_s: float = 300.0,
               connect_timeout_s: float = 300.0,
               authkey: str = None,
               device: str = None):
    """
    Args:
      socket: Path to the encoder server's Unix socket.
      transport: "auto" (cuda_ipc when both ends have CUDA, else cpu),
        "cuda_ipc" (fail if unavailable) or "cpu".
      timeout_s: Per-call timeout in seconds.
      connect_timeout_s: How long to wait for the socket file to appear and
        for the server to accept. The mapping server usually starts at the same
        time as the encoder server, so this defaults generously.
      authkey: Optional shared secret. Must match the server's.
      device: Where to materialise results. None = this process's own default
        ("cuda" when a GPU is visible, else "cpu"), which is what the mapper
        wants: under cuda_ipc that is the same allocation the server produced,
        under the cpu fallback it is one host->device copy per call.
    """
    super().__init__(device)

    self.socket_path = str(socket)
    if transport not in proto.VALID_TRANSPORTS:
      raise ValueError(f"transport must be one of {proto.VALID_TRANSPORTS}, "
                       f"got {transport!r}")
    self._requested_transport = transport
    self.timeout_s = float(timeout_s)
    self.connect_timeout_s = float(connect_timeout_s)
    self._authkey = None if authkey in (None, "") else str(authkey).encode()

    self._lock = threading.RLock()
    self._conn = None
    self._call_id = 0

    self.transport = proto.TRANSPORT_CPU
    self.server_encoder = None
    self.server_device = None
    self.server_feat_dim = None
    self.server_methods = ()

    self._connect()

  # ------------------------------------------------------------------ #
  # Connection / handshake
  # ------------------------------------------------------------------ #

  def _client_cuda(self) -> bool:
    try:
      return torch.cuda.is_available()
    except Exception:  # pragma: no cover - broken driver
      return False

  def _wait_for_socket(self):
    deadline = time.time() + self.connect_timeout_s
    warned = False
    while not os.path.exists(self.socket_path):
      if time.time() > deadline:
        raise TimeoutError(
          f"Encoder server socket {self.socket_path} did not appear within "
          f"{self.connect_timeout_s:.0f}s. Is "
          f"`python3 -m rayfronts.encoder_server` running?")
      if not warned:
        logger.info("Waiting for encoder server socket %s ...",
                    self.socket_path)
        warned = True
      time.sleep(0.25)

  def _connect(self):
    self._wait_for_socket()
    deadline = time.time() + self.connect_timeout_s
    last_err = None
    while True:
      try:
        self._conn = mp_connection.Client(self.socket_path, family="AF_UNIX",
                                          authkey=self._authkey)
        break
      except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        last_err = e
        if time.time() > deadline:
          raise ConnectionError(
            f"Could not connect to the encoder server at "
            f"{self.socket_path}: {e}") from e
        time.sleep(0.25)
    if last_err is not None:
      logger.info("Connected to encoder server after retrying (%s).", last_err)

    hello = proto.make_hello(self._requested_transport, self._client_cuda())
    self._conn.send(hello)
    if not self._conn.poll(self.timeout_s):
      raise TimeoutError("Encoder server did not answer the handshake.")
    ack = proto.check_hello_ack(self._conn.recv())

    self.transport = ack["transport"]
    self.server_encoder = ack.get("encoder")
    self.server_device = ack.get("device")
    self.server_feat_dim = ack.get("feat_dim")
    self.server_methods = tuple(ack.get("methods", ()))
    logger.info("ClientEncoder connected to %s on %s (transport=%s, "
                "results materialised on %s, feat_dim=%s).",
                self.server_encoder, self.server_device, self.transport,
                self.device, self.server_feat_dim)

  def close(self):
    with self._lock:
      if self._conn is None:
        return
      try:
        self._conn.send(proto.make_bye())
      except Exception:
        pass
      try:
        self._conn.close()
      except Exception:
        pass
      self._conn = None

  # Alias so the mapping server's generic shutdown loops find it.
  shutdown = close

  def __del__(self):  # pragma: no cover - interpreter teardown
    try:
      self.close()
    except Exception:
      pass

  # ------------------------------------------------------------------ #
  # RPC
  # ------------------------------------------------------------------ #

  def _prepare_arg(self, value):
    """Put an argument on the wire for the negotiated transport."""
    return wire.to_wire(value, self.transport, self.server_device)

  def _adopt(self, value):
    """Copy anything the server handed us so its memory can be released.

    Cloning is not an optimisation we can skip: under CUDA IPC the tensor we
    received is a *view onto the server's allocation*, which the server frees as
    soon as we ack.
    """
    return wire.from_wire(value, device=self.device, clone=True)

  def _rpc(self, msg):
    """Send one request, receive the response, ack it, return the value."""
    call_id = msg["id"]
    with self._lock:
      if self._conn is None:
        raise ConnectionError("ClientEncoder is closed.")
      try:
        self._conn.send(msg)
      except (BrokenPipeError, OSError) as e:
        self._conn = None
        raise ConnectionError(
          f"Encoder server died while sending call {call_id} "
          f"({msg.get('method') or msg.get('name')}): {e}") from e

      if not self._conn.poll(self.timeout_s):
        # Poison the connection. A late answer would still be sitting in the
        # socket when the NEXT call reads it, and every subsequent response
        # would be off by one -- returning plausible-looking features that
        # belong to a different frame. Better to force a reconnect.
        self._poison()
        raise TimeoutError(
          f"Encoder server did not answer call {call_id} "
          f"({msg.get('method') or msg.get('name')}) within "
          f"{self.timeout_s:.0f}s. The connection is now closed; restart the "
          f"mapping server.")
      try:
        response = self._conn.recv()
      except EOFError as e:
        self._conn = None
        raise ConnectionError(
          "Encoder server closed the connection mid-call. Check "
          "/tmp/offboard/rayfronts_encoder.log.") from e

      try:
        value = proto.check_response(response, call_id)
      except proto.EncoderServerError:
        # Server-side exception: the stream is still in sync (the server sent
        # exactly one message and is holding nothing for us), so no ack and no
        # poisoning -- the next call works.
        raise
      except proto.EncoderProtocolError:
        # An id mismatch or a malformed frame means the stream is no longer in
        # sync. Never hand the mapper a tensor from someone else's request.
        self._poison()
        raise

      local = self._adopt(value)
      # Only now may the server free what it sent.
      try:
        self._conn.send(proto.make_ack(call_id))
      except (BrokenPipeError, OSError) as e:
        self._conn = None
        raise ConnectionError(
          f"Encoder server died before acking call {call_id}: {e}") from e
      return local

  def _poison(self):
    """Close a connection that can no longer be trusted to be in sync."""
    conn, self._conn = self._conn, None
    if conn is not None:
      try:
        conn.close()
      except Exception:
        pass

  def _next_id(self) -> int:
    with self._lock:
      self._call_id += 1
      return self._call_id

  def call_remote(self, method: str, *args, **kwargs):
    """Public escape hatch: invoke any proxied method by name."""
    msg = proto.make_call(self._next_id(), method,
                          self._prepare_arg(list(args)),
                          self._prepare_arg(dict(kwargs)))
    return self._rpc(msg)

  def remote_getattr(self, name: str):
    """Fetch a (picklable, read-only) attribute of the remote encoder."""
    return self._rpc(proto.make_getattr(self._next_id(), name))

  # ------------------------------------------------------------------ #
  # ImageEncoder interface
  # ------------------------------------------------------------------ #

  @override
  def is_compatible_size(self, h: int, w: int) -> bool:
    return bool(self.call_remote("is_compatible_size", int(h), int(w)))

  @override
  def get_nearest_size(self, h, w) -> Tuple[int, int]:
    r = self.call_remote("get_nearest_size", int(h), int(w))
    return (int(r[0]), int(r[1]))

  @override
  def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor
                               ) -> torch.FloatTensor:
    return self.call_remote("encode_image_to_feat_map", rgb_image)

  @override
  def encode_image_to_vector(self, rgb_image: torch.FloatTensor
                             ) -> torch.FloatTensor:
    return self.call_remote("encode_image_to_vector", rgb_image)

  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor):
    return self.call_remote("encode_image_to_feat_map_and_vector", rgb_image)

  @override
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    return self.call_remote("encode_labels", list(labels))

  @override
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    return self.call_remote("encode_prompts", list(prompts))

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor
                                           ) -> torch.FloatTensor:
    return self.call_remote("align_spatial_features_with_language", features)

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor
                                          ) -> torch.FloatTensor:
    return self.call_remote("align_global_features_with_language", features)

  # ------------------------------------------------------------------ #
  # Semantic-segmentation style attributes, proxied on demand.
  # Only meaningful when the served encoder is an ImageSemSegEncoder.
  # ------------------------------------------------------------------ #

  @property
  def num_classes(self) -> int:
    return self.remote_getattr("num_classes")

  @property
  def cat_index_to_name(self):
    return self.remote_getattr("cat_index_to_name")

  @property
  def cat_name_to_index(self):
    return self.remote_getattr("cat_name_to_index")
