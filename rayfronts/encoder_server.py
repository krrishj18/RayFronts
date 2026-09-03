"""Serves ONE image encoder over a Unix socket to many mapping servers.

    python3 -m rayfronts.encoder_server encoder=radseg \\
            encoder_server.socket=/tmp/rayfronts/encoder.sock

The model is loaded once and stays resident; each client
(:class:`rayfronts.image_encoders.ClientEncoder`) gets its own thread, and the
encoder itself is called under a single lock because a torch model is not
re-entrant.

CUDA results are handed back **by IPC handle**, not by value: this process holds
a strong reference to every result until the client acks it, at which point the
reference is dropped and the allocation may be reused.  Dropping it earlier
would hand the client a view onto memory that torch is free to overwrite --
which is exactly the bug the ack exists to prevent.  CPU results travel by
value instead; :mod:`rayfronts.encoder_wire` explains why they cannot use
torch's own reductions.

See :mod:`rayfronts.encoder_protocol` for the message shapes.
"""

import atexit
import logging
import os
import random
import signal
import socket as _socket
import threading
import traceback

import hydra
import numpy as np

# Registers torch's ForkingPickler reductions (CUDA IPC included) with
# multiprocessing. Must be imported before any Connection.send of a tensor.
import torch
import torch.multiprocessing  # noqa: F401  (import for its side effect)
from multiprocessing import connection as mp_connection

from rayfronts import encoder_protocol as proto
from rayfronts import encoder_wire as wire

logger = logging.getLogger(__name__)


class EncoderServer:
  """Owns the encoder and the listening socket.

  Attributes:
    cfg: The hydra config.
    encoder: The instantiated encoder.
    socket_path: Unix socket path we listen on.
  """

  def __init__(self, cfg):
    self.cfg = cfg
    scfg = cfg.encoder_server
    self.socket_path = str(scfg.socket)
    self._authkey = (None if getattr(scfg, "authkey", None) in (None, "")
                     else str(scfg.authkey).encode())
    self._max_clients = int(getattr(scfg, "max_clients", 8))

    encoder_kwargs = dict()
    device = getattr(scfg, "device", None)
    if device is not None:
      encoder_kwargs["device"] = str(device)
    self.encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
    self.encoder_class = type(self.encoder).__name__
    self.device = str(getattr(self.encoder, "device", "cpu"))
    self.server_cuda = bool(self.device.startswith("cuda")
                            and torch.cuda.is_available())

    self._encoder_lock = threading.Lock()
    # Results sent but not yet acked, per connection. Kept alive on purpose:
    # under CUDA IPC the client holds a view onto OUR allocation until it has
    # cloned it. Exposed (rather than a local) so the invariant is testable.
    self._pending = dict()
    self._pending_lock = threading.Lock()
    self._listener = None
    self._threads = []
    self._stop = threading.Event()
    self.feat_dim = None

    self._warmup(getattr(scfg, "warmup", None))

  # ------------------------------------------------------------------ #

  @torch.no_grad()
  def _warmup(self, warmup):
    """Optional forward pass so the first real frame is not the slow one."""
    if not warmup:
      return
    try:
      h, w = (int(warmup[0]), int(warmup[1]))
    except (TypeError, ValueError, IndexError):
      logger.warning("encoder_server.warmup must be [h, w]; got %r. Skipping.",
                     warmup)
      return
    logger.info("Warming up %s at %dx%d ...", self.encoder_class, h, w)
    try:
      img = torch.zeros((1, 3, h, w), dtype=torch.float32, device=self.device)
      feat = self.encoder.encode_image_to_feat_map(img)
      self.feat_dim = int(feat.shape[1])
      logger.info("Warmup done. Feature dim = %s.", self.feat_dim)
    except Exception:
      logger.exception("Warmup failed; continuing without it.")

  def _methods(self):
    return [m for m in sorted(proto.CALLABLE_METHODS)
            if callable(getattr(self.encoder, m, None))]

  # ------------------------------------------------------------------ #
  # Socket lifecycle
  # ------------------------------------------------------------------ #

  def _prepare_socket(self):
    d = os.path.dirname(self.socket_path)
    if d:
      os.makedirs(d, exist_ok=True)
    if os.path.exists(self.socket_path):
      # A leftover file from a crashed server: refuse only if someone is
      # actually listening on it.
      probe = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
      try:
        probe.settimeout(0.5)
        probe.connect(self.socket_path)
      except OSError:
        logger.warning("Removing stale socket %s", self.socket_path)
        os.unlink(self.socket_path)
      else:
        probe.close()
        raise RuntimeError(
          f"Another encoder server is already listening on "
          f"{self.socket_path}.")
      finally:
        try:
          probe.close()
        except OSError:
          pass

  def serve_forever(self):
    self._prepare_socket()
    self._listener = mp_connection.Listener(
      self.socket_path, family="AF_UNIX", backlog=self._max_clients,
      authkey=self._authkey)
    try:
      os.chmod(self.socket_path, 0o660)
    except OSError:
      pass
    logger.info("Encoder server ready: %s on %s, listening at %s",
                self.encoder_class, self.device, self.socket_path)

    try:
      while not self._stop.is_set():
        try:
          conn = self._listener.accept()
        except OSError:
          if self._stop.is_set():
            break
          raise
        t = threading.Thread(target=self._serve_client, args=(conn,),
                             name="encoder_client", daemon=True)
        t.start()
        self._threads = [x for x in self._threads if x.is_alive()]
        self._threads.append(t)
    finally:
      self.shutdown()

  def shutdown(self):
    if self._stop.is_set():
      return
    self._stop.set()
    if self._listener is not None:
      try:
        self._listener.close()
      except OSError:
        pass
      self._listener = None
    if os.path.exists(self.socket_path):
      try:
        os.unlink(self.socket_path)
      except OSError:
        pass
    logger.info("Encoder server shut down.")

  # ------------------------------------------------------------------ #
  # Per-client loop
  # ------------------------------------------------------------------ #

  def pending_count(self, peer=None) -> int:
    """How many sent-but-unacked results we are keeping alive. For tests."""
    with self._pending_lock:
      if peer is not None:
        return len(self._pending.get(peer, {}))
      return sum(len(v) for v in self._pending.values())

  def _serve_client(self, conn):
    transport = proto.TRANSPORT_CPU
    peer = id(conn)
    with self._pending_lock:
      pending = self._pending.setdefault(peer, dict())
    try:
      hello = conn.recv()
      if not isinstance(hello, dict) or hello.get("kind") != proto.KIND_HELLO:
        conn.send(proto.make_error(None, "Expected a hello message."))
        return
      if hello.get("version") != proto.PROTOCOL_VERSION:
        conn.send(proto.make_error(
          None, f"Protocol mismatch: client v{hello.get('version')}, "
                f"server v{proto.PROTOCOL_VERSION}."))
        return
      try:
        transport = proto.negotiate_transport(
          hello.get("transport", proto.TRANSPORT_AUTO),
          bool(hello.get("client_cuda", False)), self.server_cuda)
      except proto.EncoderProtocolError as e:
        conn.send(proto.make_error(None, str(e)))
        return
      conn.send(proto.make_hello_ack(
        encoder_class=self.encoder_class, device=self.device,
        transport=transport, feat_dim=self.feat_dim,
        resolution=getattr(self.encoder, "input_resolution", None),
        methods=self._methods()))
      logger.info("Client %s connected (transport=%s).", peer, transport)

      while not self._stop.is_set():
        try:
          msg = conn.recv()
        except (EOFError, ConnectionResetError):
          break
        if not isinstance(msg, dict):
          conn.send(proto.make_error(None, f"Malformed message {msg!r}"))
          continue
        kind = msg.get("kind")

        if kind == proto.KIND_BYE:
          break

        if kind == proto.KIND_ACK:
          with self._pending_lock:
            pending.pop(msg.get("id"), None)
          continue

        if kind in (proto.KIND_CALL, proto.KIND_GETATTR):
          call_id = msg.get("id")
          try:
            value = self._dispatch(msg, transport)
          except Exception as e:  # noqa: BLE001 - report anything to the client
            logger.exception("Call %s failed", msg)
            conn.send(proto.make_error(call_id, f"{type(e).__name__}: {e}",
                                       traceback.format_exc()))
            continue
          with self._pending_lock:
            pending[call_id] = value
          try:
            conn.send(proto.make_result(call_id, value))
          except Exception:
            with self._pending_lock:
              pending.pop(call_id, None)
            raise
          continue

        conn.send(proto.make_error(msg.get("id"),
                                   f"Unknown message kind {kind!r}"))
    except (EOFError, ConnectionResetError, BrokenPipeError):
      pass
    except Exception:
      logger.exception("Client %s handler crashed", peer)
    finally:
      with self._pending_lock:
        self._pending.pop(peer, None)
      try:
        conn.close()
      except OSError:
        pass
      logger.info("Client %s disconnected.", peer)

  # torch.no_grad() rather than torch.inference_mode(): an "inference tensor"
  # carries restrictions (no version counter, no autograd interaction) that
  # follow it out of this process, and the result here is about to be shared
  # by CUDA IPC and then cloned by the client. no_grad gives an ordinary
  # requires_grad=False tensor at the same forward-only cost. mapping_server
  # keeps inference_mode because its tensors never leave the process.
  @torch.no_grad()
  def _dispatch(self, msg, transport):
    if msg["kind"] == proto.KIND_GETATTR:
      name = msg.get("name")
      if name not in proto.GETTABLE_ATTRS:
        raise AttributeError(f"Attribute {name!r} is not proxied.")
      return wire.to_wire(getattr(self.encoder, name), transport, self.device)

    method = msg.get("method")
    if method not in proto.CALLABLE_METHODS:
      raise AttributeError(f"Method {method!r} is not proxied.")
    fn = getattr(self.encoder, method, None)
    if not callable(fn):
      raise AttributeError(
        f"{self.encoder_class} does not implement {method!r}.")

    args = wire.from_wire(msg.get("args", []), device=self.device,
                          clone=False)
    kwargs = wire.from_wire(msg.get("kwargs", {}), device=self.device,
                            clone=False)
    with self._encoder_lock:
      out = fn(*args, **kwargs)
    return wire.to_wire(out, transport, self.device)


@hydra.main(version_base=None, config_path="configs",
            config_name="encoder_server")
def main(cfg=None):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
  if getattr(cfg, "seed", -1) is not None and cfg.seed >= 0:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

  server = EncoderServer(cfg)
  atexit.register(server.shutdown)

  def _handler(sig, frame):
    logger.info("Received signal %s. Shutting down encoder server.", sig)
    server.shutdown()
    raise SystemExit(0)

  signal.signal(signal.SIGINT, _handler)
  signal.signal(signal.SIGTERM, _handler)

  try:
    server.serve_forever()
  except SystemExit:
    pass
  except Exception:
    server.shutdown()
    raise


if __name__ == "__main__":
  main()
