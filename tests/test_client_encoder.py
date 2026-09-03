"""ClientEncoder <-> encoder_server, end to end.

Needs torch (the container has it, the host does not) but no GPU: the CPU
transport is exercised by default and the CUDA-IPC path is behind the ``cuda``
mark.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

from conftest import HAVE_TORCH, REPO, ensure_pythonpath

pytestmark = [pytest.mark.torch,
              pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")]

if HAVE_TORCH:
  ensure_pythonpath()
  import torch
  from omegaconf import OmegaConf
  from rayfronts import encoder_protocol as proto
  from rayfronts import encoder_wire as wire
  from rayfronts.encoder_server import EncoderServer
  from rayfronts.image_encoders import ClientEncoder, DummyEncoder


def _dummy_cfg(socket_path, device="cpu", warmup=None):
  return OmegaConf.create({
    "encoder": {"_target_": "rayfronts.image_encoders.DummyEncoder",
                "feat_dim": 16, "lang_dim": 8, "patch_size": 16, "seed": 17},
    "encoder_server": {"socket": str(socket_path), "device": device,
                       "warmup": warmup, "max_clients": 8, "authkey": None},
  })


class _ServerProc:
  """`python3 -m rayfronts.encoder_server` in a subprocess."""

  def __init__(self, socket_path, overrides=()):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    self.socket_path = str(socket_path)
    self.log = open(str(socket_path) + ".log", "wb")
    self.proc = subprocess.Popen(
      [sys.executable, "-m", "rayfronts.encoder_server",
       "encoder=dummy", f"encoder_server.socket={socket_path}", *overrides],
      cwd=str(REPO), env=env, stdout=self.log, stderr=subprocess.STDOUT)

  def wait_ready(self, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
      if os.path.exists(self.socket_path):
        return
      if self.proc.poll() is not None:
        raise RuntimeError(
          f"encoder_server exited with {self.proc.returncode}:\n"
          f"{open(self.socket_path + '.log').read()}")
      time.sleep(0.1)
    raise TimeoutError("encoder_server never created its socket:\n"
                       + open(self.socket_path + ".log").read())

  def kill(self):
    if self.proc.poll() is None:
      self.proc.kill()
      self.proc.wait(timeout=20)

  def stop(self):
    if self.proc.poll() is None:
      self.proc.terminate()
      try:
        self.proc.wait(timeout=20)
      except subprocess.TimeoutExpired:
        self.proc.kill()
    self.log.close()


# Module scoped: starting the server costs a full torch+hydra import, and the
# tests that need a FRESH one (server death, socket cleanup) build their own.
@pytest.fixture(scope="module")
def server(tmp_path_factory):
  s = _ServerProc(tmp_path_factory.mktemp("encsrv") / "encoder.sock")
  try:
    s.wait_ready()
    yield s
  finally:
    s.stop()


@pytest.fixture(scope="module")
def client(server):
  c = ClientEncoder(socket=server.socket_path, transport="auto",
                    timeout_s=60, connect_timeout_s=60)
  try:
    yield c
  finally:
    c.close()


@pytest.fixture(scope="module")
def local(client):
  """Reference encoder on the SAME device the server chose.

  The served DummyEncoder takes the server process's own default device, which
  is cuda whenever the container can see a GPU -- and conv2d on Ampere+ runs in
  TF32 by default, roughly 1e-3 off fp32. Comparing a GPU roundtrip against a
  CPU reference therefore fails with ~2% relative error on ~100% of elements,
  which reads exactly like a corrupted transport and is nothing of the kind.
  Same device, same kernels, same numbers -- and a REAL protocol fault (a
  response matched to the wrong request, a buffer reused before the client
  cloned it) still shows up loudly, because it produces an entirely different
  tensor rather than one that is 2% off.
  """
  return DummyEncoder(device=client.server_device or "cpu")


# --------------------------------------------------------------------------- #
# Handshake
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_handshake_reports_the_served_encoder(client):
  assert client.server_encoder == "DummyEncoder"
  assert client.transport in ("cpu", "cuda_ipc")
  assert "encode_image_to_feat_map" in client.server_methods
  assert "encode_labels" in client.server_methods


@pytest.mark.slow
def test_client_is_a_lang_spatial_global_encoder(client):
  from rayfronts.image_encoders import LangSpatialGlobalImageEncoder
  assert isinstance(client, LangSpatialGlobalImageEncoder)


# --------------------------------------------------------------------------- #
# Every proxied method roundtrips exactly
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_encode_image_to_feat_map(client, local):
  torch.manual_seed(3)
  img = torch.rand(2, 3, 64, 64)
  got = client.encode_image_to_feat_map(img)
  want = local.encode_image_to_feat_map(img)
  assert got.shape == want.shape == (2, 16, 4, 4)
  torch.testing.assert_close(got.cpu(), want.cpu())


@pytest.mark.slow
def test_encode_image_to_vector(client, local):
  torch.manual_seed(4)
  img = torch.rand(1, 3, 32, 32)
  torch.testing.assert_close(client.encode_image_to_vector(img).cpu(),
                             local.encode_image_to_vector(img).cpu())


@pytest.mark.slow
def test_encode_labels_and_prompts(client, local):
  labels = ["person", "fallen tree", "road"]
  torch.testing.assert_close(client.encode_labels(labels).cpu(),
                             local.encode_labels(labels).cpu())
  torch.testing.assert_close(client.encode_prompts(labels).cpu(),
                             local.encode_prompts(labels).cpu())
  # labels and prompts must not be the same encoding, or the test above would
  # pass for the wrong reason.
  assert not torch.allclose(local.encode_labels(labels),
                            local.encode_prompts(labels))


@pytest.mark.slow
def test_align_spatial_features_with_language(client, local):
  torch.manual_seed(5)
  feats = torch.rand(2, 16, 5, 7)
  torch.testing.assert_close(
    client.align_spatial_features_with_language(feats).cpu(),
    local.align_spatial_features_with_language(feats).cpu())


@pytest.mark.slow
def test_align_spatial_on_the_1x1_reshape_the_mapper_uses(client, local):
  """feature_query() aligns an NxC voxel feature matrix via a 1x1 spatial map."""
  torch.manual_seed(6)
  vox_feat = torch.rand(37, 16)
  got = client.align_spatial_features_with_language(
    vox_feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
  want = local.align_spatial_features_with_language(
    vox_feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
  assert got.shape == (37, 8)
  torch.testing.assert_close(got.cpu(), want.cpu())


@pytest.mark.slow
def test_align_global_features_with_language(client, local):
  torch.manual_seed(7)
  feats = torch.rand(4, 16)
  torch.testing.assert_close(
    client.align_global_features_with_language(feats).cpu(),
    local.align_global_features_with_language(feats).cpu())


@pytest.mark.slow
def test_size_helpers_are_proxied(client, local):
  assert client.is_compatible_size(64, 64) is True
  assert client.is_compatible_size(65, 64) is False
  assert client.get_nearest_size(70, 60) == local.get_nearest_size(70, 60)


@pytest.mark.slow
def test_feat_map_and_vector_tuple_roundtrips(client, local):
  torch.manual_seed(8)
  img = torch.rand(1, 3, 32, 48)
  f, v = client.encode_image_to_feat_map_and_vector(img)
  wf, wv = local.encode_image_to_feat_map_and_vector(img)
  torch.testing.assert_close(f.cpu(), wf.cpu())
  torch.testing.assert_close(v.cpu(), wv.cpu())


# --------------------------------------------------------------------------- #
# The ack: the server must hold the result until we have copied it
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_server_holds_the_result_until_the_ack(tmp_path):
  """In-process server over a Pipe, so `pending` can be observed directly."""
  from multiprocessing import Pipe

  server = EncoderServer(_dummy_cfg(tmp_path / "unused.sock"))
  a, b = Pipe()
  t = threading.Thread(target=server._serve_client, args=(b,), daemon=True)
  t.start()

  a.send(proto.make_hello("cpu", client_cuda=False))
  proto.check_hello_ack(a.recv())
  assert server.pending_count() == 0

  a.send(proto.make_call(1, "encode_image_to_feat_map",
                         wire.to_wire([torch.rand(1, 3, 32, 32)], "cpu")))
  msg = a.recv()
  value = proto.check_response(msg, 1)
  copied = wire.from_wire(value, clone=True)
  assert copied.shape == (1, 16, 2, 2)

  # The server is still holding it: this is what makes a CUDA IPC handle safe.
  assert server.pending_count() == 1

  a.send(proto.make_ack(1))
  deadline = time.time() + 10
  while server.pending_count() != 0 and time.time() < deadline:
    time.sleep(0.02)
  assert server.pending_count() == 0
  # Our copy survives the server dropping its reference.
  assert copied.shape == (1, 16, 2, 2)

  a.send(proto.make_bye())
  t.join(timeout=10)
  a.close()


@pytest.mark.slow
def test_disconnect_clears_everything_the_server_was_holding(tmp_path):
  from multiprocessing import Pipe

  server = EncoderServer(_dummy_cfg(tmp_path / "unused2.sock"))
  a, b = Pipe()
  t = threading.Thread(target=server._serve_client, args=(b,), daemon=True)
  t.start()
  a.send(proto.make_hello("cpu"))
  proto.check_hello_ack(a.recv())
  a.send(proto.make_call(1, "encode_labels", [["person"]]))  # no tensors
  proto.check_response(a.recv(), 1)
  assert server.pending_count() == 1
  a.close()                      # hang up without acking
  t.join(timeout=10)
  assert server.pending_count() == 0


# --------------------------------------------------------------------------- #
# Errors, concurrency, transports
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_a_server_side_exception_surfaces_as_an_error(client):
  # DummyEncoder.encode_labels on a non-iterable blows up inside the server.
  with pytest.raises(proto.EncoderServerError):
    client.call_remote("encode_labels", 5)
  # ...and the connection is still usable afterwards.
  assert client.encode_labels(["person"]).shape == (1, 8)


@pytest.mark.slow
def test_calling_an_unproxied_method_is_refused_client_side(client):
  with pytest.raises(proto.EncoderProtocolError):
    client.call_remote("__reduce__")


@pytest.mark.slow
def test_two_concurrent_clients(server):
  """Two clients hammering one server must each get their OWN answer.

  Distinct inputs on purpose: with the same image on both sides a response
  delivered to the wrong client would pass unnoticed.
  """
  torch.manual_seed(9)
  imgs = [torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)]
  assert not torch.allclose(imgs[0], imgs[1])

  clients = [ClientEncoder(socket=server.socket_path, timeout_s=60,
                           connect_timeout_s=60) for _ in range(2)]
  ref_enc = DummyEncoder(device=clients[0].server_device or "cpu")
  refs = [ref_enc.encode_image_to_feat_map(im).cpu() for im in imgs]
  assert not torch.allclose(refs[0], refs[1])

  errors = []
  seen = [[], []]

  def work(i):
    try:
      for _ in range(5):
        seen[i].append(clients[i].encode_image_to_feat_map(imgs[i]).cpu())
        clients[i].encode_labels([f"person_{i}", "road"])
    except Exception as e:  # noqa: BLE001
      errors.append(e)

  threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
  for t in threads:
    t.start()
  for t in threads:
    t.join(timeout=120)
  for c in clients:
    c.close()

  assert not errors, errors
  for i in (0, 1):
    assert len(seen[i]) == 5
    for got in seen[i]:
      torch.testing.assert_close(got, refs[i])
      # Explicitly: client i never received client (1-i)'s answer.
      assert not torch.allclose(got, refs[1 - i])


@pytest.mark.slow
def test_interleaved_clients_never_cross_answers(server):
  """Force the two clients to take turns, one call at a time, in lockstep.

  test_two_concurrent_clients lets the threads run free, so an interleaving
  that swaps two in-flight responses may simply not happen. Here a barrier
  makes both requests overlap on every single round.
  """
  torch.manual_seed(21)
  n_rounds = 12
  clients = [ClientEncoder(socket=server.socket_path, timeout_s=60,
                           connect_timeout_s=60) for _ in range(2)]
  ref_enc = DummyEncoder(device=clients[0].server_device or "cpu")
  # A different image for every (client, round) pair.
  imgs = [[torch.rand(1, 3, 32, 32) for _ in range(n_rounds)]
          for _ in range(2)]
  refs = [[ref_enc.encode_image_to_feat_map(im).cpu() for im in row]
          for row in imgs]

  barrier = threading.Barrier(2)
  errors = []
  got = [[None] * n_rounds for _ in range(2)]

  def work(i):
    try:
      for r in range(n_rounds):
        barrier.wait(timeout=120)
        got[i][r] = clients[i].encode_image_to_feat_map(imgs[i][r]).cpu()
    except Exception as e:  # noqa: BLE001
      errors.append(e)
      try:
        barrier.abort()
      except Exception:
        pass

  threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
  for t in threads:
    t.start()
  for t in threads:
    t.join(timeout=180)
  for c in clients:
    c.close()

  assert not errors, errors
  for i in (0, 1):
    for r in range(n_rounds):
      assert got[i][r] is not None, (i, r)
      torch.testing.assert_close(got[i][r], refs[i][r])


@pytest.mark.slow
def test_twenty_sequential_calls_keep_their_own_values(client, local):
  """20 different inputs down one connection, checked now AND at the end.

  Re-checking every retained result after all 20 calls is the buffer-reuse
  test: if the client handed back a view onto the server's allocation instead
  of a copy, call k+1 would overwrite the tensor call k returned, and the
  early results would have silently mutated by the time we look again.
  """
  torch.manual_seed(33)
  imgs = [torch.rand(1, 3, 32, 32) for _ in range(20)]
  refs = [local.encode_image_to_feat_map(im).cpu() for im in imgs]
  # Sanity: the inputs really are distinguishable.
  assert not torch.allclose(refs[0], refs[1])

  kept = []
  for i, im in enumerate(imgs):
    out = client.encode_image_to_feat_map(im)
    torch.testing.assert_close(out.cpu(), refs[i], msg=f"call {i} came back wrong")
    kept.append(out)

  for i, out in enumerate(kept):
    torch.testing.assert_close(
      out.cpu(), refs[i],
      msg=f"result of call {i} changed after {len(kept) - i - 1} later calls")


@pytest.mark.slow
def test_one_client_used_from_two_threads(client):
  """The mapping server calls the encoder from the map loop AND from a ROS
  callback thread; the socket must be serialised."""
  torch.manual_seed(10)
  img = torch.rand(1, 3, 32, 32)
  errors = []

  def loop(fn):
    try:
      for _ in range(10):
        fn()
    except Exception as e:  # noqa: BLE001
      errors.append(e)

  t1 = threading.Thread(target=loop,
                        args=(lambda: client.encode_image_to_feat_map(img),))
  t2 = threading.Thread(target=loop,
                        args=(lambda: client.encode_prompts(["person"]),))
  t1.start(); t2.start(); t1.join(timeout=120); t2.join(timeout=120)
  assert not errors, errors


@pytest.mark.slow
def test_server_death_gives_a_clear_error(tmp_path):
  s = _ServerProc(tmp_path / "die.sock")
  s.wait_ready()
  c = ClientEncoder(socket=s.socket_path, timeout_s=15, connect_timeout_s=30)
  assert c.encode_labels(["person"]).shape == (1, 8)
  s.kill()
  with pytest.raises((ConnectionError, TimeoutError, EOFError)) as e:
    for _ in range(5):
      c.encode_labels(["person"])
      time.sleep(0.2)
  assert "encoder server" in str(e.value).lower() or isinstance(
    e.value, (EOFError, TimeoutError))


@pytest.mark.slow
def test_missing_server_times_out_with_a_useful_message(tmp_path):
  with pytest.raises(TimeoutError) as e:
    ClientEncoder(socket=str(tmp_path / "nope.sock"), connect_timeout_s=1.0)
  assert "encoder_server" in str(e.value)


@pytest.mark.slow
def test_cpu_transport_is_forced_when_asked(server):
  c = ClientEncoder(socket=server.socket_path, transport="cpu", timeout_s=60,
                    connect_timeout_s=60)
  try:
    assert c.transport == "cpu"
    torch.manual_seed(11)
    img = torch.rand(1, 3, 32, 32)
    out = c.encode_image_to_feat_map(img)
    # Results land on THIS process's compute device whatever the transport.
    assert out.device.type == torch.device(c.device).type
    torch.testing.assert_close(
      out.cpu(),
      DummyEncoder(device=c.server_device or "cpu"
                   ).encode_image_to_feat_map(img).cpu())
  finally:
    c.close()


@pytest.mark.slow
@pytest.mark.skipif(HAVE_TORCH and torch.cuda.is_available(),
                    reason="only meaningful without a GPU")
def test_cuda_ipc_is_refused_on_a_cpu_only_box(server):
  with pytest.raises((proto.EncoderProtocolError, proto.EncoderServerError)):
    ClientEncoder(socket=server.socket_path, transport="cuda_ipc",
                  timeout_s=30, connect_timeout_s=30)


@pytest.mark.slow
def test_the_socket_is_removed_on_shutdown(tmp_path):
  s = _ServerProc(tmp_path / "bye.sock")
  s.wait_ready()
  assert os.path.exists(s.socket_path)
  s.stop()
  deadline = time.time() + 20
  while os.path.exists(s.socket_path) and time.time() < deadline:
    time.sleep(0.1)
  assert not os.path.exists(s.socket_path)


# --------------------------------------------------------------------------- #
# GPU only
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not (HAVE_TORCH and torch.cuda.is_available()),
                    reason="needs a GPU")
def test_cuda_ipc_keeps_the_features_on_the_gpu(tmp_path):
  s = _ServerProc(tmp_path / "cuda.sock", overrides=("encoder_server.device=cuda",))
  try:
    s.wait_ready()
    c = ClientEncoder(socket=s.socket_path, transport="cuda_ipc",
                      timeout_s=120, connect_timeout_s=120)
    try:
      assert c.transport == "cuda_ipc"
      img = torch.rand(1, 3, 64, 64, device="cuda")
      out = c.encode_image_to_feat_map(img)
      assert out.is_cuda, "CUDA IPC result came back on the CPU"
      ref = DummyEncoder(device="cuda").encode_image_to_feat_map(img)
      torch.testing.assert_close(out, ref)
      # The clone must be ours, not a view onto the server's allocation.
      assert out.data_ptr() != 0
      for _ in range(20):
        out = c.encode_image_to_feat_map(img)
      torch.testing.assert_close(out, ref)
    finally:
      c.close()
  finally:
    s.stop()
