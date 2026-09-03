"""Framing rules of the ClientEncoder <-> encoder_server protocol.

Pure stdlib; no torch, no sockets. The live roundtrip (including CUDA IPC and
the keep-alive-until-ack rule) is test_client_encoder.py.
"""

import pytest


def test_version_and_kinds_are_stable(proto):
  assert proto.PROTOCOL_VERSION == 1
  assert proto.KIND_HELLO == "hello"
  assert proto.KIND_CALL == "call"
  assert proto.KIND_RESULT == "result"
  assert proto.KIND_ACK == "ack"
  assert proto.KIND_ERROR == "error"


def test_hello_carries_the_version_and_the_client_capability(proto):
  h = proto.make_hello("auto", client_cuda=True)
  assert h["kind"] == "hello"
  assert h["version"] == proto.PROTOCOL_VERSION
  assert h["transport"] == "auto"
  assert h["client_cuda"] is True


def test_hello_rejects_an_unknown_transport(proto):
  with pytest.raises(ValueError):
    proto.make_hello("rdma")


def test_hello_ack_version_mismatch_is_loud(proto):
  ack = proto.make_hello_ack("DummyEncoder", "cpu", "cpu")
  ack["version"] = 999
  with pytest.raises(proto.EncoderProtocolError) as e:
    proto.check_hello_ack(ack)
  assert "protocol" in str(e.value).lower()


def test_hello_ack_error_becomes_a_server_error(proto):
  with pytest.raises(proto.EncoderServerError):
    proto.check_hello_ack(proto.make_error(None, "no cuda here"))


def test_only_whitelisted_methods_can_be_called(proto):
  proto.make_call(1, "encode_image_to_feat_map")
  with pytest.raises(proto.EncoderProtocolError):
    proto.make_call(1, "__reduce__")
  with pytest.raises(proto.EncoderProtocolError):
    proto.make_call(1, "os.system")


def test_the_methods_the_mappers_actually_use_are_proxied(proto):
  # From the internals report: what SemanticRayFrontiersMap and MappingServer
  # call on the encoder.
  for m in ("encode_image_to_feat_map", "encode_image_to_vector",
            "encode_labels", "encode_prompts",
            "align_spatial_features_with_language",
            "align_global_features_with_language",
            "is_compatible_size", "get_nearest_size"):
    assert m in proto.CALLABLE_METHODS


def test_only_whitelisted_attributes_can_be_read(proto):
  proto.make_getattr(1, "num_classes")
  with pytest.raises(proto.EncoderProtocolError):
    proto.make_getattr(1, "__dict__")


def test_check_response_returns_the_value(proto):
  assert proto.check_response(proto.make_result(3, [1, 2]), 3) == [1, 2]


def test_check_response_rejects_a_mismatched_id(proto):
  with pytest.raises(proto.EncoderProtocolError):
    proto.check_response(proto.make_result(4, None), 3)


def test_check_response_raises_the_servers_exception(proto):
  with pytest.raises(proto.EncoderServerError) as e:
    proto.check_response(proto.make_error(3, "CUDA OOM", "tb..."), 3)
  assert "CUDA OOM" in str(e.value)


def test_check_response_rejects_garbage(proto):
  for junk in (None, 42, "hello", {}, {"kind": "ack", "id": 3}):
    with pytest.raises(proto.EncoderProtocolError):
      proto.check_response(junk, 3)


@pytest.mark.parametrize("requested,client,server,expected", [
  ("auto", True, True, "cuda_ipc"),
  ("auto", True, False, "cpu"),
  ("auto", False, True, "cpu"),
  ("auto", False, False, "cpu"),
  ("cpu", True, True, "cpu"),
  ("cuda_ipc", True, True, "cuda_ipc"),
])
def test_transport_negotiation(proto, requested, client, server, expected):
  assert proto.negotiate_transport(requested, client, server) == expected


@pytest.mark.parametrize("client,server", [(False, True), (True, False),
                                           (False, False)])
def test_cuda_ipc_is_refused_rather_than_silently_downgraded(proto, client,
                                                             server):
  with pytest.raises(proto.EncoderProtocolError):
    proto.negotiate_transport("cuda_ipc", client, server)


def test_ack_and_bye(proto):
  assert proto.make_ack(5) == {"kind": "ack", "id": 5}
  assert proto.make_bye() == {"kind": "bye"}


def test_a_stale_response_is_a_protocol_error_not_a_silent_answer(proto):
  """The failure mode that matters: never return someone else's tensor.

  If a call timed out and its answer arrives late, the next call's recv() sees
  a response whose id belongs to the previous request. That must raise -- a
  feature map from the wrong frame would be accepted silently by the mapper.
  """
  stale = proto.make_result(7, "features-for-call-7")
  with pytest.raises(proto.EncoderProtocolError) as e:
    proto.check_response(stale, 8)
  assert "id mismatch" in str(e.value).lower()
