"""Per-domain ``rclpy.Context`` bookkeeping for the multi-robot server.

Every rayfronts ROS object used to attach its ``Node`` to the process-global
default context (``rclpy.init()`` with no arguments), which pins the whole
process to one ``ROS_DOMAIN_ID`` and makes ``shutdown()`` on any one object tear
down the context for all the others.  The shared multi-robot server needs one
context per robot domain, so this module hands out **refcounted, private**
contexts keyed by domain id:

  * two components asking for the same domain (a robot's input subscriber and
    its output publisher, say) share one context and therefore one DDS
    participant, which keeps discovery traffic sane;
  * ``release_context`` only shuts a context down when the last holder is gone;
  * the process-global default context is never created, used or shut down by
    anything in here.

The legacy single-robot path does not touch this module at all — passing no
``context``/``domain_id`` keeps ``Ros2Subscriber``/``Ros2Vis``/
``Ros2MessagingService`` byte-for-byte on their old behaviour.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

try:
  import rclpy
  from rclpy.context import Context
  try:
    from rclpy.signals import SignalHandlerOptions
  except ImportError:  # pragma: no cover - very old rclpy
    SignalHandlerOptions = None
except ModuleNotFoundError:  # pragma: no cover - host / no ROS
  rclpy = None
  Context = None
  SignalHandlerOptions = None

# domain_id -> [context, refcount]
_CONTEXTS = dict()
_LOCK = threading.RLock()
# Serialises the ROS_DOMAIN_ID env fallback below.
_ENV_LOCK = threading.RLock()


def _init_context(ctx, domain_id):
  """``rclpy.init`` a fresh context, pinned to ``domain_id``.

  Uses the modern ``domain_id=`` keyword (present in Jazzy) and falls back to
  temporarily setting ``ROS_DOMAIN_ID`` for older distros, which is where a
  context reads its domain from at init time.
  """
  kwargs = dict(context=ctx)
  if SignalHandlerOptions is not None:
    # Do NOT let a secondary context install process-wide signal handlers.
    # mapping_server installs its own SIGINT handler and expects to keep it.
    kwargs["signal_handler_options"] = SignalHandlerOptions.NO
  try:
    rclpy.init(domain_id=domain_id, **kwargs)
    return
  except TypeError:
    pass

  with _ENV_LOCK:
    prev = os.environ.get("ROS_DOMAIN_ID")
    if domain_id is not None:
      os.environ["ROS_DOMAIN_ID"] = str(int(domain_id))
    try:
      rclpy.init(**kwargs)
    finally:
      if prev is None:
        os.environ.pop("ROS_DOMAIN_ID", None)
      else:
        os.environ["ROS_DOMAIN_ID"] = prev


def acquire_context(domain_id):
  """Return a refcounted private context for ``domain_id``.

  Args:
    domain_id: ROS domain id.  ``None`` means "whatever ``ROS_DOMAIN_ID`` says",
      but still a private context, not the global default one.

  Returns:
    An initialised ``rclpy.Context``.  Pair every call with
    :func:`release_context`.
  """
  if rclpy is None:
    raise RuntimeError("rclpy is not available; cannot create a ROS context.")
  key = None if domain_id is None else int(domain_id)
  with _LOCK:
    entry = _CONTEXTS.get(key)
    if entry is not None and entry[0].ok():
      entry[1] += 1
      return entry[0]
    ctx = Context()
    _init_context(ctx, key)
    _CONTEXTS[key] = [ctx, 1]
    logger.info("Created private rclpy context for ROS_DOMAIN_ID=%s", key)
    return ctx


def release_context(context):
  """Drop one reference; shut the context down when the last holder leaves."""
  if context is None:
    return
  with _LOCK:
    for key, entry in list(_CONTEXTS.items()):
      if entry[0] is context:
        entry[1] -= 1
        if entry[1] <= 0:
          _CONTEXTS.pop(key, None)
          try:
            context.try_shutdown()
          except Exception:  # pragma: no cover - shutdown races
            logger.exception("Failed to shut down context for domain %s", key)
          logger.info("Shut down private rclpy context for "
                      "ROS_DOMAIN_ID=%s", key)
        return
  # Not one of ours (caller supplied its own context): leave it alone.


def owned_domains():
  """Domains this process currently holds a private context for (for tests)."""
  with _LOCK:
    return sorted(k for k in _CONTEXTS if k is not None)


def refcount(domain_id):
  """Current refcount for ``domain_id`` (0 when not held).  For tests."""
  key = None if domain_id is None else int(domain_id)
  with _LOCK:
    entry = _CONTEXTS.get(key)
    return 0 if entry is None else entry[1]


def shutdown_all():
  """Emergency teardown of every context this module created."""
  with _LOCK:
    for key, entry in list(_CONTEXTS.items()):
      try:
        entry[0].try_shutdown()
      except Exception:  # pragma: no cover
        pass
      _CONTEXTS.pop(key, None)


def resolve_ros_object(context=None, domain_id=None):
  """Shared helper for the three legacy ROS classes.

  Returns ``(context, owns_context)``:

  * both args ``None``  -> ``(None, False)``: caller must keep its legacy
    ``if not rclpy.ok(): rclpy.init()`` + ``Node(name)`` behaviour;
  * ``context`` given   -> ``(context, False)``: use it, never shut it down;
  * ``domain_id`` given -> ``(private ctx, True)``: acquired from this module,
    released on shutdown.
  """
  if context is not None:
    return context, False
  if domain_id is None:
    return None, False
  return acquire_context(domain_id), True
