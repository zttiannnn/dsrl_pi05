"""Local policy client that runs a policy in a child process and exposes
an RPC-like interface (`infer`, `get_prefix_rep`) to the parent process.

This mirrors the behaviour of `openpi/scripts/inference.py` but is tailored
for the training pipeline: the parent can `infer(obs, noise=...)` and
`get_prefix_rep(obs)` synchronously.

Usage: in training code, do

    from examples.local_policy_client import LocalPolicyClient
    client = LocalPolicyClient(checkpoint_dir, config_name)
    actions = client.infer(request_data, noise=np.asarray(noise))
    prefix, meta = client.get_prefix_rep(request_data)
    client.close()
"""
from __future__ import annotations

import multiprocessing as mp
import time
import logging
from typing import Any, Dict, Optional, Tuple

import msgpack
import numpy as np

try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except Exception:
    # best-effort import; training code should run in environment where openpi is available
    _policy_config = None
    _config = None


# Worker: runs inside the child process
def _policy_worker(in_q: mp.Queue, out_q: mp.Queue, config_name: str, checkpoint_dir: str, default_prompt: Optional[str]):
    # load model once in child
    try:
        cfg = _config.get_config(config_name) if _config is not None else None
        policy = _policy_config.create_trained_policy(cfg, checkpoint_dir, default_prompt=default_prompt)
    except Exception as e:
        logging.exception("Failed to load policy in worker: %s", e)
        out_q.put((None, {"error": str(e)}))
        return

    while True:
        req = in_q.get()
        if req is None:
            break
        req_id, method, payload = req
        try:
            if method == "infer":
                # payload expected to be dict observation; some callers pass noise in payload['noise']
                obs = payload.get("obs", payload)
                # forward noise if policy.accepts it
                if "noise" in payload:
                    res = policy.infer(obs, noise=payload["noise"])
                else:
                    res = policy.infer(obs)
                out_q.put((req_id, {"result": res}))
            elif method == "get_prefix_rep":
                obs = payload
                # some policies implement get_prefix_rep
                if hasattr(policy, "get_prefix_rep"):
                    res = policy.get_prefix_rep(obs)
                    out_q.put((req_id, {"result": res}))
                else:
                    # fallback: call infer and try to extract features if present
                    res = policy.infer(obs)
                    out_q.put((req_id, {"result": res}))
            elif method == "get_server_metadata":
                meta = getattr(policy, "metadata", {})
                out_q.put((req_id, {"result": meta}))
            else:
                out_q.put((req_id, {"error": f"unknown method {method}"}))
        except Exception as e:
            logging.exception("Policy worker error for method %s: %s", method, e)
            out_q.put((req_id, {"error": str(e)}))


class LocalPolicyClient:
    """Client that talks to the child policy process via two multiprocessing queues.

    Methods:
      - infer(obs_dict, noise=None) -> dict (same shape as policy.infer)
      - get_prefix_rep(obs_dict) -> (prefix_array, meta)  (depends on policy)
      - get_server_metadata() -> dict
      - close()
    """

    def __init__(self, checkpoint_dir: str, config_name: str = "pi05_aloha", default_prompt: Optional[str] = None, timeout: float = 10.0):
        self._ctx = mp.get_context("spawn")
        self._in_q: mp.Queue = self._ctx.Queue(maxsize=8)
        self._out_q: mp.Queue = self._ctx.Queue(maxsize=8)
        self._proc = self._ctx.Process(target=_policy_worker, args=(self._in_q, self._out_q, config_name, checkpoint_dir, default_prompt))
        self._proc.daemon = True
        self._proc.start()
        self._next_id = 1
        self._timeout = timeout

    def _rpc(self, method: str, payload: Any) -> Dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._in_q.put((req_id, method, payload))
        start = time.time()
        while True:
            try:
                rid, resp = self._out_q.get(timeout=0.1)
            except Exception:
                rid = None
                resp = None
            if rid is None:
                if time.time() - start > self._timeout:
                    raise TimeoutError(f"Timeout waiting for policy worker response (method={method})")
                continue
            if rid != req_id:
                # unexpected message: push back? ignoring (should not happen with single client)
                logging.warning("Received out-of-order response %s (expected %s)", rid, req_id)
            return resp

    def infer(self, obs: Dict, noise: Optional[np.ndarray] = None) -> Dict:
        payload = {"obs": obs}
        if noise is not None:
            payload["noise"] = noise
        resp = self._rpc("infer", payload)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["result"]

    def get_prefix_rep(self, obs: Dict) -> Tuple[np.ndarray, Any]:
        resp = self._rpc("get_prefix_rep", obs)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["result"]

    def get_server_metadata(self) -> Dict:
        resp = self._rpc("get_server_metadata", {})
        if "error" in resp:
            return {}
        return resp.get("result", {})

    def close(self):
        try:
            self._in_q.put(None)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.terminate()


if __name__ == "__main__":
    # quick local test (manual)
    import os
    ckpt = os.environ.get("LOCAL_POLICY_CHECKPOINT")
    if not ckpt:
        print("Set LOCAL_POLICY_CHECKPOINT to a trained checkpoint dir to test")
    else:
        c = LocalPolicyClient(ckpt)
        print("metadata:", c.get_server_metadata())
        c.close()
