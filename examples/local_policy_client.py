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
import os
import sys
import time
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

# 获取 openpi 目录路径（用于子进程）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("LocalPolicyClient script dir:", _SCRIPT_DIR)
_OPENPI_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "openpi")
print("LocalPolicyClient openpi dir:", _OPENPI_DIR)
# 注意：真正的 openpi 包在 openpi/src/openpi/ 下，所以只需要添加 src 目录
_OPENPI_SRC_DIR = os.path.join(_OPENPI_DIR, "src")
print("LocalPolicyClient openpi src dir:", _OPENPI_SRC_DIR)
# 尝试导入 openpi（主进程）
try:
    # 只添加 src 目录（不要添加 _OPENPI_DIR，否则会找到错误的 openpi/__init__.py）
    if _OPENPI_SRC_DIR not in sys.path:
        sys.path.insert(0, _OPENPI_SRC_DIR)
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except Exception:  # pragma: no cover - training runtime must provide openpi
    _policy_config = None
    _config = None

# The pi05 AgileX follower config matches the inference script the user relies on.
_DEFAULT_CONFIG = "pi05_agileX"


# ============ 子进程 Worker：加载策略并处理 RPC 请求 ============
# 职责：
#   1. 一次性加载 pi05_agileX 策略到 CUDA（避免主进程显存占用）
#   2. 循环接收 (req_id, method, payload) 并返回 (req_id, response)
#   3. 支持 infer、get_prefix_rep、get_server_metadata 三种方法
def _policy_worker(in_q: mp.Queue, out_q: mp.Queue, config_name: str, checkpoint_dir: str, default_prompt: Optional[str], openpi_paths: list):
    # 子进程中设置 PYTHONPATH（spawn 模式不继承父进程的 sys.path）
    # 先添加 src，再添加 openpi 根目录，以便优先从 src 加载 openpi 包（包含 policies）
    for p in openpi_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    # 在子进程中导入 openpi 包，然后把 openpi/third_party 加入 openpi.__path__，
    # 这样后续的 `from openpi.third_party...` 能被解析到 openpi/third_party 下的模块。
    try:
        import importlib

        # 导入 package 对象
        openpi_pkg = importlib.import_module("openpi")
        # 计算第三方路径（openpi/third_party）——假设 openpi_paths 最后一个是 openpi 根目录
        if len(openpi_paths) >= 2:
            third_party_path = os.path.join(openpi_paths[1], "third_party")
            if os.path.isdir(third_party_path) and third_party_path not in getattr(openpi_pkg, "__path__", []):
                openpi_pkg.__path__.append(third_party_path)

        # 现在导入内部模块
        from openpi.policies import policy_config as _policy_config_local
        from openpi.training import config as _config_local
    except Exception as e:
        logging.exception("Failed to import openpi in worker: %s", e)
        out_q.put((None, {"error": f"Failed to import openpi: {e}"}))
        return
    
    # 加载策略配置和权重（仅在子进程初始化时执行一次）
    try:
        cfg = _config_local.get_config(config_name)  # 默认 pi05_agileX
        policy = _policy_config_local.create_trained_policy(cfg, checkpoint_dir, default_prompt=default_prompt)
    except Exception as e:
        logging.exception("Failed to load policy in worker: %s", e)
        out_q.put((None, {"error": str(e)}))
        return

    # RPC 事件循环：接收请求 → 调用策略 → 返回结果
    while True:
        req = in_q.get()
        if req is None:  # 收到 None 表示训练结束，退出子进程
            break
        req_id, method, payload = req
        try:
            if method == "infer":
                # 推理请求：返回动作序列 (通常为 [chunk_size, action_dim])
                # payload 可能包含:
                #   - obs: 观测数据
                #   - noise: SAC 探索噪声 (DSRL)
                #   - constraint_actions: RTC inpainting 约束 (前一个 chunk 的未执行尾部)
                #   - start_timestep: inpainting 开始的 timestep
                #   - guidance_sigma: Smooth-as-Butter 参数 (默认 0.2)
                obs = payload.get("obs", payload)
                infer_kwargs = {}
                if "noise" in payload:
                    infer_kwargs["noise"] = payload["noise"]
                if "constraint_actions" in payload:
                    infer_kwargs["constraint_actions"] = payload["constraint_actions"]
                if "start_timestep" in payload:
                    infer_kwargs["start_timestep"] = payload["start_timestep"]
                if "guidance_sigma" in payload:
                    infer_kwargs["guidance_sigma"] = payload["guidance_sigma"]
                res = policy.infer(obs, **infer_kwargs)
                out_q.put((req_id, {"result": res}))
            elif method == "get_prefix_rep":
                # 获取视觉特征：用于构造 RL agent 的 state（joint + image embedding）
                obs = payload
                if hasattr(policy, "get_prefix_rep"):
                    res = policy.get_prefix_rep(obs)  # 返回 [batch, seq, feat_dim]
                    out_q.put((req_id, {"result": res}))
                else:
                    error_msg = (
                        f"Policy {config_name} does not support 'get_prefix_rep'. "
                        "DSRL training requires the policy to expose internal visual features."
                    )
                    logging.error(error_msg)
                    out_q.put((req_id, {"error": error_msg}))
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

    def __init__(self, checkpoint_dir: str, config_name: str = _DEFAULT_CONFIG, default_prompt: Optional[str] = None, timeout: float = 60.0):
        self._ctx = mp.get_context("spawn")
        self._in_q: mp.Queue = self._ctx.Queue(maxsize=8)
        self._out_q: mp.Queue = self._ctx.Queue(maxsize=8)
        # 先传递 src 目录，再传递 openpi 根目录。
        # 这样子进程在优先从 src 加载 openpi 包（含 policies），
        # 然后我们把 openpi/third_party 加入到导入路径以支持 openpi.third_party.*
        openpi_paths = [_OPENPI_SRC_DIR, _OPENPI_DIR]
        self._proc = self._ctx.Process(target=_policy_worker, args=(self._in_q, self._out_q, config_name, checkpoint_dir, default_prompt, openpi_paths))
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

    def infer(
        self,
        obs: Dict,
        noise: Optional[np.ndarray] = None,
        constraint_actions: Optional[np.ndarray] = None,
        start_timestep: Optional[int] = None,
        guidance_sigma: float = 0.2,
    ) -> Dict:
        """Perform policy inference with optional RTC (Real-Time Action Chunking) constraints.
        
        Args:
            obs: Observation dictionary.
            noise: Optional noise array from RL agent (DSRL).
            constraint_actions: Tail of the previous action chunk for RTC inpainting.
                               Shape: (H-d, action_dim) where d = steps consumed.
            start_timestep: Optional timestep to start inpainting from.
            guidance_sigma: Smooth-as-Butter parameter for tighter guidance (default 0.2).
        
        Returns:
            Inference result dictionary containing 'actions'.
        """
        payload = {"obs": obs}
        if noise is not None:
            payload["noise"] = noise
        if constraint_actions is not None:
            payload["constraint_actions"] = constraint_actions
        if start_timestep is not None:
            payload["start_timestep"] = start_timestep
        payload["guidance_sigma"] = guidance_sigma
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
