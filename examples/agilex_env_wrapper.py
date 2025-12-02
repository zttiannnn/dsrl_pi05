# """AgileX real-robot environment wrapper for the DSRL training pipeline.

# This adapter mirrors the hardware configuration logic from
# ``openpi/scripts/inference.py`` so that the training loop can drive the
# same AgileX follower robot (cameras, serial port, units) without relying
# on the websocket `serve_policy` stack.
# """
# from __future__ import annotations

# import logging
# import os
# from dataclasses import dataclass
# from typing import Any, Dict, Optional

# import numpy as np
# import yaml


# try:
#     from openpi.third_party.agilex.agilexfollower import AlohaAgileXFollower
#     from openpi.third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
#     from openpi.third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
#     from openpi.third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
# except ImportError:
#     # Fallback: assume 'third_party' is in PYTHONPATH (e.g. added by launch script)
#     from third_party.agilex.agilexfollower import AlohaAgileXFollower
#     from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
#     from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
#     from third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig


# LOGGER = logging.getLogger(__name__)


# @dataclass
# class AgileXCameraConfig:
#     """Stores the minimal camera parameters we accept from YAML/CLI."""

#     type: str
#     index_or_path: Any
#     width: int = 640
#     height: int = 480
#     fps: int = 30
#     use_depth: bool = False


# def _make_camera_config(cfg: AgileXCameraConfig):
#     """Re-create the helper from inference.py but with type hints."""

#     if cfg.type.lower() == "opencv":
#         return OpenCVCameraConfig(
#             index_or_path=cfg.index_or_path,
#             width=cfg.width,
#             height=cfg.height,
#             fps=cfg.fps,
#         )
#     if cfg.type.lower() == "orbbec":
#         return OrbbecCameraConfig(
#             index_or_path=cfg.index_or_path,
#             width=cfg.width,
#             height=cfg.height,
#             fps=cfg.fps,
#         )
#     raise ValueError(f"Unsupported camera type: {cfg.type!r}")


# def _load_camera_configs(
#     *, camera_config: Optional[Dict[str, Any]] = None, camera_config_path: Optional[str] = None
# ) -> Dict[str, Any]:
#     """Parse YAML/in-memory camera definitions and return config objects."""

#     raw_configs: Dict[str, Any] = {}
#     if camera_config_path:
#         path = os.path.expanduser(camera_config_path)
#         with open(path, "r", encoding="utf-8") as f:
#             data = yaml.safe_load(f) or {}
#         raw_configs.update(data)
#     if camera_config:
#         raw_configs.update(camera_config)

#     cameras = {}
#     for name, cfg in raw_configs.items():
#         if cfg is None:
#             continue
#         cameras[name] = _make_camera_config(AgileXCameraConfig(**cfg))
#     return cameras


# class AgileXFollowerEnv:
#     """AgileX 机械臂环境包装器，提供与训练循环兼容的 reset/get_observation/step 接口。
    
#     职责：
#       1. 连接松灵机械臂（通过串口）和配置的摄像头（OpenCV/Orbbec）
#       2. 封装硬件操作为标准 gym-like API
#       3. 与 openpi/scripts/inference.py 共享相同的硬件配置逻辑
    
#     返回观测格式：
#       {
#         "state": np.ndarray([7,]),           # 6 关节角度 + 1 夹爪角度
#         "images": {cam_name: np.ndarray},    # 多摄像头图像字典
#         "image_masks": {cam_name: bool}      # 图像有效性标记
#       }
#     """

#     def __init__(
#         self,
#         *,
#         port: str,                              # 串口设备路径，如 /dev/ttyACM0
#         robot_id: str = "left",                # 机械臂标识符
#         camera_config: Optional[Dict[str, Any]] = None,
#         camera_config_path: Optional[str] = None,  # YAML 摄像头配置文件路径
#         max_relative_target: Optional[int] = None,  # 动作幅度限制
#         use_degrees: bool = False,              # 是否使用角度制（默认编码器单位）
#         prompt: Optional[str] = None,           # 任务提示词（记录用）
#     ) -> None:
#         # 从 YAML 加载摄像头配置并转换为 openpi 配置对象
#         cameras = _load_camera_configs(camera_config=camera_config, camera_config_path=camera_config_path)
#         cfg = AlohaAgileXFollowerConfig(
#             port=port,
#             id=robot_id,
#             cameras=cameras,
#             max_relative_target=max_relative_target,
#             use_degrees=use_degrees,
#         )
#         self._robot = AlohaAgileXFollower(cfg)  # 实例化底层硬件接口
#         self._prompt = prompt
#         self._connect_once()  # 建立串口和摄像头连接

#     def _connect_once(self) -> None:
#         if not self._robot.is_connected:
#             LOGGER.info("Connecting AgileX follower on port %s", self._robot.config.port)
#             self._robot.connect()
#             LOGGER.info("AgileX follower connected.")

#     def reset(self) -> None:
#         # """重置机械臂状态（通过重连串口实现，因底层未暴露 home 位姿）。
        
#         # 注意：真机训练中通常需要人工复位场景，此方法仅刷新连接状态。
#         # """
#         # try:
#         #     if self._robot.is_connected:
#         #         self._robot.disconnect_port()  # 断开串口
#         # except Exception as exc:  # pragma: no cover - hardware errors
#         #     LOGGER.warning("disconnect_port failed during reset: %s", exc)
#         # self._connect_once()  # 重新建立连接
#         self._connect_once()  # 重新建立连接
#         LOGGER.info("AgileX follower reset complete.")

#     def get_observation(self) -> Dict[str, Any]:
#         return self._robot.get_observation()

#     def step(self, action: np.ndarray) -> None:
#         action = np.asarray(action).reshape(-1)
#         if action.size < 7:
#             raise ValueError(f"AgileX expects >=7 action dims, got shape {action.shape}")
#         self._robot.send_action_np(action[:7])

#     def close(self) -> None:
#         if self._robot.is_connected:
#             self._robot.disconnect()

#     def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
#         try:
#             self.close()
#         except Exception:
#             pass

"""AgileX real-robot environment wrapper for the DSRL training pipeline.

This adapter mirrors the hardware configuration logic from
``openpi/scripts/inference.py`` so that the training loop can drive the
same AgileX follower robot (cameras, serial port, units) without relying
on the websocket `serve_policy` stack.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import yaml


try:
    from openpi.third_party.agilex.agilexfollower import AlohaAgileXFollower
    from openpi.third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
    from openpi.third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from openpi.third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
except ImportError:
    # Fallback: assume 'third_party' is in PYTHONPATH (e.g. added by launch script)
    from third_party.agilex.agilexfollower import AlohaAgileXFollower
    from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
    from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class AgileXCameraConfig:
    """Stores the minimal camera parameters we accept from YAML/CLI."""

    type: str
    index_or_path: Any
    width: int = 640
    height: int = 480
    fps: int = 30
    use_depth: bool = False


def _make_camera_config(cfg: AgileXCameraConfig):
    """Re-create the helper from inference.py but with type hints."""

    if cfg.type.lower() == "opencv":
        return OpenCVCameraConfig(
            index_or_path=cfg.index_or_path,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        )
    if cfg.type.lower() == "orbbec":
        return OrbbecCameraConfig(
            index_or_path=cfg.index_or_path,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        )
    raise ValueError(f"Unsupported camera type: {cfg.type!r}")


def _load_camera_configs(
    *, camera_config: Optional[Dict[str, Any]] = None, camera_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Parse YAML/in-memory camera definitions and return config objects."""

    raw_configs: Dict[str, Any] = {}
    if camera_config_path:
        path = os.path.expanduser(camera_config_path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_configs.update(data)
    if camera_config:
        raw_configs.update(camera_config)

    cameras = {}
    for name, cfg in raw_configs.items():
        if cfg is None:
            continue
        cameras[name] = _make_camera_config(AgileXCameraConfig(**cfg))
    return cameras


class AgileXFollowerEnv:
    """AgileX 机械臂环境包装器，提供与训练循环兼容的 reset/get_observation/step 接口。
    
    职责：
      1. 连接松灵机械臂（通过串口）和配置的摄像头（OpenCV/Orbbec）
      2. 封装硬件操作为标准 gym-like API
      3. 与 openpi/scripts/inference.py 共享相同的硬件配置逻辑
    
    返回观测格式：
      {
        "state": np.ndarray([7,]),           # 6 关节角度 + 1 夹爪角度
        "images": {cam_name: np.ndarray},    # 多摄像头图像字典
        "image_masks": {cam_name: bool}      # 图像有效性标记
      }
    """

    def __init__(
        self,
        *,
        port: str,                              # 串口设备路径，如 /dev/ttyACM0
        robot_id: str = "left",                # 机械臂标识符
        camera_config: Optional[Dict[str, Any]] = None,
        camera_config_path: Optional[str] = None,  # YAML 摄像头配置文件路径
        max_relative_target: Optional[int] = None,  # 动作幅度限制
        use_degrees: bool = False,              # 是否使用角度制（默认编码器单位）
        prompt: Optional[str] = None,           # 任务提示词（记录用）
    ) -> None:
        # 从 YAML 加载摄像头配置并转换为 openpi 配置对象
        cameras = _load_camera_configs(camera_config=camera_config, camera_config_path=camera_config_path)
        cfg = AlohaAgileXFollowerConfig(
            port=port,
            id=robot_id,
            cameras=cameras,
            max_relative_target=max_relative_target,
            use_degrees=use_degrees,
        )
        self._robot = AlohaAgileXFollower(cfg)  # 实例化底层硬件接口
        self._prompt = prompt
        self._connect_once()  # 建立串口和摄像头连接

    def _connect_once(self) -> None:
        if not self._robot.is_connected:
            LOGGER.info("Connecting AgileX follower on port %s", self._robot.config.port)
            self._robot.connect()
            LOGGER.info("AgileX follower connected.")

    def reset(
        self,
        auto_home: bool = True,
        home_position: Optional[np.ndarray] = None,
        move_duration: float = 3.0,
        move_steps: int = 50,
    ) -> None:
        """重置机械臂状态，自动回零位。
        
        Args:
            auto_home: 是否自动移动到零位。默认 True。
            home_position: 目标零位位姿，shape=(7,)，分别为6个关节+夹爪。
                           默认 None 表示全零位置。
            move_duration: 回零动作的总时长（秒），默认 3.0 秒。
            move_steps: 插值步数，越大越平滑，默认 50 步。
        
        注意：
          - 回零过程采用线性插值，确保运动平滑
          - 不断开串口连接，保持通讯稳定
        """
        # 确保连接存在
        self._connect_once()
        LOGGER.info("AgileX reset called")
        
        if not auto_home:
            LOGGER.info("auto_home=False, skipping automatic home movement")
            return
        
        # 默认零位
        if home_position is None:
            home_position = np.zeros(7, dtype=np.float32)
        else:
            home_position = np.asarray(home_position, dtype=np.float32).reshape(7)
        
        # 读取当前关节状态
        try:
            obs = self._robot.get_observation()
            current_pos = obs["state"].copy()
            LOGGER.info(f"Current position: {current_pos}")
            LOGGER.info(f"Target home position: {home_position}")
        except Exception as e:
            LOGGER.warning(f"Failed to get current state, skipping auto home: {e}")
            return
        
        # 计算插值路径（线性插值）
        step_interval = move_duration / move_steps
        
        LOGGER.info(f"Moving to home position over {move_duration}s in {move_steps} steps...")
        
        for i in range(1, move_steps + 1):
            alpha = i / move_steps  # 0 -> 1
            interp_pos = (1 - alpha) * current_pos + alpha * home_position
            
            try:
                self._robot.send_action_np(interp_pos)
            except Exception as e:
                LOGGER.error(f"Failed to send action at step {i}: {e}")
                break
            
            time.sleep(step_interval)
        
        # 最后发送一次精确的目标位置
        try:
            self._robot.send_action_np(home_position)
        except Exception as e:
            LOGGER.warning(f"Failed to send final home position: {e}")
        
        LOGGER.info("AgileX reset complete, robot at home position")

    def get_observation(self) -> Dict[str, Any]:
        return self._robot.get_observation()

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action).reshape(-1)
        if action.size < 7:
            raise ValueError(f"AgileX expects >=7 action dims, got shape {action.shape}")
        self._robot.send_action_np(action[:7])

    def close(self) -> None:
        if self._robot.is_connected:
            self._robot.disconnect()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
