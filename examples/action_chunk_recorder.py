"""
Action Chunk Recorder: 轻量级数据记录器
在评估时收集动作块数据，评估后进行可视化和分析
"""
import numpy as np
import pickle
import os
from typing import Optional, List, Dict, Any


class ActionChunkRecorder:
    """
    轻量级数据记录器，用于记录动作块的衔接信息
    支持两种模式：cubic_transition 和 RTC
    """
    
    def __init__(self, save_dir: str, mode: str = "cubic"):
        """
        Args:
            save_dir: 数据保存目录
            mode: "cubic" 或 "rtc"
        """
        self.save_dir = save_dir
        self.mode = mode
        self.records = []  # 记录所有衔接事件
        self.transition_diffs = []  # 记录 cubic 模式下的差值
        
        os.makedirs(save_dir, exist_ok=True)
        
    def record_cubic_transition(
        self,
        global_step: int,
        old_actions: np.ndarray,
        new_actions: np.ndarray,
        blended_actions: np.ndarray,
        time_offset: int,
    ):
        """
        记录 cubic_transition 模式的动作衔接
        
        Args:
            global_step: 全局步数
            old_actions: chunkA 剩余动作 (n_overlap, D)
            new_actions: chunkB 新动作 (H, D)
            blended_actions: 融合后的动作 (n_overlap + (H-n_overlap), D)
            time_offset: 时间偏移量
        """
        n_overlap = len(old_actions)
        
        # 计算融合区域的差值：blended[0:n_overlap] - old_actions
        if n_overlap > 0:
            diff = blended_actions[:n_overlap] - old_actions
            # 计算每个维度的 L2 范数
            diff_norm_per_dim = np.linalg.norm(diff, axis=0)  # (D,)
            diff_norm_total = np.linalg.norm(diff)
            
            self.transition_diffs.append({
                'global_step': global_step,
                'diff_per_dim': diff_norm_per_dim,
                'diff_total': diff_norm_total,
                'n_overlap': n_overlap,
            })
        
        # 记录完整数据
        record = {
            'mode': 'cubic',
            'global_step': global_step,
            'time_offset': time_offset,
            'old_actions': old_actions.copy(),
            'new_actions': new_actions.copy(),
            'blended_actions': blended_actions.copy(),
            'n_overlap': n_overlap,
        }
        self.records.append(record)
        
    def record_rtc_transition(
        self,
        global_step: int,
        old_chunk: np.ndarray,
        new_chunk: np.ndarray,
        rtc_skip: int,
        time_offset: int,
        inference_latency: int,
        constraint_len: int,
    ):
        """
        记录 RTC 模式的动作衔接
        
        Args:
            global_step: 全局步数
            old_chunk: chunkA 完整动作 (H, D)
            new_chunk: chunkB 完整新动作 (H, D)
            rtc_skip: 跳过的步数
            time_offset: 时间偏移量
            inference_latency: 推理延迟步数
            constraint_len: 约束长度
        """
        record = {
            'mode': 'rtc',
            'global_step': global_step,
            'time_offset': time_offset,
            'old_chunk': old_chunk.copy(),
            'new_chunk': new_chunk.copy(),
            'rtc_skip': rtc_skip,
            'inference_latency': inference_latency,
            'constraint_len': constraint_len,
        }
        self.records.append(record)
        
    def save(self, filename: str = "action_records.pkl"):
        """保存记录到文件"""
        save_path = os.path.join(self.save_dir, filename)
        data = {
            'mode': self.mode,
            'records': self.records,
            'transition_diffs': self.transition_diffs,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved {len(self.records)} action records to {save_path}")
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        获取统计摘要
        """
        if self.mode == "cubic" and len(self.transition_diffs) > 0:
            # 计算所有差值的统计量
            all_diffs_per_dim = np.array([d['diff_per_dim'] for d in self.transition_diffs])
            all_diffs_total = np.array([d['diff_total'] for d in self.transition_diffs])
            
            return {
                'num_transitions': len(self.transition_diffs),
                'diff_mean_per_dim': np.mean(all_diffs_per_dim, axis=0),
                'diff_std_per_dim': np.std(all_diffs_per_dim, axis=0),
                'diff_mean_total': np.mean(all_diffs_total),
                'diff_std_total': np.std(all_diffs_total),
                'diff_max_total': np.max(all_diffs_total),
                'diff_min_total': np.min(all_diffs_total),
            }
        elif self.mode == "rtc" and len(self.records) > 0:
            rtc_skips = [r['rtc_skip'] for r in self.records]
            latencies = [r['inference_latency'] for r in self.records]
            
            return {
                'num_transitions': len(self.records),
                'rtc_skip_mean': np.mean(rtc_skips),
                'rtc_skip_std': np.std(rtc_skips),
                'latency_mean': np.mean(latencies),
                'latency_std': np.std(latencies),
            }
        return {}
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_summary_stats()
        print(f"\n{'='*60}")
        print(f"Action Chunk Recording Summary ({self.mode.upper()} mode)")
        print(f"{'='*60}")
        print(f"Total transitions recorded: {len(self.records)}")
        
        if self.mode == "cubic":
            if 'diff_mean_total' in stats:
                print(f"\nTransition Difference Statistics:")
                print(f"  Mean total diff: {stats['diff_mean_total']:.6f} ± {stats['diff_std_total']:.6f}")
                print(f"  Max total diff:  {stats['diff_max_total']:.6f}")
                print(f"  Min total diff:  {stats['diff_min_total']:.6f}")
                print(f"\n  Per-dimension mean diff:")
                for i, (m, s) in enumerate(zip(stats['diff_mean_per_dim'], stats['diff_std_per_dim'])):
                    print(f"    Dim {i}: {m:.6f} ± {s:.6f}")
        
        elif self.mode == "rtc":
            if 'rtc_skip_mean' in stats:
                print(f"\nRTC Skip Statistics:")
                print(f"  Mean skip: {stats['rtc_skip_mean']:.2f} ± {stats['rtc_skip_std']:.2f} steps")
                print(f"\nInference Latency Statistics:")
                print(f"  Mean latency: {stats['latency_mean']:.2f} ± {stats['latency_std']:.2f} steps")
        
        print(f"{'='*60}\n")


def load_records(filepath: str) -> Dict[str, Any]:
    """加载记录文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
