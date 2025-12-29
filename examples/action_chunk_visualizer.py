"""
Action Chunk Visualizer: 离线可视化工具
读取记录的数据并生成可视化图表
支持 cubic_transition 和 RTC 两种模式
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import Dict, List, Optional, Any
import argparse


class ActionChunkVisualizer:
    """动作块可视化器"""
    
    def __init__(self, save_dir: str, action_dim: int = 7, is_dual_arm: bool = False):
        """
        Args:
            save_dir: 图片保存路径
            action_dim: 动作维度
            is_dual_arm: 是否双臂 (影响布局)
        """
        self.save_dir = os.path.join(save_dir, "action_visualizations")
        os.makedirs(self.save_dir, exist_ok=True)
        self.action_dim = action_dim
        self.is_dual_arm = is_dual_arm
        
        # 布局计算
        self.cols = 2 if is_dual_arm and action_dim % 2 == 0 else 1
        self.rows = action_dim // self.cols if self.cols == 2 else action_dim
        self.fig_size = (7 * self.cols, 2.5 * self.rows)
        
        print(f"✓ Visualizer initialized: {action_dim} dims, layout={self.rows}x{self.cols}")
        
    def plot_cubic_transition(
        self,
        record: Dict[str, Any],
        save_prefix: str = "cubic",
    ):
        """
        绘制 cubic_transition 模式的三种动作数据
        
        Args:
            record: 记录字典，包含 old_actions, new_actions, blended_actions
            save_prefix: 文件名前缀
        """
        global_step = record['global_step']
        time_offset = record['time_offset']
        old_actions = record['old_actions']  # (n_overlap, D)
        new_actions = record['new_actions']  # (H, D)
        blended_actions = record['blended_actions']  # (H, D)
        n_overlap = record['n_overlap']
        
        fig, axes = plt.subplots(self.rows, self.cols, figsize=self.fig_size, sharex=True)
        if self.cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 时间轴
        # old: 从 0 到 n_overlap-1
        # blended: 从 0 到 len(blended)-1
        # new: 从 time_offset 开始
        t_old = np.arange(len(old_actions))
        t_blended = np.arange(len(blended_actions))
        t_new = np.arange(len(new_actions)) + time_offset
        
        for d in range(self.action_dim):
            row, col = self._get_subplot_index(d)
            ax = axes[row, col]
            
            # 1. 绘制 old_actions (蓝色虚线)
            if len(old_actions) > 0:
                ax.plot(t_old, old_actions[:, d], 
                       label='Chunk A (old)', color='blue', 
                       linewidth=2.0, linestyle='--', alpha=0.6)
            
            # 2. 绘制 blended_actions (绿色实线) - 这是实际执行的
            ax.plot(t_blended, blended_actions[:, d], 
                   label='Blended (executed)', color='green', 
                   linewidth=2.5, alpha=0.8)
            
            # 3. 绘制 new_actions (红色虚线)
            ax.plot(t_new, new_actions[:, d], 
                   label='Chunk B (new)', color='red', 
                   linewidth=2.0, linestyle='--', alpha=0.6)
            
            # 4. 高亮融合区域
            if n_overlap > 0:
                ax.axvspan(0, n_overlap-1, alpha=0.1, color='yellow', 
                          label='Blending zone')
            
            # 5. 标记切换时刻
            ax.axvline(x=time_offset, color='gray', linestyle=':', 
                      linewidth=2, alpha=0.7, label='Switch point')
            
            # 样式
            ax.set_ylabel(self._get_dim_label(d), fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize='small', ncol=2)
            if row == self.rows - 1:
                ax.set_xlabel("Step #", fontsize=9)
        
        plt.suptitle(
            f"Cubic Transition at Step {global_step} (overlap={n_overlap}, offset={time_offset})",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        filename = f"{save_prefix}_step{global_step:06d}.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"  → Saved: {filename}")
        
    def plot_rtc_transition(
        self,
        record: Dict[str, Any],
        save_prefix: str = "rtc",
    ):
        """
        绘制 RTC 模式的两种动作数据
        
        Args:
            record: 记录字典，包含 old_chunk, new_chunk, rtc_skip
            save_prefix: 文件名前缀
        """
        global_step = record['global_step']
        time_offset = record['time_offset']
        old_chunk = record['old_chunk']  # (H, D)
        new_chunk = record['new_chunk']  # (H, D)
        rtc_skip = record['rtc_skip']
        constraint_len = record['constraint_len']
        
        fig, axes = plt.subplots(self.rows, self.cols, figsize=self.fig_size, sharex=True)
        if self.cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 时间轴
        t_old = np.arange(len(old_chunk))
        t_new = np.arange(len(new_chunk)) + time_offset
        t_new_executed = t_new[rtc_skip:]  # 实际执行的部分
        
        for d in range(self.action_dim):
            row, col = self._get_subplot_index(d)
            ax = axes[row, col]
            
            # 1. 绘制 old_chunk (蓝色)
            ax.plot(t_old, old_chunk[:, d], 
                   label='Chunk A (old)', color='blue', 
                   linewidth=2.0, alpha=0.7)
            
            # 2. 绘制 new_chunk 的约束部分 (橙色虚线)
            if rtc_skip > 0:
                ax.plot(t_new[:rtc_skip], new_chunk[:rtc_skip, d], 
                       label='Chunk B (constrained)', color='orange', 
                       linewidth=2.0, linestyle='--', alpha=0.5)
            
            # 3. 绘制 new_chunk 的执行部分 (红色实线)
            ax.plot(t_new_executed, new_chunk[rtc_skip:, d], 
                   label='Chunk B (executed)', color='red', 
                   linewidth=2.5, alpha=0.9)
            
            # 4. 高亮约束区域
            if constraint_len > 0:
                constraint_start = time_offset - constraint_len
                ax.axvspan(constraint_start, time_offset, alpha=0.1, 
                          color='cyan', label='Constraint zone')
            
            # 5. 标记 RTC skip 位置
            if rtc_skip > 0:
                skip_point = time_offset + rtc_skip
                ax.axvline(x=skip_point, color='purple', linestyle='-.', 
                          linewidth=2, alpha=0.7, label=f'Execute from here')
            
            # 6. 标记切换时刻
            ax.axvline(x=time_offset, color='gray', linestyle=':', 
                      linewidth=2, alpha=0.5, label='Inference ready')
            
            # 样式
            ax.set_ylabel(self._get_dim_label(d), fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize='x-small', ncol=2)
            if row == self.rows - 1:
                ax.set_xlabel("Step #", fontsize=9)
        
        plt.suptitle(
            f"RTC at Step {global_step} (skip={rtc_skip}, constraint={constraint_len}, latency={record['inference_latency']})",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        filename = f"{save_prefix}_step{global_step:06d}.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"  → Saved: {filename}")
        
    def plot_transition_diff_summary(
        self,
        transition_diffs: List[Dict[str, Any]],
        filename: str = "cubic_diff_summary.png"
    ):
        """
        绘制 cubic_transition 模式下的差值统计图
        
        Args:
            transition_diffs: 差值记录列表
            filename: 保存文件名
        """
        if len(transition_diffs) == 0:
            print("⚠ No transition diffs to plot")
            return
        
        # 提取数据
        steps = [d['global_step'] for d in transition_diffs]
        diffs_total = [d['diff_total'] for d in transition_diffs]
        diffs_per_dim = np.array([d['diff_per_dim'] for d in transition_diffs])  # (N, D)
        
        # 计算统计量
        mean_total = np.mean(diffs_total)
        std_total = np.std(diffs_total)
        
        mean_per_dim = np.mean(diffs_per_dim, axis=0)
        std_per_dim = np.std(diffs_per_dim, axis=0)
        
        # 创建画布：上下两个子图
        fig = plt.figure(figsize=(14, 10))
        
        # 子图1: 总体差值随时间变化
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(steps, diffs_total, 'o-', color='steelblue', 
                linewidth=2, markersize=6, alpha=0.7, label='Total Diff')
        ax1.axhline(y=mean_total, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_total:.4f}')
        ax1.fill_between(steps, mean_total - std_total, mean_total + std_total, 
                        alpha=0.2, color='red', label=f'±1 Std: {std_total:.4f}')
        ax1.set_xlabel('Global Step', fontsize=11)
        ax1.set_ylabel('L2 Norm of Difference', fontsize=11)
        ax1.set_title('Cubic Transition: Total Difference over Time', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.4)
        
        # 子图2: 各维度差值统计 (柱状图)
        ax2 = plt.subplot(2, 1, 2)
        x_pos = np.arange(self.action_dim)
        bars = ax2.bar(x_pos, mean_per_dim, yerr=std_per_dim, 
                      capsize=5, color='seagreen', alpha=0.7, 
                      error_kw={'linewidth': 2})
        ax2.set_xlabel('Action Dimension', fontsize=11)
        ax2.set_ylabel('Mean L2 Diff ± Std', fontsize=11)
        ax2.set_title('Cubic Transition: Per-Dimension Difference Statistics', 
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([self._get_dim_label(i) for i in range(self.action_dim)], 
                           rotation=45, ha='right')
        ax2.grid(True, axis='y', linestyle=':', alpha=0.4)
        
        # 在柱状图上标注数值
        for i, (m, s) in enumerate(zip(mean_per_dim, std_per_dim)):
            ax2.text(i, m + s + 0.01 * max(mean_per_dim), 
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"✓ Saved summary plot: {filename}")
        
    def _get_subplot_index(self, dim: int):
        """获取子图索引"""
        if self.is_dual_arm and self.action_dim % 2 == 0:
            half_dim = self.action_dim // 2
            if dim < half_dim:
                return dim, 0
            else:
                return dim - half_dim, 1
        else:
            return dim, 0
            
    def _get_dim_label(self, dim: int):
        """获取维度标签"""
        if self.is_dual_arm and self.action_dim % 2 == 0:
            half_dim = self.action_dim // 2
            if dim < half_dim:
                return f"L-Joint{dim}"
            else:
                return f"R-Joint{dim - half_dim}"
        else:
            return f"Dim {dim}"


def visualize_from_file(
    record_file: str,
    action_dim: int = 7,
    is_dual_arm: bool = False,
    max_plots: Optional[int] = None,
):
    """
    从记录文件生成所有可视化图表
    
    Args:
        record_file: 记录文件路径 (.pkl)
        action_dim: 动作维度
        is_dual_arm: 是否双臂
        max_plots: 最多生成多少张图 (None = 全部)
    """
    from action_chunk_recorder import load_records
    
    print(f"\n{'='*60}")
    print(f"Loading records from: {record_file}")
    print(f"{'='*60}")
    
    # 加载数据
    data = load_records(record_file)
    mode = data['mode']
    records = data['records']
    transition_diffs = data.get('transition_diffs', [])
    
    print(f"Mode: {mode.upper()}")
    print(f"Total records: {len(records)}")
    print(f"Total transition diffs: {len(transition_diffs)}")
    
    # 创建可视化器
    save_dir = os.path.dirname(record_file)
    visualizer = ActionChunkVisualizer(save_dir, action_dim, is_dual_arm)
    
    # 生成各个时刻的图表
    print(f"\nGenerating transition plots...")
    plot_count = 0
    for i, record in enumerate(records):
        if max_plots is not None and plot_count >= max_plots:
            print(f"  ... (限制为 {max_plots} 张图)")
            break
            
        if mode == 'cubic':
            visualizer.plot_cubic_transition(record, save_prefix=f"cubic_{i:03d}")
        elif mode == 'rtc':
            visualizer.plot_rtc_transition(record, save_prefix=f"rtc_{i:03d}")
        
        plot_count += 1
    
    # 生成统计摘要图
    if mode == 'cubic' and len(transition_diffs) > 0:
        print(f"\nGenerating diff summary plot...")
        visualizer.plot_transition_diff_summary(transition_diffs)
    
    print(f"\n✓ All visualizations saved to: {visualizer.save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化动作块记录数据")
    parser.add_argument("record_file", type=str, help="记录文件路径 (.pkl)")
    parser.add_argument("--action_dim", type=int, default=7, help="动作维度")
    parser.add_argument("--dual_arm", action="store_true", help="是否双臂")
    parser.add_argument("--max_plots", type=int, default=None, help="最多生成多少张图")
    
    args = parser.parse_args()
    
    visualize_from_file(
        args.record_file,
        action_dim=args.action_dim,
        is_dual_arm=args.dual_arm,
        max_plots=args.max_plots,
    )
