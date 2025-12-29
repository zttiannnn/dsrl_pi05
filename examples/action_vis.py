import matplotlib.pyplot as plt
import numpy as np
import os
import time

class ActionChunkVisualizer:
    def __init__(self, save_dir, action_dim=14, is_dual_arm=False):
        """
        初始化可视化器
        :param save_dir: 图片保存路径
        :param action_dim: 动作维度 (例如 14 包含双臂+夹爪)
        :param is_dual_arm: 是否尝试按左右臂分栏显示 (默认 False)
        """
        self.save_dir = os.path.join(save_dir, "action_plots")
        os.makedirs(self.save_dir, exist_ok=True)
        self.action_dim = action_dim
        self.is_dual_arm = is_dual_arm
        
        # 预先创建 Figure，避免重复创建销毁带来的开销
        # 如果是双臂(14维)，布局为 2列 x 7行 (或者只画前6个关节)
        # 这里为了通用性，我们动态计算行列
        self.cols = 2 if is_dual_arm and action_dim % 2 == 0 else 1
        self.rows = action_dim // self.cols
        
        # 设置画布大小
        self.fig_size = (6 * self.cols, 2 * self.rows) 
        
    def plot_and_save(self, global_step, prior_chunk, new_chunk, time_offset, filename_prefix="step"):
        """
        绘制动作块对比图并保存
        :param global_step: 全局步数，用于标题
        :param prior_chunk: 上一次生成的动作块 (H, D) - 对应图中蓝色 Prior
        :param new_chunk: 新生成的动作块 (H, D) - 对应图中红色 New
        :param time_offset: 新动作块相对于旧动作块的时间偏移量 (即推理延迟/间隔步数)
        """
        if prior_chunk is None or len(prior_chunk) == 0:
            return

        # 确保数据是 numpy 数组
        prior = np.asarray(prior_chunk)
        new = np.asarray(new_chunk)
        
        H_prior = prior.shape[0]
        H_new = new.shape[0]
        
        # 创建画布
        fig, axes = plt.subplots(self.rows, self.cols, figsize=self.fig_size, sharex=True)
        if self.cols == 1:
            axes = axes[:, np.newaxis] # 统一维度方便索引
        
        # 生成时间轴 x 坐标
        # Prior 从 0 开始: [0, 1, ..., H_prior-1]
        t_prior = np.arange(H_prior)
        # New 从 time_offset 开始: [offset, offset+1, ..., offset+H_new-1]
        t_new = np.arange(H_new) + time_offset
        
        # 遍历每个维度绘图
        for d in range(self.action_dim):
            # 计算行列索引
            if self.is_dual_arm:
                # 假设前一半是左臂，后一半是右臂
                half_dim = self.action_dim // 2
                if d < half_dim:
                    row, col = d, 0
                    title_prefix = "Left Joint"
                else:
                    row, col = d - half_dim, 1
                    title_prefix = "Right Joint"
            else:
                row, col = d, 0
                title_prefix = "Dim"
            
            ax = axes[row, col]
            
            # 绘制 Prior (Blue)
            ax.plot(t_prior, prior[:, d], label='Prior', color='blue', linewidth=1.5, alpha=0.7)
            
            # 绘制 New (Red)
            ax.plot(t_new, new[:, d], label='New', color='red', linewidth=1.5, alpha=0.9)
            
            # 绘制当前时刻分割线 (Vertical Dashed Line)
            # 在 time_offset 处画线，表示这是新推理刚刚生效的时刻
            ax.axvline(x=time_offset, color='gray', linestyle='--', alpha=0.5)
            
            # 仅在第一行显示 Legend，避免遮挡
            if row == 0:
                ax.legend(loc='upper right', fontsize='small')
            
            # 设置标题和标签
            ax.set_ylabel(f"{title_prefix} {d}", fontsize=8)
            if row == self.rows - 1:
                ax.set_xlabel("Step #")
                
            # 优化网格
            ax.grid(True, linestyle=':', alpha=0.3)

        plt.suptitle(f"Action Chunk Stitching at Step {global_step} (Offset={time_offset})", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图片
        filename = f"{filename_prefix}_{global_step:06d}.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=80)
        plt.close(fig) # 必须关闭，否则内存泄漏
