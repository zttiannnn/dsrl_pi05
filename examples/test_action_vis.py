"""
Quick test script for action chunk recording and visualization
测试记录和可视化系统
"""
import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from action_chunk_recorder import ActionChunkRecorder
from action_chunk_visualizer import visualize_from_file


def generate_mock_data():
    """生成模拟数据用于测试"""
    print("Generating mock action chunk data...")
    
    # 模拟参数
    H = 50  # horizon
    D = 7   # action dimension
    n_transitions = 10
    
    # 测试 Cubic 模式
    print("\n--- Testing Cubic Transition Mode ---")
    recorder_cubic = ActionChunkRecorder("test_output", mode="cubic")
    
    for i in range(n_transitions):
        # 模拟旧动作块剩余部分
        n_overlap = np.random.randint(10, 20)
        old_actions = np.random.randn(n_overlap, D) * 0.1
        
        # 模拟新动作块
        new_actions = np.random.randn(H, D) * 0.1
        
        # 模拟融合（这里简单地线性插值）
        blended = []
        for j in range(n_overlap):
            alpha = j / n_overlap
            blend = (1 - alpha) * old_actions[j] + alpha * new_actions[j]
            blended.append(blend)
        blended.extend(new_actions[n_overlap:])
        blended = np.array(blended)
        
        recorder_cubic.record_cubic_transition(
            global_step=i * 50,
            old_actions=old_actions,
            new_actions=new_actions,
            blended_actions=blended,
            time_offset=i * 50,
        )
    
    recorder_cubic.save("cubic_test_records.pkl")
    recorder_cubic.print_summary()
    
    # 测试 RTC 模式
    print("\n--- Testing RTC Mode ---")
    recorder_rtc = ActionChunkRecorder("test_output", mode="rtc")
    
    for i in range(n_transitions):
        # 模拟完整的旧动作块
        old_chunk = np.random.randn(H, D) * 0.1
        
        # 模拟新动作块
        new_chunk = np.random.randn(H, D) * 0.1
        
        # 模拟 RTC 参数
        inference_latency = np.random.randint(15, 25)
        constraint_len = np.random.randint(30, 40)
        rtc_skip = min(inference_latency, constraint_len)
        
        recorder_rtc.record_rtc_transition(
            global_step=i * 50,
            old_chunk=old_chunk,
            new_chunk=new_chunk,
            rtc_skip=rtc_skip,
            time_offset=i * 50,
            inference_latency=inference_latency,
            constraint_len=constraint_len,
        )
    
    recorder_rtc.save("rtc_test_records.pkl")
    recorder_rtc.print_summary()
    
    return "test_output/cubic_test_records.pkl", "test_output/rtc_test_records.pkl"


def test_visualization(cubic_file, rtc_file):
    """测试可视化功能"""
    print("\n--- Testing Visualization ---")
    
    # 可视化 Cubic 模式
    print("\n1. Visualizing Cubic Transition mode...")
    visualize_from_file(cubic_file, action_dim=7, max_plots=3)
    
    # 可视化 RTC 模式
    print("\n2. Visualizing RTC mode...")
    visualize_from_file(rtc_file, action_dim=7, max_plots=3)
    
    print("\n✓ Test completed successfully!")
    print(f"  Check 'test_output/action_visualizations/' for generated plots")


if __name__ == "__main__":
    print("="*60)
    print("Action Chunk Recording & Visualization System Test")
    print("="*60)
    
    cubic_file, rtc_file = generate_mock_data()
    test_visualization(cubic_file, rtc_file)
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
