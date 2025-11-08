import torch
import torch.nn as nn

from net import UNet

def test_unet_model():

    BATCH_SIZE = 4
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IN_CHANNELS = 3
    OUT_CHANNELS = 1

    print("--- 开始 U-Net 模型测试 ---")
    print(f"测试参数: Batch Size={BATCH_SIZE}, Input Shape=({IN_CHANNELS}, {IMG_HEIGHT}, {IMG_WIDTH})")

    print("\n正在初始化 U-Net 模型...")
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
    model.eval()
    print("模型初始化成功。")

    print(f"\n正在创建形状为 ({BATCH_SIZE}, {IN_CHANNELS}, {IMG_HEIGHT}, {IMG_WIDTH}) 的随机输入张量...")
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    print("正在执行前向传播...")
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print("前向传播成功。")

    print("\n--- 测试结果 ---")
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")

    expected_shape = (BATCH_SIZE, OUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    assert output_tensor.shape == expected_shape, \
        f"输出形状不匹配! 期望: {expected_shape}, 实际: {output_tensor.shape}"
    assert torch.all(output_tensor >= 0) and torch.all(output_tensor <= 1), \
        "输出值不在 [0, 1] 范围内，Sigmoid 激活函数可能有问题。"

    print("\n✅ 所有断言通过！")
    print("U-Net 模型测试成功。输出形状和值范围均符合预期。")
    print("--- 测试结束 ---")


if __name__ == "__main__":
    test_unet_model()

