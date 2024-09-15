import os

os.environ["OMP_NUM_THREADS"] = "20"

from mmdet3d.models.backbones.res16unet import Res16UNet14E
import MinkowskiEngine as ME
import torch
from torch import nn
from mmcv import Config

class ME_Voxelization(nn.Module):
    def __init__(self, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size

    def forward(self, points):
        # 使用MinkowskiEngine的sparse_quantize进行点云的体素化
        return ME.utils.sparse_quantize(
            points / self.voxel_size, device=points.device.type
        )


def extract_bg_points_features(config, voxel_size, bg_points):
    # 实例化体素化模块
    voxelize_module = ME_Voxelization(voxel_size)
    # 对每个背景点云进行体素化，取前三个维度作为坐标
    coords = [voxelize_module(x[:, :3]).float() for x in bg_points]
    # 初始化每个点的特征为1
    feats = [torch.ones((len(x), 64), device=coords[0].device) for x in coords]
    # 创建batch index
    batch_index = [
        torch.ones((len(x), 1), device=coords[0].device) * i
        for i, x in enumerate(coords)
    ]
    # 将所有点云的坐标、特征和batch索引拼接起来
    coords = torch.cat(coords, dim=0).int()
    feats = torch.cat(feats, dim=0)
    batch_index = torch.cat(batch_index, dim=0).int()
    # 将batch索引与坐标合并
    coords = torch.cat([batch_index, coords], dim=1)
    
    # 构建稀疏张量
    tensor = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
    )
    

    # 初始化Res16UNet14E
    spars3DConv = Res16UNet14E(config).to(coords.device)
    # 对稀疏张量进行卷积运算
    output = spars3DConv(tensor)

    a=1
    return output


def main():
    voxel_size = 0.4  # 设定体素大小
    batch_size = 4  # 假设有4个batch
    num_points = 10000  # 每个点云有10000个点
    point_dim = 4  # 每个点的维度（通常是xyz + 额外信息，比如强度）

    # 随机初始化点云数据，每个点的xyz坐标范围为[0, 10]，第4维可以是强度或其他信息
    points = [
        torch.rand((num_points, point_dim), device='cuda') * 10
        for _ in range(batch_size)
    ]

    # 配置Res16UNet14E的初始化参数
    config_dict = {
        "in_channels": 64,  # 输入的通道数
        "out_channels": 64,  # 输出的通道数
        "bn_momentum": 0.1,  # BatchNorm的动量
        "conv1_kernel_size": 3,  # 第一层卷积的核大小
    }
    config = Config(config_dict)


    # 调用特征提取函数
    features = extract_bg_points_features(config, voxel_size, points)

    print("Extracted features shape:", features.F.shape)  # 打印输出特征的形状


if __name__ == "__main__":
    main()
