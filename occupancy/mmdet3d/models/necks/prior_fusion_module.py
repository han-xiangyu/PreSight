import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, xavier_init, constant_init
import numpy as np
from mmcv.runner import BaseModule, force_fp32
from IPython import embed
import ipdb
import MinkowskiEngine as ME
from mmdet3d.models.backbones.res16unet import Res16UNet14E
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

@NECKS.register_module()
class PriorFusion2D(nn.Module):
    def __init__(self,
                 prior_pc_range,
                 prior_voxel_size,
                 bev_feats_channels=256, 
                 voxel_channels=68, 
                 num_pool_buckets=4,
                 hidden_channels=256,
                 dropout=0.0,
                ):
        super().__init__()
        self.prior_pc_range = torch.tensor(prior_pc_range)
        self.prior_voxel_size = torch.tensor(prior_voxel_size)

        self.bev_feats_channels = bev_feats_channels
        self.voxel_channels = voxel_channels
        self.num_prior_z = int((prior_pc_range[5] - prior_pc_range[2]) / prior_voxel_size[2])
        self.num_pool_buckets = num_pool_buckets
        self.hidden_channels = hidden_channels
        self.num_z_pooled = int(self.num_prior_z / num_pool_buckets)
        
        self.voxel_feature_extractor = nn.Sequential(
            nn.Linear(voxel_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.z_max_pooling = nn.MaxPool1d(kernel_size=self.num_z_pooled, stride=self.num_z_pooled)
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_channels * self.num_z_pooled, hidden_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels + bev_feats_channels, bev_feats_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(bev_feats_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_feats_channels, bev_feats_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_feats_channels),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        
            elif isinstance(m, nn.Linear):
                xavier_init(m)
            
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
    
    @force_fp32()
    def forward(self, bev_feats: torch.Tensor, prior_feats: torch.Tensor, prior_voxels_coords):
        """
            bev_feats: shape (bs, bev_c, h, w)
            prior_feats: list[tensor(num_voxels, c)]
            prior_voxels_coords: list[tensor(num_voxels, 3)]
        """
        bs = len(prior_feats)
        # num_pts_per_sample = [len(i) for i in prior_feats]
        # flattened_prior_feats = torch.cat(prior_feats, dim=0) # (total_num_voxels, c)
        # # first extract per-voxel feature
        # prior_feats = self.voxel_feature_extractor(
        #     flattened_prior_feats
        # ) # (total_num_voxels, hidden)
        # split_idx = [0] + np.cumsum(num_pts_per_sample).tolist()
        # prior_feats_list = [prior_feats[split_idx[i]: split_idx[i+1]] for i in range(bs)] # list[tensor(num_voxels, hidden)]
        prior_feats_list = [self.voxel_feature_extractor(p) for p in prior_feats] # list[tensor(num_voxels, hidden)]
        prior_voxels = self.formulate_voxels(prior_feats_list, prior_voxels_coords) # (bs, w, h, z, hidden)
        prior_voxels = prior_voxels.permute(0, 4, 2, 1, 3) # (bs, hidden, h, w, z)
        
        bs, hidden, h, w, z = prior_voxels.shape

        # max pooling by z-axis
        prior_feats_pooled = self.z_max_pooling(
            prior_voxels.flatten(0, -2) # (-1, num_z)
        ).view(bs, self.hidden_channels, h, w, self.num_z_pooled) # (bs, hidden, h, w, num_z_pooled)

        prior_feats_pooled = prior_feats_pooled.permute(0, 1, 4, 2, 3).flatten(1, 2) # (bs, hidden*num_z_pooled, h, w)
        
        # compress z-axis
        prior_bev_feats = self.block1(prior_feats_pooled) # (bs, hidden, h, w)

        # aggregate bev features
        if prior_bev_feats.shape[-2:] != bev_feats.shape[-2:]:
            bev_h, bev_w = bev_feats.shape[-2:]
            prior_bev_feats = F.interpolate(prior_bev_feats, size=(bev_h, bev_w),
                                            mode="bilinear", align_corners=False)
        
        bev_feats = torch.cat([bev_feats, prior_bev_feats], dim=1) # (bs, hidden+bev_c, h, w)
        bev_feats = self.block2(bev_feats) # (bs, bev_c, h, w)

        return bev_feats
    
    def formulate_voxels(self, prior_voxels, prior_voxels_coords):
        bs = len(prior_voxels)
        voxel_resolution = torch.ceil(
            (self.prior_pc_range[3:] - self.prior_pc_range[:3]) / self.prior_voxel_size
        ).long()
        voxels = []
        for i in range(bs):
            points = prior_voxels[i]
            coords = prior_voxels_coords[i].long()
            dim_feats = points.size(1)
            voxel = torch.zeros((*voxel_resolution, dim_feats), 
                                dtype=torch.float32, device=points.device)
            voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = points.float()
            voxels.append(voxel)
        
        voxels = torch.stack(voxels)
        return voxels

@NECKS.register_module()
class PriorFusion3D_voxel(nn.Module):
    def __init__(self,
                 prior_pc_range,
                 prior_voxel_size=[0.4, 0.4, 0.4],
                 bev_hidden_channels=256, 
                 prior_in_channels=68, 
                 prior_voxel_hidden_channels=64,
                 out_num_z=8,
                 out_channels=80,
                 dropout=0.0,
                 residual=True,
                ):
        super().__init__()
        self.prior_pc_range = torch.tensor(prior_pc_range)
        self.prior_voxel_size = torch.tensor(prior_voxel_size)

        self.bev_hidden_channels = bev_hidden_channels
        self.prior_in_channels = prior_in_channels
        self.out_num_z = out_num_z
        self.num_prior_z = int((prior_pc_range[5] - prior_pc_range[2]) / prior_voxel_size[2])
        self.prior_voxel_hidden_channels = prior_voxel_hidden_channels
        self.out_channels = out_channels
        
        self.voxel_feature_extractor = nn.Sequential(
            nn.Linear(prior_in_channels, prior_voxel_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prior_voxel_hidden_channels, prior_voxel_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.block1 = nn.Sequential(
            nn.Conv2d(prior_voxel_hidden_channels * self.num_prior_z, bev_hidden_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(bev_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_hidden_channels, bev_hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.block2 = nn.Sequential(
            # nn.Conv3d(int(bev_hidden_channels / out_num_z) + out_channels, out_channels, kernel_size=1, padding=0),
            nn.Conv3d(2 * out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_channels),
        )
        self.residual = residual

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
        
            elif isinstance(m, nn.Linear):
                xavier_init(m)
            
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)
    

    def split_coords_and_feats(self, coords, feats, batch_size):
        """
        将包含batch index的coords和卷积后的特征feats拆分回n个batch的list结构。
        
        Args:
            coords: Tensor (num_points, 4) - 包含[batch_index, x, y, z]的4维坐标
            feats: Tensor (num_points, feat_dim) - 对应点的特征
            batch_size: int - batch 的大小(n)
            
        Returns:
            split_coords_list: list of Tensors - 每个元素包含该batch的所有点的坐标
            split_feats_list: list of Tensors - 每个元素包含该batch的所有点的特征
        """
        split_coords_list = []
        split_feats_list = []
        
        # 遍历每个batch，分别提取属于该batch的坐标和特征
        for i in range(batch_size):
            # 获取属于当前batch的点索引
            batch_mask = coords[:, 0] == i
            
            # 提取该batch的所有点的坐标（去掉第0列，即batch index）
            batch_coords = coords[batch_mask, 1:]
            
            # 提取该batch的所有点的特征
            batch_feats = feats[batch_mask]
            
            # 将结果添加到列表中
            split_coords_list.append(batch_coords)
            split_feats_list.append(batch_feats)
        
        return split_coords_list, split_feats_list

    def extract_bg_points_features(self, prior_feats, prior_voxels_coords):
        # 配置Res16UNet14E的初始化参数
        config_dict = {
            "in_channels": 68,  # 输入的通道数
            "out_channels": 64,  # 输出的通道数
            "bn_momentum": 0.1,  # BatchNorm的动量
            "conv1_kernel_size": 3,  # 第一层卷积的核大小
        }
        config = Config(config_dict)


        # voxelize_module = ME_Voxelization(self.prior_voxel_size)
        # 创建batch index
        batch_index = [
            torch.ones((len(x), 1), device=prior_voxels_coords[0].device) * i
            for i, x in enumerate(prior_voxels_coords)
        ]
        # 将所有点云的坐标、特征和batch索引拼接起来
        coords = torch.cat(prior_voxels_coords, dim=0).int()
        feats = torch.cat(prior_feats, dim=0)
        batch_index = torch.cat(batch_index, dim=0).int()
        # 将batch索引与坐标合并
        coords = torch.cat([batch_index, coords], dim=1)
        tensor = ME.SparseTensor(
            features=feats.float(),
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        )
        # ipdb.set_trace()
        # 初始化Res16UNet14E
        spars3DConv = Res16UNet14E(config).to(coords.device)
        # 对稀疏张量进行卷积运算
        output = spars3DConv(tensor)

        return output.coordinates, output.features
    
    @force_fp32()
    def forward(self, bev_feats: torch.Tensor, prior_feats: torch.Tensor, prior_voxels_coords):
        """
            bev_feats: shape (bs, bev_c, bev_h, bev_w, bev_z)
            prior_feats: list[tensor(num_voxels, c)]
            prior_voxels_coords: list[tensor(num_voxels, 3)]
        """

        bs, bev_c, bev_h, bev_w, bev_z = bev_feats.shape
        extracted_coords, extracted_feats = self.extract_bg_points_features(prior_feats, prior_voxels_coords)
        split_coords, split_feats = self.split_coords_and_feats(extracted_coords, extracted_feats, bs)
        prior_bev_feats = self.formulate_voxels(split_coords, split_feats) # (bs, w, h, z, hidden)
        prior_bev_feats = prior_bev_feats.permute(0, 4, 2, 1, 3) # (bs, prior_feat_c, bev_h, bev_w, bev_z)
        prior_bev_feats = F.interpolate(prior_bev_feats, size=(200, 200, 16), mode='trilinear', align_corners=False)

        x = torch.cat([bev_feats, prior_bev_feats], dim=1) # (bs, bev_c + prior_feat_c, bev_h, bev_w, bev_z)
        if self.residual:
            fused_bev_feats = F.relu(self.block2(x) + bev_feats) # (bs, bev_c, bev_h, bev_w, bev_z)
        else:
            fused_bev_feats = F.relu(self.block2(x)) # (bs, bev_c, bev_h, bev_w, bev_z)
        return fused_bev_feats


    def formulate_voxels(self, prior_voxels_coords, prior_voxels):
        bs = len(prior_voxels)
        voxel_resolution = torch.ceil(
            (self.prior_pc_range[3:] - self.prior_pc_range[:3]) / self.prior_voxel_size
        ).long()
        voxels = []
        for i in range(bs):
            points = prior_voxels[i]
            coords = prior_voxels_coords[i].long()
            dim_feats = points.size(1)
            voxel = torch.zeros((*voxel_resolution, dim_feats), 
                                dtype=torch.float32, device=points.device)
            voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = points.float()
            voxels.append(voxel)
        
        voxels = torch.stack(voxels)
        return voxels




    # @force_fp32()
    # def forward(self, bev_feats: torch.Tensor, prior_feats: torch.Tensor, prior_voxels_coords):
    #     """
    #         bev_feats: shape (bs, bev_c, bev_h, bev_w, bev_z)
    #         prior_feats: list[tensor(num_voxels, c)]
    #         prior_voxels_coords: list[tensor(num_voxels, 3)]
    #     """

    #     bs, bev_c, bev_h, bev_w, bev_z = bev_feats.shape
    #     prior_feats_list = [self.voxel_feature_extractor(p) for p in prior_feats] # list[tensor(num_voxels, hidden)]
    #     prior_voxels_feats = self.formulate_voxels(prior_feats_list, prior_voxels_coords) # (bs, w, h, z, hidden)
    #     prior_voxels_feats = prior_voxels_feats.permute(0, 4, 3, 2, 1) # (bs, hidden1, z, h, w)

    #     prior_bev_feats = prior_voxels_feats.flatten(1, 2) # (bs, hidden1*z, h, w)
    #     bs, _, h, w = prior_bev_feats.shape

    #     prior_bev_feats = self.block1(prior_bev_feats) # (bs, hidden2, h, w)
    #     prior_bev_feats = F.max_pool2d(prior_bev_feats, kernel_size=2) # (bs, hidden2, h/2, w/2)

    #     # aggregate bev voxel features
    #     if prior_bev_feats.shape[-2:] != bev_feats.shape[-2:]:
    #         prior_bev_feats = F.interpolate(prior_bev_feats, size=(bev_h, bev_w),
    #                                         mode="bilinear", align_corners=True) # (bs, hidden2, bev_h, bev_w)
        
    #     assert self.out_num_z == bev_z
    #     prior_bev_feats = prior_bev_feats.view(bs, -1, self.out_num_z, bev_h, bev_w).permute(0, 1, 3, 4, 2) 
    #     # (bs, hidden2/z, bev_h, bev_w, bev_z)
        
    #     x = torch.cat([bev_feats, prior_bev_feats], dim=1) # (bs, hidden2/z+bev_c, bev_h, bev_w, bev_z)
    #     if self.residual:
    #         bev_feats = F.relu(self.block2(x) + bev_feats) # (bs, bev_c, bev_h, bev_w, bev_z)
    #     else:
    #         bev_feats = F.relu(self.block2(x)) # (bs, bev_c, bev_h, bev_w, bev_z)
    #     # ipdb.set_trace()
    #     return bev_feats
    
    # def formulate_voxels(self, prior_voxels, prior_voxels_coords):
    #     bs = len(prior_voxels)
    #     voxel_resolution = torch.ceil(
    #         (self.prior_pc_range[3:] - self.prior_pc_range[:3]) / self.prior_voxel_size
    #     ).long()
    #     voxels = []
    #     for i in range(bs):
    #         points = prior_voxels[i]
    #         coords = prior_voxels_coords[i].long()
    #         dim_feats = points.size(1)
    #         voxel = torch.zeros((*voxel_resolution, dim_feats), 
    #                             dtype=torch.float32, device=points.device)
    #         voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = points.float()
    #         voxels.append(voxel)
        
    #     voxels = torch.stack(voxels)
    #     return voxels


@NECKS.register_module()
class PriorFusion3D_crossattn(nn.Module):
    def __init__(self,
                 prior_pc_range,
                 prior_voxel_size,
                 bev_hidden_channels=256, 
                 prior_in_channels=68, 
                 prior_voxel_hidden_channels=64,
                 out_num_z=8,
                 out_channels=80,
                 dropout=0.0,
                 residual=True,
                 num_bev_win=10,
                 bev_h=200,
                 bev_w=200
                ):
        super().__init__()
        self.prior_pc_range = torch.tensor(prior_pc_range)
        self.prior_voxel_size = torch.tensor(prior_voxel_size)

        self.bev_hidden_channels = bev_hidden_channels
        self.prior_in_channels = prior_in_channels
        self.out_num_z = out_num_z
        self.num_prior_z = int((prior_pc_range[5] - prior_pc_range[2]) / prior_voxel_size[2])
        self.prior_voxel_hidden_channels = prior_voxel_hidden_channels
        self.out_channels = out_channels
        
        self.voxel_feature_extractor = nn.Sequential(
            nn.Linear(prior_in_channels, prior_voxel_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prior_voxel_hidden_channels, prior_voxel_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.block1 = nn.Sequential(
            nn.Conv2d(prior_voxel_hidden_channels * self.num_prior_z, bev_hidden_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(bev_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_hidden_channels, bev_hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bev_hidden_channels),
            nn.ReLU(inplace=True),
        )
        from .window_cross_attention import WindowCrossAttention
        self.cross_attn = WindowCrossAttention(
            num_bev_win=num_bev_win,
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=bev_hidden_channels,
            dropout=dropout,
        )

        self.bev_in_proj = nn.Linear(out_num_z*out_channels, bev_hidden_channels)
        self.bev_out_proj = nn.Linear(bev_hidden_channels, out_num_z*out_channels)
        
        self.residual = residual
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
        
            elif isinstance(m, nn.Linear):
                xavier_init(m)
            
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)
    
    @force_fp32()
    def forward(self, bev_feats: torch.Tensor, prior_feats: torch.Tensor, prior_voxels_coords):
        """
            bev_feats: shape (bs, bev_c, bev_h, bev_w, bev_z)
            prior_feats: list[tensor(num_voxels, c)]
            prior_voxels_coords: list[tensor(num_voxels, 3)]
        """

        bs, bev_c, bev_h, bev_w, bev_z = bev_feats.shape
        prior_feats_list = [self.voxel_feature_extractor(p) for p in prior_feats] # list[tensor(num_voxels, hidden)]
        prior_voxels_feats = self.formulate_voxels(prior_feats_list, prior_voxels_coords) # (bs, w, h, z, hidden)
        prior_voxels_feats = prior_voxels_feats.permute(0, 4, 3, 2, 1) # (bs, hidden1, z, h, w)

        prior_bev_feats = prior_voxels_feats.flatten(1, 2) # (bs, hidden1*z, h, w)
        bs, _, h, w = prior_bev_feats.shape

        prior_bev_feats = self.block1(prior_bev_feats) # (bs, hidden2, h, w)
        prior_bev_feats = F.max_pool2d(prior_bev_feats, kernel_size=2) # (bs, hidden2, h/2, w/2)

        # aggregate bev voxel features
        if prior_bev_feats.shape[-2:] != bev_feats.shape[-2:]:
            prior_bev_feats = F.interpolate(prior_bev_feats, size=(bev_h, bev_w),
                                            mode="bilinear", align_corners=True) # (bs, hidden2, bev_h, bev_w)
        
        assert self.out_num_z == bev_z
        
        prior_bev_feats = prior_bev_feats.permute(0, 2, 3, 1) # (bs, bev_h, bev_w, hidden2)
        bev_feats = bev_feats.permute(0, 2, 3, 4, 1).flatten(3, 4) # (bs, bev_h, bev_w, bev_z*bev_c)
        bev_feats = self.bev_in_proj(bev_feats) # (bs, bev_h, bev_w, hidden2)
        bev_feats = self.cross_attn(
            query=bev_feats.flatten(1, 2), 
            key=prior_bev_feats.flatten(1, 2)
        ).view(bs, bev_h, bev_w, -1) # (bs, bev_h, bev_w, hidden2)
        bev_feats = self.bev_out_proj(bev_feats) # (bs, bev_h, bev_w, bev_z*bev_c)
        bev_feats = bev_feats.view(bs, bev_h, bev_w, bev_z, bev_c).permute(0, 4, 1, 2, 3) # (bs, bev_c, bev_h, bev_w, bev_z)

        return bev_feats
    
    def formulate_voxels(self, prior_voxels, prior_voxels_coords):
        bs = len(prior_voxels)
        voxel_resolution = torch.ceil(
            (self.prior_pc_range[3:] - self.prior_pc_range[:3]) / self.prior_voxel_size
        ).long()
        voxels = []
        for i in range(bs):
            points = prior_voxels[i]
            coords = prior_voxels_coords[i].long()
            dim_feats = points.size(1)
            voxel = torch.zeros((*voxel_resolution, dim_feats), 
                                dtype=torch.float32, device=points.device)
            voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = points.float()
            voxels.append(voxel)
        
        voxels = torch.stack(voxels)
        return voxels
