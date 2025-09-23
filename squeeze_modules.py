import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, features_only=True):
        super(SqueezeNet, self).__init__()
        
        # Stage 1: Initial pooling
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool1 (融合 ViT block1-2)
        )
        
        # Stage 2: First Fire group
        self.fire2 = Fire(96, 16, 64, 64)  # Input: pool1 + vit_block1_2
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # pool3 (融合 ViT block3-4)
        
        # Stage 3: Second Fire group
        self.fire5 = Fire(256, 32, 128, 128)  # Input: pool3 + vit_block3_4
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # pool5 (融合 ViT block5-6)
        
        # Stage 4: Final Fire
        self.fire9 = Fire(512, 64, 256, 256)  # Input: pool5 + vit_block5_6
        
        self.features_only = features_only

    def forward(self, x, vit_features=None, return_features=False):
        # 檢查輸入特徵的形狀
        B, C, H, W = x.shape
        if C != 96:
            raise ValueError(f"Expected input channel size 96, but got {C}")
            
        # Initial pooling
        pool1 = self.features(x)
        
        # Add VIT features if provided
        if vit_features is not None:
            # 檢查必要的 ViT 特徵是否存在
            required_features = ['block1_2', 'block3_4', 'block5_6']
            missing_features = [f for f in required_features if f not in vit_features]
            if missing_features:
                raise ValueError(f"Missing required ViT features: {missing_features}")
            
            # 檢查特徵維度
            if vit_features['block1_2'].shape[1] != 96:
                raise ValueError(f"Expected block1_2 channel size 96, but got {vit_features['block1_2'].shape[1]}")
            
            pool1 = pool1 + vit_features['block1_2']
        
        # First fire group
        fire2 = self.fire2(pool1)
        fire3 = self.fire3(fire2)
        fire4 = self.fire4(fire3)
        pool3 = self.pool3(fire4)
        
        # Add VIT features before fire5 if provided
        if vit_features is not None:
            # 檢查特徵維度
            if vit_features['block3_4'].shape[1] != 256:
                raise ValueError(f"Expected block3_4 channel size 256, but got {vit_features['block3_4'].shape[1]}")
                
            pool3 = pool3 + vit_features['block3_4']
            
        # Second fire group
        fire5 = self.fire5(pool3)
        fire6 = self.fire6(fire5)
        fire7 = self.fire7(fire6)
        fire8 = self.fire8(fire7)
        pool5 = self.pool5(fire8)
        
        # Add VIT features before fire9 if provided
        if vit_features is not None:
            # 檢查特徵維度
            if vit_features['block5_6'].shape[1] != 512:
                raise ValueError(f"Expected block5_6 channel size 512, but got {vit_features['block5_6'].shape[1]}")
                
            pool5 = pool5 + vit_features['block5_6']        
            
        # Final fire
        fire9 = self.fire9(pool5)
        
        # 根據模式返回不同的輸出
        if return_features:
            # 返回所有關鍵特徵圖，用於特徵融合和多尺度處理
            return {
                'pool1': pool1,    # 用於與 ViT block1-2 融合的特徵
                'pool3': pool3,    # 用於與 ViT block3-4 融合的特徵
                'pool5': pool5,    # 用於與 ViT block5-6 融合的特徵
                'fire9': fire9     # 最終的特徵輸出
            }
        
        # 一般模式：只返回最終特徵
        return fire9

class FCU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCU, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.norm = nn.BatchNorm2d(out_dim)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x