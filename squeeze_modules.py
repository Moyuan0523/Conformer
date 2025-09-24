import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

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
        
        # Stage 1: Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 55x55
        
        # Stage 2: First Fire group (保持 55x55 尺寸)
        self.fire2 = Fire(96, 16, 64, 64)     # 55x55
        self.fire3 = Fire(128, 16, 64, 64)    # 55x55
        self.fire4 = Fire(128, 32, 128, 128)  # 55x55
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 27x27
        
        # Stage 3: Second Fire group (保持 27x27 尺寸)
        self.fire5 = Fire(256, 32, 128, 128)  # 27x27
        self.fire6 = Fire(256, 48, 192, 192)  # 27x27
        self.fire7 = Fire(384, 48, 192, 192)  # 27x27
        self.fire8 = Fire(384, 64, 256, 256)  # 27x27
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 13x13
        
        # Stage 4: Final Fire
        self.fire9 = Fire(512, 64, 256, 256)  # 13x13
        
        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.conv1:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.features_only = features_only

    def forward(self, x, vit_features=None, return_features=False):
        # 檢查輸入特徵的形狀
        B, C, H, W = x.shape
        if C != 3:
            raise ValueError(f"Expected input channel size 3, but got {C}")
            
        # Initial convolution and pooling (224 -> 111 -> 55)
        x = self.conv1(x)      # 111x111
        x = self.relu1(x)
        pool1 = self.pool1(x)  # 55x55
        
        # Add VIT features if provided
        if vit_features is not None:
            # 檢查必要的 ViT 特徵是否存在
            required_features = ['block1_2', 'block3_4', 'block5_6']
            missing_features = [f for f in required_features if f not in vit_features]
            if missing_features:
                raise ValueError(f"Missing required ViT features: {missing_features}")
            
            # Stage 1: block1_2 fusion (55x55)
            if vit_features['block1_2'].shape[1] != 96:
                raise ValueError(f"Expected block1_2 channel size 96, but got {vit_features['block1_2'].shape[1]}")
            
            vit_feat1 = vit_features['block1_2']
            if vit_feat1.shape[2:] != pool1.shape[2:]:
                logger.info(f"Resizing block1_2 from {vit_feat1.shape[2:]} to {pool1.shape[2:]}")
                vit_feat1 = torch.nn.functional.interpolate(vit_feat1, 
                                                          size=pool1.shape[2:],
                                                          mode='bilinear', 
                                                          align_corners=False)
            pool1 = pool1 + vit_feat1
        
        # First fire group (維持 55x55)
        fire2 = self.fire2(pool1)   # 55x55
        fire3 = self.fire3(fire2)   # 55x55
        fire4 = self.fire4(fire3)   # 55x55
        pool3 = self.pool3(fire4)   # 27x27
        
        # Stage 2: block3_4 fusion (27x27)
        if vit_features is not None:
            if vit_features['block3_4'].shape[1] != 256:
                raise ValueError(f"Expected block3_4 channel size 256, but got {vit_features['block3_4'].shape[1]}")
            
            vit_feat2 = vit_features['block3_4']
            if vit_feat2.shape[2:] != pool3.shape[2:]:
                logger.info(f"Resizing block3_4 from {vit_feat2.shape[2:]} to {pool3.shape[2:]}")
                vit_feat2 = torch.nn.functional.interpolate(vit_feat2, 
                                                          size=pool3.shape[2:],
                                                          mode='bilinear', 
                                                          align_corners=False)
            pool3 = pool3 + vit_feat2
            
        # Second fire group (維持 27x27)
        fire5 = self.fire5(pool3)   # 27x27
        fire6 = self.fire6(fire5)   # 27x27
        fire7 = self.fire7(fire6)   # 27x27
        fire8 = self.fire8(fire7)   # 27x27
        pool5 = self.pool5(fire8)   # 13x13
        
        # Stage 3: block5_6 fusion (13x13)
        if vit_features is not None:
            if vit_features['block5_6'].shape[1] != 512:
                raise ValueError(f"Expected block5_6 channel size 512, but got {vit_features['block5_6'].shape[1]}")
            
            vit_feat3 = vit_features['block5_6']
            if vit_feat3.shape[2:] != pool5.shape[2:]:
                logger.info(f"Resizing block5_6 from {vit_feat3.shape[2:]} to {pool5.shape[2:]}")
                vit_feat3 = torch.nn.functional.interpolate(vit_feat3, 
                                                          size=pool5.shape[2:],
                                                          mode='bilinear', 
                                                          align_corners=False)
            pool5 = pool5 + vit_feat3
            
        # Final fire
        fire9 = self.fire9(pool5)  # 13x13
        
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
        
        # 確保特徵圖尺寸正確
        target_size = 56  # first stage
        if H != target_size:
            x = torch.nn.functional.interpolate(x, size=(target_size, target_size), 
                                               mode='bilinear', align_corners=False)
        return x