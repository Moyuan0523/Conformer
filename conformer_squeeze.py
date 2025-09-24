import torch
import torch.nn as nn
from functools import partial
from conformer import Conformer, Block
from squeeze_modules import SqueezeNet, FCU
from transformers import ViTModel
import torchvision.models as models
from collections import OrderedDict

class ConformerSqueeze(nn.Module):
    def __init__(self, num_classes=1000, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=6, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, conv_stem=True,
                 pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.pretrained = pretrained

        # Conv1 Stem
        if conv_stem:
            self.conv_stem = nn.Sequential(
                nn.Conv2d(in_chans, 96, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv_stem = nn.Identity()

        # ViT path
        self.patch_embed = nn.Conv2d(96, embed_dim, kernel_size=patch_size//2, stride=patch_size//2)
        num_patches = (img_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        # SqueezeNet path
        self.squeeze_net = SqueezeNet(features_only=True)
        
        # FCUs for feature alignment
        self.fcu1 = FCU(embed_dim, 96)  # For Block 1-2 output
        self.fcu2 = FCU(embed_dim, 256)  # For Block 3-4 output
        self.fcu3 = FCU(embed_dim, 512)  # For Block 5-6 output
        
        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, num_classes)

        # Load pretrained weights if specified
        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        # Load pretrained ViT weights from Hugging Face
        vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Only load the first 6 transformer blocks
        for i in range(min(6, len(self.blocks))):
            # Load transformer block weights
            self.blocks[i].load_state_dict(vit_model.encoder.layer[i].state_dict(), strict=False)
            
        # Load position embeddings and cls token
        self.pos_embed.data = vit_model.embeddings.position_embeddings
        self.cls_token.data = vit_model.embeddings.cls_token
        
        # Load pretrained SqueezeNet weights from torchvision
        squeezenet = models.squeezenet1_0(pretrained=True)
        
        # Create a state dict for our modified SqueezeNet
        squeeze_state_dict = OrderedDict()
        
        # Map the pretrained weights to our modified architecture
        squeeze_state_dict.update({
            'fire2.squeeze.weight': squeezenet.features[2].squeeze.weight,
            'fire2.expand1x1.weight': squeezenet.features[2].expand1x1.weight,
            'fire2.expand3x3.weight': squeezenet.features[2].expand3x3.weight,
            # ... 添加其他 fire 模塊的權重映射
        })
        
        # Load the mapped weights to our SqueezeNet
        self.squeeze_net.load_state_dict(squeeze_state_dict, strict=False)
        
        print("Pretrained weights loaded successfully!")

    def _get_vit_features(self, x):
        B, C, H, W = x.shape
        if C != 96:
            raise ValueError(f"Expected input channel size 96, but got {C}")
            
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        features = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 1:  # After block 2
                features['block1_2'] = self.fcu1(x[:, 1:])
            elif i == 3:  # After block 4
                features['block3_4'] = self.fcu2(x[:, 1:])
            elif i == 5:  # After block 6
                features['block5_6'] = self.fcu3(x[:, 1:])
        
        return features

    def forward(self, x):
        # Initial convolution
        x = self.conv_stem(x)
        
        # Get ViT features
        vit_features = self._get_vit_features(x)
        
        # SqueezeNet path with feature fusion at each stage
        squeeze_features = self.squeeze_net(x, vit_features=vit_features, return_features=True)
        
        # Multi-scale fusion
        x = torch.cat([
            squeeze_features['pool3'],  # Already fused with block3_4
            squeeze_features['pool5'],  # Already fused with block5_6
            squeeze_features['fire9']
        ], dim=1)
        x = self.fusion(x)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x