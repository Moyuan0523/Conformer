import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from functools import partial
import argparse
import time
import copy
from pathlib import Path
from PIL import Image
import numpy as np
import logging
from datetime import datetime
from fvcore.nn import FlopCountAnalysis

from conformer_squeeze import ConformerSqueeze
from datasets import build_dataset
import utils
import torch.nn as nn
import torchvision.transforms as transforms

# 自定義 Mixup 實現
class Mixup:
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None,
                 prob=1.0, switch_prob=0.5, mode='batch',
                 label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode

    def _mixup_data(self, x, y):
        alpha = self.mixup_alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def __call__(self, x, target):
        if np.random.rand() < self.prob:
            if self.mode == 'batch':
                x, y_a, y_b, lam = self._mixup_data(x, target)
                return x, y_a, y_b, lam
        return x, target, target, 1.

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * torch.nn.functional.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def get_args_parser():
    parser = argparse.ArgumentParser('Conformer evaluation and training script', add_help=False)
    
    # 基本參數
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--model-type', default='base', choices=['base', 'small'],
                        help='ConformerSqueeze model type: base or small')
    
    # 優化器參數
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
                        
    # 學習率設定
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr')
    parser.add_argument('--min-lr', type=float, default=0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
                        
    # EMA 相關
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)    # 預設為 False
    
    # 數據集參數
    parser.add_argument('--data-path', default=((Path(__file__).resolve().parent).parent) / "sow" / "Dataset_7_29" / "ultrasound_split", type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'CIFAR10', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='number of classes')
    parser.add_argument('--input-size', default=224, type=int,
                        help='image input size')
    
    # 數據增強參數
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    
    # Random Erase
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    
    # Mixup 參數
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # 模型參數
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size for image tokenization (default: 16)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1')

    # 運行設定
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num-workers', default=10, type=int)
    parser.add_argument('--eval-only', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--output-dir', type=str, default='./runs',
                        help='path where to save logs and checkpoints')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='experiment name for logging')

    return parser

def create_model(args):
    # 根據模型類型設置參數
    if args.model_type == 'base':
        embed_dim = 768
        num_heads = 12
    else:  # small
        embed_dim = 384
        num_heads = 6

    # 創建模型實例，並自動加載預訓練權重
    model = ConformerSqueeze(
        patch_size=16,  # 固定為 16，因為預訓練模型也是使用 16x16 的 patch
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        pretrained=True
    )
    return model

# def load_checkpoint(model, checkpoint_path):
#     # 加載預訓練權重
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     # 如果checkpoint中包含'model'鍵（這是常見的格式）
#     if 'model' in checkpoint:
#         state_dict = checkpoint['model']
#     else:
#         state_dict = checkpoint
    
#     model.load_state_dict(state_dict, strict=False)
#     return model

def prepare_input(image_path, image_size=224):
    # 準備輸入數據的轉換
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加載和預處理圖像
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # 添加batch維度

def benchmark_model(model, device, input_size=(1, 3, 224, 224), num_runs=100):

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs 測試 (用 copy 避免污染原模型)
    flops = 0
    try:
        m = copy.deepcopy(model).eval().cpu()
        dummy_cpu = torch.randn(input_size)
        flops = float(FlopCountAnalysis(m, dummy_cpu).total())
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to compute FLOPs: {str(e)}. Skipping FLOPs calculation.")
        flops = -1  # 標記為無法計算

    # Inference benchmark
    dummy_input = torch.randn(input_size).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # 預熱
        for _ in range(10):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(num_runs):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    if was_training:
        model.train()

    inference_time = (end - start) / num_runs * 1000.0  # 毫秒
    throughput = num_runs / (end - start)               # FPS

    report = (
        f"Details:\n"
        f"[Parameters]\n"
        f"  Total parameters: {total_params:,}\n"
        f"  Trainable parameters: {trainable_params:,}\n"
        f"  Non-trainable parameters: {total_params - trainable_params:,}\n"
        f"[Complexity]\n"
        f"  FLOPs (fvcore) @224x224, bs=1: {flops/1e9:.3f} GFLOPs\n"
        f"[Efficiency]\n"
        f"  Inference Time: {inference_time:.2f} ms\n"
        f"  Throughput: {throughput:.2f} FPS\n"
    )

    return report


def main(args):
    # 設置日誌
    if not args.experiment_name:
        args.experiment_name = f'conformer_{args.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    log_dir = Path(args.output_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日誌
    log_file = log_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 記錄實驗配置
    logger.info(f"{'='*50}\nExperiment: {args.experiment_name}\n{'='*50}")
    logger.info(f"Arguments:\n{args}\n{'-'*50}")
    
    # 設置設備
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 固定隨機種子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 加載數據集
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    logger.info(f"Using dataset from {args.data_path}\n{args.nb_classes} classes")

    # 設置 mixup 和 loss function
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes
        )
        # mixup 使用特別的 loss
        criterion = SoftTargetCrossEntropy()
    else:
        # 一般使用 label smoothing loss
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    
    # 創建數據加載器
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 創建模型
    print(f"Creating ConformerSqueeze-{args.model_type}")
    model = create_model(args)
    model = model.to(device)

    # 設置優化器
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                betas=(0.9, 0.999) if args.opt_betas is None else tuple(args.opt_betas),
                                eps=args.opt_eps,
                                weight_decay=args.weight_decay)

    # 設置學習率規劃器
    if args.sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=args.epochs - args.warmup_epochs - args.cooldown_epochs,
                                                             eta_min=args.min_lr)
    else:
        scheduler = None

    # 打印模型資訊
    print("\033[92m[Model Statistics]\033[0m")
    report = benchmark_model(model, device)
    print(report)
    
    print("\033[92m[Training Configuration]\033[0m")
    print(f"Optimizer: {args.opt} (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Scheduler: {args.sched} (warmup={args.warmup_epochs}, cooldown={args.cooldown_epochs})")
    print(f"Mixup: {args.mixup}, Cutmix: {args.cutmix}, Label smoothing: {args.smoothing}")
    print(f"Model EMA: {args.model_ema}")

    # 評估模式
    model.eval()

    if args.eval_only:
        # 在驗證集上評估
        print("Running evaluation...")
        model.eval()
        with torch.no_grad():
            for images, target in data_loader_val:
                images = images.to(device)
                target = target.to(device)
                output = model(images)
                # 這裡可以添加評估指標的計算
    else:
        # 訓練邏輯
        print(f"Starting training for {args.epochs} epochs")
        for epoch in range(args.epochs):
            model.train()
            
            for batch_idx, (images, target) in enumerate(data_loader_train):
                images = images.to(device)
                target = target.to(device)

                # 如果有 mixup，則應用 mixup
                if mixup_fn is not None:
                    images, target_a, target_b, lam = mixup_fn(images, target)
                
                # 前向傳播
                output = model(images)
                
                # 計算損失
                if mixup_fn is not None:
                    loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                else:
                    loss = criterion(output, target)

                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

                if batch_idx % 20 == 0:
                    logger.info(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(data_loader_train.dataset)} '
                          f'({100. * batch_idx / len(data_loader_train):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # 更新學習率
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch {epoch}: lr = {current_lr}')
            
            # 每個 epoch 結束後評估
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for images, target in data_loader_val:
                    images = images.to(device)
                    target = target.to(device)
                    output = model(images)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(data_loader_val)
            accuracy = 100. * correct / len(data_loader_val.dataset)
            logger.info(f'Validation set: Average loss: {val_loss:.4f}, '
                  f'Accuracy: {correct}/{len(data_loader_val.dataset)} ({accuracy:.2f}%)\n')
            
            # 保存每個 epoch 的指標
            metrics_file = log_dir / 'metrics.txt'
            with open(metrics_file, 'a') as f:
                f.write(f'Epoch {epoch}: loss={val_loss:.4f}, accuracy={accuracy:.2f}%\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConformerSqueeze evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # 創建輸出目錄
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)