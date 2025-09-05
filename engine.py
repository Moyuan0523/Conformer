"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import numpy as np
import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if isinstance(outputs, list):
                loss_list = [criterion(o, targets) / len(outputs) for o in outputs]
                loss = sum(loss_list)
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if isinstance(outputs, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
@torch.no_grad()
def evaluate(data_loader, model, device, output_dir):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    # 若要同時觀察各 head 的混淆矩陣，可另外開三個容器（可選）
    # all_preds_h1, all_preds_h2 = [], []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # forward
        # 新版 AMP 提示：若未升級，仍可用 autocast(cuda)；這裡先沿用原寫法避免相依版本差異
        with torch.cuda.amp.autocast():
            output = model(images)

            if isinstance(output, list):
                # Conformer 兩個 head
                # 統一用平均 logits 當「融合」輸出
                logits_fused = (output[0] + output[1]) / 2.0
                loss_list = [criterion(o, target) / len(output) for o in output]
                loss = sum(loss_list)

                # 分別計算三種 acc：fused、head1、head2
                acc1_fused = accuracy(logits_fused, target, topk=(1,))[0]
                acc1_h1    = accuracy(output[0],    target, topk=(1,))[0]
                acc1_h2    = accuracy(output[1],    target, topk=(1,))[0]

                logits = logits_fused  # 用於後續 argmax / 混淆矩陣

            else:
                # 其他模型只有單一輸出
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                logits = output

        # 蒐集混淆矩陣資料（以 fused/單頭為主）
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # 若也想要各 head 的混淆矩陣（可選）
        # if isinstance(output, list):
        #     all_preds_h1.extend(torch.argmax(output[0], dim=1).cpu().numpy())
        #     all_preds_h2.extend(torch.argmax(output[1], dim=1).cpu().numpy())

        # logging 累計
        bs = images.size(0)
        metric_logger.update(loss=loss.item())
        if isinstance(output, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            metric_logger.meters['acc1'].update(acc1_fused.item(), n=bs)     # fused
            metric_logger.meters['acc1_head1'].update(acc1_h1.item(), n=bs)  # head1
            metric_logger.meters['acc1_head2'].update(acc1_h2.item(), n=bs)  # head2
        else:
            metric_logger.meters['acc1'].update(acc1.item(), n=bs)
            metric_logger.meters['acc5'].update(acc5.item(), n=bs)

    # ======= epoch 結束：輸出混淆矩陣 =======
    cm = confusion_matrix(all_labels, all_preds)

     # ======= 由完整驗證集計算指標 =======
    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    y_prob = np.asarray(all_probs)  # shape [N, C]

    # F1 / Precision / Recall：多類別用 macro，二分類可用 binary；為穩定起見用 macro（醫影報告常見）
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Specificity（特異度）：對每個類別算 TN/(TN+FP)，再做 macro 平均
    # 適用二分類與多分類
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    # TN_k = 所有元素總和 - 該列總和 - 該行總和 + 對角元素
    TN = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    FP = cm.sum(axis=0) - np.diag(cm)
    spec_per_class = TN / (TN + FP + 1e-12)
    spec = float(np.mean(spec_per_class))

    # AUC：二分類用 y_prob[:,1]；多分類用 ovr_macro
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    epoch_result = (f"F1(macro): {f1:.4f}, AUC: {auc:.4f}, "
            f"Specificity(macro): {spec:.4f}, Recall(macro): {rec:.4f}, Precision(macro): {prec:.4f}\n {cm}")

    # 若也要各 head 的 CM（可選）
    # if len(all_preds_h1) and len(all_preds_h2):
    #     cm_h1 = confusion_matrix(all_labels, all_preds_h1)
    #     cm_h2 = confusion_matrix(all_labels, all_preds_h2)
    #     print("Confusion Matrix (head1):\n", cm_h1)
    #     print("Confusion Matrix (head2):\n", cm_h2)
    #     np.savetxt("confusion_matrix_head1_epoch.csv", cm_h1, fmt="%d", delimiter=",")
    #     np.savetxt("confusion_matrix_head2_epoch.csv", cm_h2, fmt="%d", delimiter=",")

    # ======= 收尾輸出 =======
    if 'acc1_head1' in metric_logger.meters:  # Conformer
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} '
              'Acc@head_1 {head1_top1.global_avg:.3f} '
              'Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} '
              'loss@1 {loss_0.global_avg:.3f} '
              'loss@2 {loss_1.global_avg:.3f}'
              .format(
                  heads_top1=metric_logger.acc1,
                  head1_top1=metric_logger.acc1_head1,
                  head2_top1=metric_logger.acc1_head2,
                  losses=metric_logger.loss,
                  loss_0=metric_logger.loss_0,
                  loss_1=metric_logger.loss_1))
    else:  # 單頭
        print('* Acc@1 {top1.global_avg:.3f} '
              'Acc@5 {top5.global_avg:.3f} '
              'loss {losses.global_avg:.3f}'
              .format(
                  top1=metric_logger.acc1,
                  top5=metric_logger.acc5,
                  losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, epoch_result 

