import json
import re
import numpy as np

log_file = "/home/r76131060/SOWSOW/vit2/Result/log.txt"

# 存放各種指標
acc_list, f1_list, auc_list, sp_list, recall_list, prec_list = [], [], [], [], [], []

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()

        # 解析 JSON 區塊 (test_acc1)
        try:
            log = json.loads(line)
            if "test_acc1" in log:
                acc_list.append(float(log["test_acc1"]))
        except json.JSONDecodeError:
            pass

        # 解析指標區塊
        if "F1(macro)" in line and "AUC" in line:
            match = re.findall(r"([\w\(\)]+): ([0-9.]+)", line)
            if match:
                metric_dict = {k: float(v) for k, v in match}
                f1_list.append(metric_dict.get("F1(macro)", np.nan))
                auc_list.append(metric_dict.get("AUC", np.nan))
                sp_list.append(metric_dict.get("Specificity(macro)", np.nan))
                recall_list.append(metric_dict.get("Recall(macro)", np.nan))
                prec_list.append(metric_dict.get("Precision(macro)", np.nan))

# 統計輸出
def summarize(values, name):
    if len(values) == 0:
        print(f"⚠️ {name} 沒有找到")
    else:
        arr = np.array(values)
        print(f"{name}: 平均 {np.mean(arr):.4f}, 最佳 {np.max(arr):.4f}, 標準差 {np.std(arr, ddof=1):.4f}")

print("=== 測試結果統計 ===")
summarize(acc_list, "Acc@1")
summarize(f1_list, "F1(macro)")
summarize(auc_list, "AUC")
summarize(sp_list, "Specificity(macro)")
summarize(recall_list, "Recall(macro)")
summarize(prec_list, "Precision(macro)")
