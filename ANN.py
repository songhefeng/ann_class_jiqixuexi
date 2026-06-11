# -*- coding: utf-8 -*-
"""
威斯康星乳腺癌数据集分类 - ANN人工神经网络
课程大作业
作者：宋禾丰
日期：2026-06-09
环境：Python + PyTorch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

# 配置Matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # ===================== 1. 数据加载与预处理 =====================
    df = pd.read_csv(r"D:\data.csv")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.iloc[:, 2:32].values
    y = df["diagnosis"].values

    print("数据集形状：", df.shape)
    print("标签分布：\n", df["diagnosis"].value_counts())

    # ===================== 2. 数据集划分与标准化 =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 保存特征名称，用于后续可视化
    feature_names = df.columns[2:32].tolist()

    # ===================== 3. 特征相关性热力图 =====================
    feature_df = df.iloc[:, 2:32]
    corr = feature_df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("数据集特征相关性热力图-宋禾丰")
    plt.tight_layout()
    plt.show()

    # ===================== 4. 神经网络模型构建 =====================
    class ANN(nn.Module):
        def __init__(self, input_size):
            super(ANN, self).__init__()
            self.fc1 = nn.Linear(input_size, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    model = ANN(input_size=30)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ===================== 5. 模型训练 =====================
    epochs = 100
    train_losses = []

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # ===================== 6. 模型预测与综合性能评价 =====================
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_class = (y_pred_test > 0.5).float()

    acc = accuracy_score(y_test, y_pred_class)
    print(f"测试集准确率：{acc:.4f}")

    print("\n分类报告：")
    print(classification_report(y_test, y_pred_class, target_names=["良性", "恶性"]))

    # ===================== 7. 模型结果可视化（混淆矩阵、损失曲线） =====================
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("ANN模型混淆矩阵-宋禾丰")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()

    # 训练损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="训练损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练损失曲线-宋禾丰")
    plt.legend()
    plt.show()

    # ===================== 8. 模型综合性能分析（ROC曲线、P-R曲线） =====================
    y_true = y_test.numpy().ravel()
    y_score = y_pred_test.numpy().ravel()

    # ROC曲线 + AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率(FPR)')
    plt.ylabel('真阳性率(TPR)')
    plt.title('ANN模型ROC曲线-宋禾丰')
    plt.legend(loc="lower right")
    plt.show()

    # 精确率-召回率(P-R)曲线
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label='精确率-召回率曲线')
    plt.xlabel('召回率(Recall)')
    plt.ylabel('精确率(Precision)')
    plt.title('ANN模型精确率-召回率曲线-宋禾丰')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.show()

    # ===================== 9. 模型可解释性分析（SHAP特征图/船形图） =====================
    def model_predict(data):
        data_tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            output = model(data_tensor)
        return output.numpy()

    background = X_train.numpy()[:100]
    test_samples = X_test.numpy()[:50]

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_samples)

    # SHAP蜂群图（船形图）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values.reshape(-1, 30), test_samples, feature_names=feature_names, show=False)
    plt.title("ANN模型特征贡献SHAP蜂群图(船形图)-宋禾丰")
    plt.tight_layout()
    plt.show()

    # SHAP特征重要性条形图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values.reshape(-1, 30), test_samples, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("ANN模型特征重要性SHAP条形图-宋禾丰")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
