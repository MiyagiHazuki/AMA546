import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import argparse
from scipy import stats

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 自定义数据集
class FinancialReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.logvol_plus_12 = self.data['logvol+12'].values
        self.logvol_minus_12 = self.data['logvol-12'].values
        self.text = self.data['tok'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.text[index])
        logvol_plus_12 = float(self.logvol_plus_12[index])
        logvol_minus_12 = float(self.logvol_minus_12[index])
        
        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'logvol_plus_12': torch.tensor(logvol_plus_12, dtype=torch.float),
            'logvol_minus_12': torch.tensor(logvol_minus_12, dtype=torch.float)
        }

# 定义BERT回归模型
class BERTRegressor(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', dropout_rate=0.3):
        super(BERTRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # BERT输出维度 + logvol-12特征维度
        bert_hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(bert_hidden_size + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask, logvol_minus_12):
        # 获取BERT的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的输出作为文本表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 将BERT输出与logvol-12特征连接
        logvol_minus_12 = logvol_minus_12.unsqueeze(1)
        combined_features = torch.cat((pooled_output, logvol_minus_12), dim=1)
        
        # 通过回归器预测logvol+12
        return self.regressor(combined_features)

# 加载数据
def load_data(data_dir='data', train_years=range(1996, 2006), test_year=2006):
    train_data = []
    
    # 加载训练数据（1996-1999年）
    for year in train_years:
        file_path = os.path.join(data_dir, f"{year}.csv")
        if os.path.exists(file_path):
            print(f"加载{year}年训练数据...")
            df = pd.read_csv(file_path, sep='\t')
            train_data.append(df)
    
    # 合并训练数据
    train_df = pd.concat(train_data, ignore_index=True)
    
    # 加载测试数据（2000年）
    test_file_path = os.path.join(data_dir, f"{test_year}.csv")
    print(f"加载{test_year}年测试数据...")
    test_df = pd.read_csv(test_file_path, sep='\t')
    
    return train_df, test_df

# 训练模型
def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in train_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logvol_minus_12 = batch['logvol_minus_12'].to(device)
            logvol_plus_12 = batch['logvol_plus_12'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, logvol_minus_12)
            loss = criterion(outputs.squeeze(), logvol_plus_12)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logvol_minus_12 = batch['logvol_minus_12'].to(device)
                logvol_plus_12 = batch['logvol_plus_12'].to(device)
                
                outputs = model(input_ids, attention_mask, logvol_minus_12)
                loss = criterion(outputs.squeeze(), logvol_plus_12)
                
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_bert_regressor.pt')
            print(f'  模型已保存 (Val Loss: {best_val_loss:.4f})')
    
    return train_losses, val_losses

# 评估模型
def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logvol_minus_12 = batch['logvol_minus_12'].to(device)
            logvol_plus_12 = batch['logvol_plus_12'].to(device)
            
            outputs = model(input_ids, attention_mask, logvol_minus_12)
            loss = criterion(outputs.squeeze(), logvol_plus_12)
            total_loss += loss.item()
            
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(logvol_plus_12.cpu().numpy())
    
    # 计算评估指标
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    
    print(f'测试结果:')
    print(f'  MSE: {mse:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R²: {r2:.4f}')
    
    return mse, rmse, r2, all_predictions, all_targets

# 绘制训练过程和预测结果
def plot_results(train_losses, val_losses, predictions, targets):
    # 创建一个包含多个子图的大图
    plt.figure(figsize=(15, 12))
    
    # 1. 训练和验证损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    
    # 2. 预测值与真实值的散点图
    plt.subplot(2, 2, 2)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Predicted Value vs True Value')
    
    # 3. 残差图（预测误差）
    plt.subplot(2, 2, 3)
    residuals = np.array(predictions) - np.array(targets)
    plt.scatter(targets, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Value')
    plt.ylabel('Residual (Predicted Value - True Value)')
    plt.title('Residual Plot')
    
    # 4. 误差分布直方图
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Error')
    
    plt.tight_layout()
    plt.savefig('bert_regression_results.png')
    
    # 额外创建一个图表：真实值和预测值的时间序列对比
    if len(targets) > 100:
        # 如果样本太多，只显示前100个样本
        sample_size = 100
    else:
        sample_size = len(targets)
    
    plt.figure(figsize=(12, 6))
    indices = range(sample_size)
    plt.plot(indices, targets[:sample_size], 'b-', label='True Value', alpha=0.7)
    plt.plot(indices, predictions[:sample_size], 'r-', label='Predicted Value', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Comparison of True Values and Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_comparison.png')
    
    # 创建Q-Q图（量化-量化图）检查误差的正态性
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Checking Normality of Prediction Errors')
    plt.savefig('qq_plot.png')
    
    # 创建预测值与残差的关系图
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Predicted Value vs Residual')
    plt.grid(True, alpha=0.3)
    plt.savefig('pred_vs_residuals.png')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='BERT Regression for Financial Reports')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--bert_model', type=str, default='allenai/longformer-base-4096', help='BERT预训练模型名称')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--force_train', action='store_true', help='强制重新训练模型，即使本地已存在模型')
    parser.add_argument('--model_path', type=str, default='best_bert_regressor.pt', help='模型保存/加载路径')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    train_df, test_df = load_data(args.data_dir)
    print(f'训练数据大小: {len(train_df)}')
    print(f'测试数据大小: {len(test_df)}')
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    # 划分训练集和验证集
    train_size = int((1 - args.val_size) * len(train_df))
    train_dataset = train_df.iloc[:train_size].reset_index(drop=True)
    val_dataset = train_df.iloc[train_size:].reset_index(drop=True)
    
    # 创建数据集
    train_dataset = FinancialReportDataset(train_dataset, tokenizer, args.max_length)
    val_dataset = FinancialReportDataset(val_dataset, tokenizer, args.max_length)
    test_dataset = FinancialReportDataset(test_df, tokenizer, args.max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    model = BERTRegressor(args.bert_model)
    model.to(device)
    
    # 检查是否存在已训练的模型
    model_exists = os.path.isfile(args.model_path)
    train_losses = []
    val_losses = []
    
    if model_exists and not args.force_train:
        print(f'发现已训练的模型: {args.model_path}')
        print('加载已有模型...')
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print('模型加载成功！')
        except Exception as e:
            print(f'模型加载失败: {str(e)}')
            print('将重新训练模型...')
            model_exists = False
    
    # 如果模型不存在或强制重新训练
    if not model_exists or args.force_train:
        print('开始训练模型...')
        train_losses, val_losses = train_model(
            model, 
            train_loader, 
            val_loader, 
            device, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate
        )
        print(f'模型训练完成，已保存到: {args.model_path}')
    else:
        print('跳过训练阶段，直接进行评估...')
    
    # 评估模型
    print('开始评估模型...')
    mse, rmse, r2, predictions, targets = evaluate_model(model, test_loader, device)
    
    # 绘制结果
    if train_losses and val_losses:
        print('绘制训练过程和评估结果...')
        plot_results(train_losses, val_losses, predictions, targets)
    else:
        print('仅绘制评估结果（无训练过程图表）...')
        # 创建一个简化版的绘图函数，不包含训练损失曲线
        plot_evaluation_results(predictions, targets)

# 简化版绘图函数，用于仅评估模式（无训练过程）
def plot_evaluation_results(predictions, targets):
    # 创建一个包含多个子图的大图
    plt.figure(figsize=(15, 12))
    
    # 1. 预测值与真实值的散点图
    plt.subplot(2, 2, 1)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    
    # 2. 残差图（预测误差）
    plt.subplot(2, 2, 2)
    residuals = np.array(predictions) - np.array(targets)
    plt.scatter(targets, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals (Predictions - True Values)')
    plt.title('Residual Plot')
    
    # 3. 误差分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Errors')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    
    # 4. 预测值与残差的关系图
    plt.subplot(2, 2, 4)
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals (Predictions - True Values)')
    plt.title('Predictions vs Residuals')
    
    plt.tight_layout()
    plt.savefig('bert_regression_results.png')
    
    # 额外创建一个图表：真实值和预测值的时间序列对比
    if len(targets) > 100:
        # 如果样本太多，只显示前100个样本
        sample_size = 100
    else:
        sample_size = len(targets)
    
    plt.figure(figsize=(12, 6))
    indices = range(sample_size)
    plt.plot(indices, targets[:sample_size], 'b-', label='True Values', alpha=0.7)
    plt.plot(indices, predictions[:sample_size], 'r-', label='Predictions', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Comparison of True Values and Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_comparison.png')
    
    # 创建Q-Q图（量化-量化图）检查误差的正态性
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Checking Normality of Prediction Errors')
    plt.savefig('qq_plot.png')
    
    plt.show()

if __name__ == '__main__':
    main()
