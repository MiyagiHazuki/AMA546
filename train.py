import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
# 将xgboost库导入重命名为xgb_lib，避免与本地xgboost.py文件冲突
import xgboost as xgb_lib
from xgboost import DMatrix  # 显式导入DMatrix类
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
from scipy import stats
import time

# 导入BERT模型和数据集
from bert import BERTRegressor, FinancialReportDataset, load_data, set_seed
from transformers import BertTokenizer, AutoTokenizer
# 导入本地XGBoost模型模块
import xgboost_model

# 从BERT模型中提取特征
def extract_bert_features(model, data_loader, device):
    model.eval()
    all_features = []
    all_logvol_minus_12 = []
    all_logvol_plus_12 = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='提取BERT特征'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logvol_minus_12 = batch['logvol_minus_12'].to(device)
            logvol_plus_12 = batch['logvol_plus_12'].to(device)
            
            # 获取BERT的输出
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 使用[CLS]标记的输出作为文本表示
            # 注意：不同模型可能有不同的输出结构
            if hasattr(outputs, 'pooler_output'):
                pooled_output = outputs.pooler_output
            else:
                # 对于某些模型（如Longformer），可能需要使用last_hidden_state的第一个token
                pooled_output = outputs.last_hidden_state[:, 0, :]
            
            # 将特征移到CPU并转换为NumPy数组
            features = pooled_output.cpu().numpy()
            logvol_minus_12_np = logvol_minus_12.cpu().numpy()
            logvol_plus_12_np = logvol_plus_12.cpu().numpy()
            
            all_features.append(features)
            all_logvol_minus_12.append(logvol_minus_12_np)
            all_logvol_plus_12.append(logvol_plus_12_np)
    
    # 合并所有批次的特征
    all_features = np.vstack(all_features)
    all_logvol_minus_12 = np.concatenate(all_logvol_minus_12)
    all_logvol_plus_12 = np.concatenate(all_logvol_plus_12)
    
    return all_features, all_logvol_minus_12, all_logvol_plus_12

# 训练XGBoost模型
def train_xgboost(X_train, y_train, X_val=None, y_val=None, params=None, num_rounds=100):
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1,
            'lambda': 1,
            'seed': 42
        }
    
    # 创建DMatrix对象
    try:
        dtrain = xgb_lib.DMatrix(X_train, label=y_train)
    except Exception as e:
        print(f"创建DMatrix时出错: {str(e)}")
        print("尝试使用替代方法...")
        dtrain = xgb_lib.DMatrix(data=X_train, label=y_train)
    
    # 如果有验证集
    evals = []
    if X_val is not None and y_val is not None:
        try:
            dval = xgb_lib.DMatrix(X_val, label=y_val)
        except Exception as e:
            print(f"创建验证集DMatrix时出错: {str(e)}")
            print("尝试使用替代方法...")
            dval = xgb_lib.DMatrix(data=X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'val')]
    else:
        evals = [(dtrain, 'train')]
    
    # 训练模型
    print("训练XGBoost模型...")
    model = xgb_lib.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    return model

# 评估模型
def evaluate_model(model, X_test, y_test):
    try:
        dtest = xgb_lib.DMatrix(X_test)
    except Exception as e:
        print(f"创建测试集DMatrix时出错: {str(e)}")
        print("尝试使用替代方法...")
        dtest = xgb_lib.DMatrix(data=X_test)
    
    predictions = model.predict(dtest)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f'测试结果:')
    print(f'  MSE: {mse:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R²: {r2:.4f}')
    
    return mse, rmse, r2, predictions

# 绘制特征重要性
def plot_feature_importance(model, feature_names=None):
    plt.figure(figsize=(12, 8))
    xgb_lib.plot_importance(model, max_num_features=20, height=0.8, importance_type='gain')
    plt.title('Feature Importance of XGBoost')
    plt.tight_layout()
    plt.savefig('bert_xgboost_feature_importance.png')

# 绘制评估结果
def plot_evaluation_results(predictions, targets):
    # 创建一个包含多个子图的大图
    plt.figure(figsize=(15, 12))
    
    # 1. 预测值与真实值的散点图
    plt.subplot(2, 2, 1)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Predicted Value vs True Value')
    
    # 2. 残差图（预测误差）
    plt.subplot(2, 2, 2)
    residuals = np.array(predictions) - np.array(targets)
    plt.scatter(targets, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Value')
    plt.ylabel('Residual (Predicted Value - True Value)')
    plt.title('Residual Plot')
    
    # 3. 误差分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Error')
    
    # 4. 预测值与残差的关系图
    plt.subplot(2, 2, 4)
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Relationship between Predicted Value and Residual')
    
    plt.tight_layout()
    plt.savefig('bert_xgboost_regression_results.png')
    
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
    plt.ylabel('Value')
    plt.title('Comparison of True Value and Predicted Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bert_xgboost_prediction_comparison.png')
    
    # 创建Q-Q图（量化-量化图）检查误差的正态性
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q plot: Check the normality of the error')
    plt.savefig('bert_xgboost_qq_plot.png')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='BERT+XGBoost Regression for Financial Reports')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--bert_model', type=str, default='allenai/longformer-base-4096', help='BERT预训练模型名称')
    parser.add_argument('--bert_model_path', type=str, default='best_bert_regressor.pt', help='BERT模型路径')
    parser.add_argument('--xgboost_model_path', type=str, default='bert_xgboost_model.pkl', help='XGBoost模型保存/加载路径')
    parser.add_argument('--max_length', type=int, default=4096, help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_rounds', type=int, default=100, help='XGBoost训练轮数')
    parser.add_argument('--val_size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--force_train', action='store_true', help='强制重新训练模型，即使本地已存在模型')
    parser.add_argument('--force_extract', action='store_true', help='强制重新提取BERT特征，即使本地已存在特征文件')
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
    train_dataset_df = train_df.iloc[:train_size].reset_index(drop=True)
    val_dataset_df = train_df.iloc[train_size:].reset_index(drop=True)
    
    # 创建数据集
    train_dataset = FinancialReportDataset(train_dataset_df, tokenizer, args.max_length)
    val_dataset = FinancialReportDataset(val_dataset_df, tokenizer, args.max_length)
    test_dataset = FinancialReportDataset(test_df, tokenizer, args.max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)  # 不打乱顺序，保持特征和标签对应
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 检查BERT模型是否存在
    if not os.path.isfile(args.bert_model_path):
        print(f'错误: BERT模型文件 {args.bert_model_path} 不存在')
        print('请先运行bert.py训练BERT模型')
        return
    
    # 加载BERT模型
    print(f'加载BERT模型: {args.bert_model_path}')
    bert_model = BERTRegressor(args.bert_model)
    
    try:
        # 尝试直接加载模型
        bert_model.load_state_dict(torch.load(args.bert_model_path, map_location=device))
    except RuntimeError as e:
        print(f"标准加载失败，错误: {str(e)}")
        print("尝试使用strict=False加载模型...")
        # 使用strict=False允许加载部分权重
        bert_model.load_state_dict(torch.load(args.bert_model_path, map_location=device), strict=False)
        print("模型已加载，但某些权重可能未匹配")
    
    bert_model.to(device)
    print('BERT模型加载成功')
    
    # 定义特征文件路径
    train_features_file = 'bert_train_features.npz'
    val_features_file = 'bert_val_features.npz'
    test_features_file = 'bert_test_features.npz'
    
    # 检查是否需要提取特征
    extract_features = args.force_extract or not all(os.path.isfile(f) for f in [train_features_file, val_features_file, test_features_file])
    
    if extract_features:
        print('开始提取BERT特征...')
        
        # 提取训练集特征
        print('提取训练集特征...')
        train_features, train_logvol_minus_12, train_logvol_plus_12 = extract_bert_features(bert_model, train_loader, device)
        np.savez(train_features_file, 
                 features=train_features, 
                 logvol_minus_12=train_logvol_minus_12, 
                 logvol_plus_12=train_logvol_plus_12)
        
        # 提取验证集特征
        print('提取验证集特征...')
        val_features, val_logvol_minus_12, val_logvol_plus_12 = extract_bert_features(bert_model, val_loader, device)
        np.savez(val_features_file, 
                 features=val_features, 
                 logvol_minus_12=val_logvol_minus_12, 
                 logvol_plus_12=val_logvol_plus_12)
        
        # 提取测试集特征
        print('提取测试集特征...')
        test_features, test_logvol_minus_12, test_logvol_plus_12 = extract_bert_features(bert_model, test_loader, device)
        np.savez(test_features_file, 
                 features=test_features, 
                 logvol_minus_12=test_logvol_minus_12, 
                 logvol_plus_12=test_logvol_plus_12)
        
        print('BERT特征提取完成')
    else:
        print('加载已保存的BERT特征...')
        
        # 加载训练集特征
        train_data = np.load(train_features_file)
        train_features = train_data['features']
        train_logvol_minus_12 = train_data['logvol_minus_12']
        train_logvol_plus_12 = train_data['logvol_plus_12']
        
        # 加载验证集特征
        val_data = np.load(val_features_file)
        val_features = val_data['features']
        val_logvol_minus_12 = val_data['logvol_minus_12']
        val_logvol_plus_12 = val_data['logvol_plus_12']
        
        # 加载测试集特征
        test_data = np.load(test_features_file)
        test_features = test_data['features']
        test_logvol_minus_12 = test_data['logvol_minus_12']
        test_logvol_plus_12 = test_data['logvol_plus_12']
        
        print('BERT特征加载成功')
    
    # 合并BERT特征和logvol-12特征
    print('合并BERT特征和logvol-12特征...')
    X_train = np.hstack((train_features, train_logvol_minus_12.reshape(-1, 1)))
    y_train = train_logvol_plus_12
    
    X_val = np.hstack((val_features, val_logvol_minus_12.reshape(-1, 1)))
    y_val = val_logvol_plus_12
    
    X_test = np.hstack((test_features, test_logvol_minus_12.reshape(-1, 1)))
    y_test = test_logvol_plus_12
    
    print(f'特征维度: {X_train.shape[1]}')
    print(f'训练样本数: {X_train.shape[0]}')
    print(f'验证样本数: {X_val.shape[0]}')
    print(f'测试样本数: {X_test.shape[0]}')
    
    # 检查是否存在已训练的XGBoost模型
    model_exists = os.path.isfile(args.xgboost_model_path)
    
    if model_exists and not args.force_train:
        print(f'发现已训练的XGBoost模型: {args.xgboost_model_path}')
        print('加载已有模型...')
        try:
            with open(args.xgboost_model_path, 'rb') as f:
                model = pickle.load(f)
            print('XGBoost模型加载成功！')
        except Exception as e:
            print(f'XGBoost模型加载失败: {str(e)}')
            print('将重新训练模型...')
            model_exists = False
    
    # 如果模型不存在或强制重新训练
    if not model_exists or args.force_train:
        print('开始训练BERT+XGBoost模型...')
        start_time = time.time()
        model = train_xgboost(
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            num_rounds=args.num_rounds
        )
        training_time = time.time() - start_time
        print(f'训练完成，耗时: {training_time:.2f}秒')
        
        # 保存模型
        with open(args.xgboost_model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'模型训练完成，已保存到: {args.xgboost_model_path}')
    
    # 评估模型
    print('开始评估模型...')
    mse, rmse, r2, predictions = evaluate_model(model, X_test, y_test)
    
    # 绘制结果
    print('绘制评估结果...')
    plot_evaluation_results(predictions, y_test)
    
    # 绘制特征重要性
    print('绘制特征重要性...')
    plot_feature_importance(model)
    
    # 与纯XGBoost模型比较（如果存在）
    xgboost_only_model_path = 'xgboost_model.pkl'
    if os.path.isfile(xgboost_only_model_path):
        print('加载纯XGBoost模型进行比较...')
        try:
            with open(xgboost_only_model_path, 'rb') as f:
                xgboost_only_model = pickle.load(f)
            
            # 准备纯XGBoost模型的测试数据
            X_test_xgboost_only = test_logvol_minus_12.reshape(-1, 1)
            
            # 评估纯XGBoost模型
            print('评估纯XGBoost模型...')
            xgboost_only_mse, xgboost_only_rmse, xgboost_only_r2, _ = evaluate_model(
                xgboost_only_model, 
                X_test_xgboost_only, 
                y_test
            )
            
            # 打印比较结果
            print('\n模型比较:')
            print(f'  BERT+XGBoost - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
            print(f'  纯XGBoost    - MSE: {xgboost_only_mse:.4f}, RMSE: {xgboost_only_rmse:.4f}, R²: {xgboost_only_r2:.4f}')
            
            # 计算改进百分比
            mse_improvement = (xgboost_only_mse - mse) / xgboost_only_mse * 100
            rmse_improvement = (xgboost_only_rmse - rmse) / xgboost_only_rmse * 100
            r2_improvement = (r2 - xgboost_only_r2) / abs(xgboost_only_r2) * 100 if xgboost_only_r2 != 0 else float('inf')
            
            print(f'  改进 - MSE: {mse_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%, R²: {r2_improvement:.2f}%')
        except Exception as e:
            print(f'加载纯XGBoost模型失败: {str(e)}')

if __name__ == '__main__':
    main()
