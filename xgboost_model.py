import os
import pandas as pd
import numpy as np
# 将xgboost库导入重命名为xgb_lib，避免与本地文件名冲突
import xgboost as xgb_lib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
from scipy import stats

# 加载数据
def load_data(data_dir='data', train_years=range(1996, 2000), test_year=2000):
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
def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(12, 8))
    xgb_lib.plot_importance(model, max_num_features=20, height=0.8, importance_type='gain')
    plt.title('XGBoost特征重要性')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')

# 绘制评估结果
def plot_evaluation_results(predictions, targets):
    # 创建一个包含多个子图的大图
    plt.figure(figsize=(15, 12))
    
    # 1. 预测值与真实值的散点图
    plt.subplot(2, 2, 1)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值')
    
    # 2. 残差图（预测误差）
    plt.subplot(2, 2, 2)
    residuals = np.array(predictions) - np.array(targets)
    plt.scatter(targets, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('真实值')
    plt.ylabel('残差 (预测值 - 真实值)')
    plt.title('残差图')
    
    # 3. 误差分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title('预测误差分布')
    
    # 4. 预测值与残差的关系图
    plt.subplot(2, 2, 4)
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('预测值与残差关系图')
    
    plt.tight_layout()
    plt.savefig('xgboost_regression_results.png')
    
    # 额外创建一个图表：真实值和预测值的时间序列对比
    if len(targets) > 100:
        # 如果样本太多，只显示前100个样本
        sample_size = 100
    else:
        sample_size = len(targets)
    
    plt.figure(figsize=(12, 6))
    indices = range(sample_size)
    plt.plot(indices, targets[:sample_size], 'b-', label='真实值', alpha=0.7)
    plt.plot(indices, predictions[:sample_size], 'r-', label='预测值', alpha=0.7)
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.title('真实值与预测值对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('xgboost_prediction_comparison.png')
    
    # 创建Q-Q图（量化-量化图）检查误差的正态性
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q图：检查误差的正态性')
    plt.savefig('xgboost_qq_plot.png')
    
    plt.show()

# 独立运行XGBoost（不依赖BERT特征）
def main():
    parser = argparse.ArgumentParser(description='XGBoost Regression for Financial Reports')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--num_rounds', type=int, default=100, help='XGBoost训练轮数')
    parser.add_argument('--val_size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model_path', type=str, default='xgboost_model.pkl', help='模型保存/加载路径')
    parser.add_argument('--force_train', action='store_true', help='强制重新训练模型，即使本地已存在模型')
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 加载数据
    train_df, test_df = load_data(args.data_dir)
    print(f'训练数据大小: {len(train_df)}')
    print(f'测试数据大小: {len(test_df)}')
    
    # 提取特征和目标变量
    # 注意：这里只使用logvol-12作为特征，没有使用文本特征
    # 在train.py中，我们将结合BERT提取的文本特征
    X_train = train_df[['logvol-12']].values
    y_train = train_df['logvol+12'].values
    
    X_test = test_df[['logvol-12']].values
    y_test = test_df['logvol+12'].values
    
    # 划分训练集和验证集
    train_size = int((1 - args.val_size) * len(X_train))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    
    # 检查是否存在已训练的模型
    model_exists = os.path.isfile(args.model_path)
    
    if model_exists and not args.force_train:
        print(f'发现已训练的模型: {args.model_path}')
        print('加载已有模型...')
        try:
            with open(args.model_path, 'rb') as f:
                model = pickle.load(f)
            print('模型加载成功！')
        except Exception as e:
            print(f'模型加载失败: {str(e)}')
            print('将重新训练模型...')
            model_exists = False
    
    # 如果模型不存在或强制重新训练
    if not model_exists or args.force_train:
        print('开始训练XGBoost模型...')
        model = train_xgboost(
            X_train_split, 
            y_train_split, 
            X_val, 
            y_val, 
            num_rounds=args.num_rounds
        )
        
        # 保存模型
        with open(args.model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'模型训练完成，已保存到: {args.model_path}')
    
    # 评估模型
    print('开始评估模型...')
    mse, rmse, r2, predictions = evaluate_model(model, X_test, y_test)
    
    # 绘制结果
    print('绘制评估结果...')
    plot_evaluation_results(predictions, y_test)
    
    # 绘制特征重要性（这里只有一个特征，所以不是很有意义）
    # plot_feature_importance(model, ['logvol-12'])

if __name__ == '__main__':
    main()
