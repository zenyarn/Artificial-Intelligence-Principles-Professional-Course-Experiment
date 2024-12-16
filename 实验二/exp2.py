# exp2_main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def load_and_explore_data(train_path):
    """加载并探索数据"""
    train_data = pd.read_csv(train_path)
    
    print("数据集形状:", train_data.shape)
    print("\n前几行数据:")
    print(train_data.head())
    print("\n数据基本信息:")
    print(train_data.info())
    print("\n数据统计描述:")
    print(train_data.describe())
    print("\n缺失值统计:")
    print(train_data.isnull().sum())
    
    return train_data

def visualize_data(train_data):
    """数据可视化分析"""
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(15, 10))
    
    # 1. 目标变量分布
    plt.subplot(2, 2, 1)
    sns.histplot(train_data['MEDV'], kde=True)
    plt.title('房价(MEDV)分布')
    plt.xlabel('房价')
    plt.ylabel('频数')
    
    # 2. 相关性热力图
    plt.subplot(2, 2, 2)
    correlation_matrix = train_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title('特征相关性热力图')
    
    # 3. 重要特征与目标变量的散点图
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=train_data, x='RM', y='MEDV')
    plt.title('房间数(RM)与房价关系')
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=train_data, x='LSTAT', y='MEDV')
    plt.title('低收入人群占比(LSTAT)与房价关系')
    
    plt.tight_layout()
    plt.show()
    
    correlations = correlation_matrix['MEDV'].sort_values(ascending=False)
    print("\n与房价相关性最强的特征:")
    print(correlations)

def feature_engineering(data):
    """特征工程"""
    df = data.copy()
    
    # 创建新特征
    df['RM_sq'] = df['RM'] ** 2
    df['LSTAT_log'] = np.log1p(df['LSTAT'])
    df['DIS_log'] = np.log1p(df['DIS'])
    df['RM_LSTAT'] = df['RM'] * df['LSTAT']
    
    # 标准化数值特征
    scaler = StandardScaler()
    numeric_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 
                       'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 
                       'RM_sq', 'LSTAT_log', 'DIS_log', 'RM_LSTAT']
    
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df, scaler

def train_and_evaluate_models(X, y):
    """训练和评估多个模型"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        print(f'\n{name}:')
        print(f'RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})')
        
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f'R2: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})')

def optimize_xgboost(X, y):
    """优化XGBoost模型"""
    xgb_model = xgb.XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'min_child_weight': [1, 3]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    print("\n最佳参数:", grid_search.best_params_)
    print("最佳RMSE:", np.sqrt(-grid_search.best_score_))
    
    return grid_search.best_estimator_

def predict_and_save(model, test_data, test_processed, save_path):
    """预测并保存结果"""
    X_test = test_processed.drop(['MEDV', 'CHAS', 'idx'], axis=1)
    predictions = model.predict(X_test)
    test_data['MEDV'] = predictions
    test_data = test_data.sort_values('idx')
    test_data.to_csv(save_path, index=False)
    print(f"\n预测完成，结果已保存到 {save_path}")

def predict_train_data(model, train_data, save_path):
    """预测训练集数据"""
    original_medv = train_data['MEDV'].copy()
    train_data_for_pred = train_data.copy()
    train_processed_for_pred, _ = feature_engineering(train_data_for_pred)
    X_train_pred = train_processed_for_pred.drop(['MEDV', 'CHAS'], axis=1)
    train_predictions = model.predict(X_train_pred)
    train_data_for_pred['MEDV'] = train_predictions
    train_data_for_pred.to_csv(save_path, index=False)
    train_rmse = np.sqrt(mean_squared_error(original_medv, train_predictions))
    print(f"\n训练集上的RMSE: {train_rmse:.4f}")

def main():
    # 加载数据
    train_data = load_and_explore_data('train.csv')
    
    # 数据可视化
    visualize_data(train_data)
    
    # 特征工程
    train_processed, scaler = feature_engineering(train_data)
    
    # 准备训练数据
    X = train_processed.drop(['MEDV', 'CHAS'], axis=1)
    y = train_processed['MEDV']
    
    # 训练和评估模型
    train_and_evaluate_models(X, y)
    
    # 优化XGBoost
    best_model = optimize_xgboost(X, y)
    
    # 预测测试集
    test_data = pd.read_csv('test.csv')
    test_processed, _ = feature_engineering(test_data)
    predict_and_save(best_model, test_data, test_processed, 'test_predictions.csv')
    
    # 预测训练集
    predict_train_data(best_model, train_data, 'train_predictions.csv')

if __name__ == "__main__":
    main()