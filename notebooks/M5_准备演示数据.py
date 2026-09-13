"""M5 演示数据准备：把 App 需要的所有聚合结果预先算好，Streamlit 只做轻量读取。"""
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
OUT = f'{BASE}/output/app_data'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f'{BASE}/data/processed/UserBehavior_clean.csv',
                 dtype={'user_id': str, 'item_id': str, 'category_id': str})
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])
rfm = pd.read_csv(f'{BASE}/data/processed/UserBehavior_rfm.csv', dtype={'user_id': str})
data = pd.read_csv(f'{BASE}/data/processed/UserBehavior_features.csv', dtype={'user_id': str})

# 1. 核心指标
total_users = rfm['user_id'].nunique()
all_users = df['user_id'].nunique()
kpis = pd.DataFrame([
    {'指标': '总行为数（清洗后）', '数值': f'{len(df):,}'},
    {'指标': '用户总数', '数值': f'{all_users:,}'},
    {'指标': '人均行为数（9 天）', '数值': f'{len(df) / all_users:.1f}'},
    {'指标': '行为转化率（购/览）', '数值': f"{(df['behavior_type'] == 'buy').sum() / (df['behavior_type'] == 'pv').sum():.1%}"},
    {'指标': '购买用户渗透率', '数值': f"{total_users / all_users:.1%}"},
    {'指标': '流失预测 AUC（随机森林）', '数值': '0.736'},
])
kpis.to_csv(f'{OUT}/kpis.csv', index=False)

# 2. 日趋势
daily = df.groupby('date').agg(行为数=('user_id', 'size'), 活跃用户数=('user_id', 'nunique'))
daily.index = daily.index.strftime('%m-%d')
daily.to_csv(f'{OUT}/daily.csv')

# 3. 小时分布
hourly = df.groupby(df['datetime'].dt.hour).size().rename('行为数')
hourly.index.name = '小时'
hourly.to_csv(f'{OUT}/hourly.csv')

# 4. 转化漏斗（用户口径）
funnel = pd.DataFrame({
    '环节': ['浏览用户', '加购用户', '购买用户'],
    '人数': [
        df.loc[df['behavior_type'] == 'pv', 'user_id'].nunique(),
        df.loc[df['behavior_type'] == 'cart', 'user_id'].nunique(),
        df.loc[df['behavior_type'] == 'buy', 'user_id'].nunique(),
    ],
})
funnel.to_csv(f'{OUT}/funnel.csv', index=False)

# 5. 用户分层画像
churn = data.set_index('user_id')['churn']
profile = rfm.copy()
profile['churn'] = profile['user_id'].map(churn)
cluster_profile = profile.groupby('cluster_name').agg(
    人数=('user_id', 'count'), R均值=('R', 'mean'), F均值=('F', 'mean'),
    M均值=('M', 'mean'), 实际流失率=('churn', 'mean')).round(2)
cluster_profile.to_csv(f'{OUT}/cluster_profile.csv')

# 6. 模型对比 + 特征重要性（复现 M4，保证 App 数字与分析结果同源）
X = data.drop(columns=['user_id', 'churn', 'cluster', 'cluster_name'])
y = data['churn']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

rows = []
for name, model in [('逻辑回归', lr), ('随机森林', rf)]:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    rows.append({
        '模型': name,
        'AUC': round(roc_auc_score(y_te, y_prob), 3),
        '流失召回率': round(recall_score(y_te, y_pred), 3),
        '流失F1': round(f1_score(y_te, y_pred), 3),
    })
pd.DataFrame(rows).to_csv(f'{OUT}/model_metrics.csv', index=False)
rf_auc = rows[1]['AUC']
assert abs(rf_auc - 0.736) < 0.005, f'AUC 复现偏差过大: {rf_auc}'

pd.Series(rf.feature_importances_, index=X.columns, name='importance').to_csv(f'{OUT}/feature_importance.csv')

print('演示数据准备完成 →', OUT)
print(os.listdir(OUT))
