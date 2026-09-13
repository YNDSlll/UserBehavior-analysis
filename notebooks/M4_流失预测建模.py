"""
M4 流失预测 · 建模脚本
口径说明：
  - 研究人群：观察期内有购买行为的用户（rfm 表，6689 人）
  - 流失定义：观察期最后 3 天（2017-12-01 ~ 12-03）无任何购买行为 → churn=1
  - 防数据穿越：特征仅用 2017-11-25 ~ 11-30（前 6 天）构造
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).resolve().parents[1]

# ---------- 1. 数据准备 ----------
df = pd.read_csv(f'{BASE}/data/processed/UserBehavior_clean.csv',
                 dtype={'user_id': str, 'item_id': str, 'category_id': str})
df['date'] = pd.to_datetime(df['date'])

train_part = df[df['date'] < '2017-12-01']    # 前 6 天：特征原料
label_part = df[df['date'] >= '2017-12-01']   # 后 3 天：标签区

rfm = pd.read_csv(f'{BASE}/data/processed/UserBehavior_rfm.csv',
                  dtype={'user_id': str})
users = rfm[['user_id', 'cluster', 'cluster_name']].copy()

buyers_last3 = set(label_part.loc[label_part['behavior_type'] == 'buy', 'user_id'])
users['churn'] = (~users['user_id'].isin(buyers_last3)).astype(int)

# ---------- 2. 特征表（仅前 6 天） ----------
feat = train_part.groupby('user_id').agg(
    pv_count=('behavior_type', lambda s: (s == 'pv').sum()),
    cart_count=('behavior_type', lambda s: (s == 'cart').sum()),
    fav_count=('behavior_type', lambda s: (s == 'fav').sum()),
    buy_count=('behavior_type', lambda s: (s == 'buy').sum()),
    active_days=('date', 'nunique'),
    cat_breadth=('category_id', 'nunique'),
)
last_act = train_part.groupby('user_id')['date'].max()
feat['recency'] = (pd.Timestamp('2017-11-30') - last_act).dt.days

data = users.merge(feat, on='user_id', how='left')
n_no_hist = int(data['pv_count'].isnull().sum())
data = data.dropna().copy()
data.to_csv(f'{BASE}/data/processed/UserBehavior_features.csv', index=False)

# ---------- 3. 方法学检查 ----------
fm_corr = rfm['F'].corr(rfm['M'])
cluster_churn = users.groupby('cluster_name')['churn'].agg(['mean', 'count']).round(3)

# ---------- 4. 建模 ----------
X = data.drop(columns=['user_id', 'churn', 'cluster', 'cluster_name'])
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

lr = Pipeline([('scaler', StandardScaler()),
               ('clf', LogisticRegression(max_iter=1000))])
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                            random_state=42, n_jobs=-1)

results = {}
for name, model in [('逻辑回归', lr), ('随机森林', rf)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'report': classification_report(y_test, y_pred, digits=3),
        'auc': roc_auc_score(y_test, y_prob),
        'cm': confusion_matrix(y_test, y_pred),
    }

# 逻辑回归标准化系数（方向解读用）
lr_clf = lr.named_steps['clf']
lr_coef = pd.Series(lr_clf.coef_[0], index=X.columns).sort_values()

# 随机森林特征重要性
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

# ---------- 5. 特征重要性图 ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
imp.plot(kind='barh', ax=ax, color='#185FA5')
ax.set_title('特征重要性（随机森林）')
ax.set_xlabel('importance')
fig.tight_layout()
fig.savefig(f'{BASE}/output/charts/M4_特征重要性.png', dpi=150)

# ---------- 6. 结果落盘 ----------
r_lr, r_rf = results['逻辑回归'], results['随机森林']
lines = []
lines.append('# M4 流失预测 · 建模结果\n')
lines.append('> 研究人群：观察期内有购买行为的用户（6689 人）。流失 = 最后 3 天（12-01~12-03）无购买。'
             '特征仅用前 6 天（11-25~11-30）构造，防止数据穿越。训练/测试 = 75/25（分层抽样，random_state=42）。\n')
lines.append('## 一、数据准备\n')
lines.append(f'- 无前 6 天历史的用户：{n_no_hist} 人（已剔除），建模样本 {len(data)} 人')
lines.append(f'- 标签分布：churn=1 流失 {(y == 1).sum()} 人（{(y == 1).mean():.1%}），'
             f'churn=0 留存 {(y == 0).sum()} 人（{(y == 0).mean():.1%}），接近均衡，无需重采样\n')
lines.append('## 二、M3 聚类 × M4 标签交叉验证\n')
lines.append('各分层用户的实际流失率：\n')
lines.append(cluster_churn.to_string())
lines.append('\n\n（预期：早期沉睡用户流失率显著最高，与 M3 的 R=6.6 画像互相印证）\n')
lines.append('## 三、F/M 共线性检查\n')
lines.append(f'F（购买次数）与 M（购买商品种类数）的皮尔逊相关系数 = **{fm_corr:.3f}**。'
             '两者高度相关，说明"买得多"与"买得广"在本数据中几乎是同一信息，'
             'RFM 的价值维度实际由一个综合购买强度驱动。此为方法论局限，已在报告中说明。\n')
lines.append('## 四、模型对比\n')
lines.append('### 逻辑回归\n```')
lines.append(r_lr['report'])
lines.append(f"AUC = {r_lr['auc']:.3f}\n```")
lines.append('### 随机森林\n```')
lines.append(r_rf['report'])
lines.append(f"AUC = {r_rf['auc']:.3f}\n```")
lines.append('### 混淆矩阵（随机森林，行=真实 0/1，列=预测 0/1）\n```')
lines.append(str(r_rf['cm']))
lines.append('```\n')
lines.append('## 五、可解释性\n')
lines.append('### 随机森林特征重要性\n```')
lines.append(imp.sort_values(ascending=False).round(4).to_string())
lines.append('```\n')
lines.append('### 逻辑回归标准化系数（正=提高流失风险，负=降低）\n```')
lines.append(lr_coef.round(3).to_string())
lines.append('```\n')

with open(f'{BASE}/output/reports/M4_建模结果.md', 'w') as f:
    f.write('\n'.join(lines))

# ---------- 7. 控制台摘要 ----------
print('无历史用户:', n_no_hist, '| 建模样本:', len(data))
print('\n流失比例:', f"{(y == 1).mean():.1%}")
print('\n各分层流失率:\n', cluster_churn)
print('\nF/M 相关系数:', round(fm_corr, 3))
for name, r in results.items():
    print(f'\n==== {name} ==== AUC={r["auc"]:.3f}')
    print(r['report'])
print('\n特征重要性:\n', imp.sort_values(ascending=False).round(4))
print('\nLR 系数:\n', lr_coef.round(3))
print('\n已保存: data/processed/UserBehavior_features.csv')
print('已保存: output/charts/M4_特征重要性.png')
print('已保存: output/reports/M4_建模结果.md')
