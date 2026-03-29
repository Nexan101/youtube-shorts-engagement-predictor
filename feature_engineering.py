import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('data/youtube_shorts_processed.csv')

FEATURES = ['duration_secs', 'day_of_week', 'hour_posted', 'days_live', 'title_length']

x = df[FEATURES]
y = df['like_ratio']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(x_train, y_train)

for name, model in [('RF', rf), ('GB', gb)]:
    preds = model.predict(x_test)
    print(f'{name} R2 Score: {r2_score(y_test, preds):.3f} MAE={mean_absolute_error(y_test, preds):.3f}')

os.makedirs('generated', exist_ok=True)
joblib.dump(rf, 'generated/model.pkl')
joblib.dump(FEATURES, 'generated/features.pkl')
print('Model and features saved to generated/')

loaded_model = joblib.load('generated/model.pkl')
test_pred = loaded_model.predict(x_test[:1])
print(f'Test prediction: {test_pred[0]:.4f}')
