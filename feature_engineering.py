from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES = ['duration_secs', 'day_of_week', 'hour_posted', 'days_live', 'title_length']

x = df[FEATURES]
y = df['like_ratio']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(x_train,y_train)

for name, model in [('RF', rf), ('GB', gb)]:
    preds = model.predict(x_test)
    print(f'{name} R2 Score: {r2_score(y_test, preds):.3f} MAE={mean_absolute_error(y_test,preds):.3f}')