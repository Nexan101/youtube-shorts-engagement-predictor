import joblib

# Save the best model to a file
joblib.dump(rf, 'model.pkl')
print('Model saved!')

# Also save the feature list
# (API needs to know the right column order)
joblib.dump(FEATURES, 'features.pkl')

# Test loading it back
loaded_model = joblib.load('model.pkl')
test_pred = loaded_model.predict(X_test[:1])
print(f'Test prediction: {test_pred[0]:.4f}')
