# LOSO-Cross-Validation for ANFIS Model on Ankle Angle Prediction (All 4 Cases)
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from anfisarchitecture import ANFIS, fis_parameters

# Load complete dataset (must contain SubjectID, EMG, GRF, Knee_Angle, ANKLE)
data = pd.read_csv('LSTM_1kmhr_LOSO_Cross_validation.csv')

# Define 4 cases and their respective input features
cases = {
    'Case I': ['EMG', 'GRF'],
    'Case II': ['EMG', 'Knee_Angle'],
    'Case III': ['GRF', 'Knee_Angle'],
    'Case IV': ['EMG', 'GRF', 'Knee_Angle']
}

subjects = data['SubjectID'].unique()

summary = []

for case_name, features in cases.items():
    print(f"\n--- Running {case_name} ---")
    rmse_list = []
    r2_list = []

    for test_subject in subjects:
        # Split data
        train_data = data[data['SubjectID'] != test_subject].copy()
        test_data = data[data['SubjectID'] == test_subject].copy()

        # Extract inputs and outputs
        X_train = train_data[features].values
        y_train = train_data['ANKLE'].values
        X_test = test_data[features].values
        y_test = test_data['ANKLE'].values

        # Normalize input and output
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

        # Truncate to nearest multiple of 32 (to avoid shape mismatch)
        def truncate_to_batch(x, y, batch_size=32):
            length = (x.shape[0] // batch_size) * batch_size
            return x[:length], y[:length]

        X_train_scaled, y_train_scaled = truncate_to_batch(X_train_scaled, y_train_scaled)
        X_test_scaled, y_test_scaled = truncate_to_batch(X_test_scaled, y_test_scaled)

        # Define ANFIS model parameters
        param = fis_parameters(
            n_input=len(features),
            n_memb=4,
            batch_size=32,
            memb_func='gaussian',
            optimizer='sgd',
            loss=tf.keras.losses.MeanSquaredError(),
            n_epochs=400
        )

        # Initialize and compile ANFIS model
        anfis = ANFIS(n_input=param.n_input, n_memb=param.n_memb,
                      batch_size=param.batch_size, memb_func=param.memb_func)
        anfis.model.compile(optimizer=param.optimizer, loss=param.loss)

        # Fit the model
        anfis.fit(X_train_scaled, y_train_scaled,
                  epochs=param.n_epochs,
                  validation_data=(X_test_scaled, y_test_scaled), verbose=0)

        # Predict
        y_pred_scaled = anfis.model.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test[:len(y_pred)], y_pred))
        r2 = r2_score(y_test[:len(y_pred)], y_pred)
        rmse_list.append(rmse)
        r2_list.append(r2)
        print(f"Subject {test_subject}: RMSE = {rmse:.3f}, R2 = {r2:.3f}")

    # Summarize case results
    summary.append({
        'Case': case_name,
        'Mean RMSE': np.mean(rmse_list),
        'Std RMSE': np.std(rmse_list),
        'Mean R2': np.mean(r2_list)
    })

# Save and display final results
summary_df = pd.DataFrame(summary)
print("\n--- LOSO ANFIS Evaluation Summary ---")
print(summary_df)
summary_df.to_csv('LOSO_ANFIS_AnkleAngle_Summary.csv', index=False)
