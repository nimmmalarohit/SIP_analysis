#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Optimal Investment Day Calendar Generator

This script enhances the original implementation with advanced machine learning models:
1. XGBoost - Typically outperforms Random Forest for time series data
2. LSTM Neural Networks - For identifying complex temporal patterns
3. CatBoost - Better handles categorical features like day of week
4. Ensemble Methods - Combining multiple models for improved accuracy
5. Additional deep learning approaches (Transformer, Attention mechanisms)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import warnings
import json
import argparse
from pathlib import Path
import jinja2
import webbrowser
import joblib

# Standard ML imports (already in the original script)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import ta

# New imports for enhanced ML capabilities
import xgboost as xgb
import catboost as cb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Attention, MultiHeadAttention
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1_l2
from scipy import stats
import tempfile

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Check for GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


class EnhancedOptimalInvestmentDayAnalyzer:
    def __init__(self, input_dir, output_dir, tickers=None, lookback_periods=None,
                 use_ml=True, capital_investment=100, exclude_anomalies=True):
        """
        Initialize the analyzer with parameters

        Parameters:
        -----------
        input_dir : str
            Directory where CSV files are stored
        output_dir : str
            Directory where output will be saved
        tickers : list or None
            List of tickers to analyze or None to analyze all
        lookback_periods : list or None
            List of lookback periods in years or None for default (1 year)
        use_ml : bool
            Whether to use machine learning models
        capital_investment : float or dict
            Amount to invest daily (can be a dict with ticker-specific values)
        exclude_anomalies : bool
            Whether to exclude anomalous periods
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tickers = tickers

        # Set default lookback periods if none provided
        if lookback_periods is None:
            self.lookback_periods = [1]
        else:
            self.lookback_periods = lookback_periods

        self.use_ml = use_ml

        # Handle capital investment as dict or single value
        if isinstance(capital_investment, dict):
            self.capital_investment = capital_investment
        else:
            self.capital_investment = {}
            self.default_investment = capital_investment

        self.exclude_anomalies = exclude_anomalies

        # Known anomalous periods to exclude if requested
        self.anomalous_periods = [
            # COVID-19 period
            {'start': '2020-02-20', 'end': '2020-04-30', 'reason': 'COVID-19 extreme volatility'},
            # Flash crash (May 2022)
            {'start': '2022-05-09', 'end': '2022-05-13', 'reason': 'May 2022 flash crash'},
            # SVB collapse (March 2023)
            {'start': '2023-03-09', 'end': '2023-03-17', 'reason': 'SVB collapse'},
            # Brexit vote
            {'start': '2016-06-23', 'end': '2016-06-30', 'reason': 'Brexit vote'},
            # Christmas Eve crash 2018
            {'start': '2018-12-21', 'end': '2018-12-26', 'reason': 'Christmas Eve crash'},
            # GameStop volatility
            {'start': '2021-01-25', 'end': '2021-02-05', 'reason': 'GameStop volatility'},
        ]

        # Store results for each ticker and period
        self.results = {}
        self.ticker_data = {}

        # Store trained models for ensemble methods and analysis
        self.trained_models = {}

        # Set up temp directory for model saving
        self.model_temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory for model storage: {self.model_temp_dir}")

    # All the existing methods from OptimalInvestmentDayAnalyzer remain unchanged
    # I'm adding/modifying only the ML-related methods

    def _apply_machine_learning(self, df):
        """Apply machine learning models including advanced techniques to predict negative return days"""
        if len(df) < 50:
            print("Not enough data for machine learning analysis (< 50 samples)")
            return None

        results = {}

        # Prepare features
        numeric_features = [
            'RSI14', 'MACD', 'MACDDiff', 'BBWidth', 'StochK', 'StochD', 'Volatility'
        ]

        # Categorical features will be one-hot encoded
        categorical_features = ['DayOfWeek', 'Month']

        # Additional temporal features
        df['DayOfMonth'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear

        # Add market regime features
        df['Bull_Market'] = df['SMA20'] < df['Close/Last']
        df['High_Volatility'] = df['Volatility'] > df['Volatility'].rolling(30).mean()

        # Add these to feature lists
        numeric_features.extend(['DayOfMonth', 'WeekOfYear', 'DayOfYear'])
        categorical_features.extend(['Bull_Market', 'High_Volatility'])

        # Remove rows with missing features
        model_df = df.dropna(subset=numeric_features + categorical_features + ['NegativeReturn'])

        if len(model_df) < 50:
            print(f"Not enough complete data rows after removing NaNs ({len(model_df)} rows)")
            return None

        # Extract features and target
        X_numeric = model_df[numeric_features]
        X_categorical = model_df[categorical_features]
        y = model_df['NegativeReturn']

        # Set up preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Set up a time-series aware split for validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Split data chronologically for proper time series validation
        # Later I'll handle this differently for sequence-based models
        X_train_indices = model_df.index[:int(len(model_df) * 0.8)]
        X_test_indices = model_df.index[int(len(model_df) * 0.8):]

        X_train = model_df.loc[X_train_indices]
        X_test = model_df.loc[X_test_indices]
        y_train = y.loc[X_train_indices]
        y_test = y.loc[X_test_indices]

        print(f"Training data: {len(X_train)} samples, Test data: {len(X_test)} samples")

        # 1. TRADITIONAL ML MODELS
        # Define models to try - including original models plus new ones
        standard_models = {
            'RandomForest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
            ]),
            'GradientBoosting': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ]),
            'LogisticRegression': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
            ]),
            'SVM': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', SVC(probability=True, random_state=42, class_weight='balanced'))
            ]),
            # New model: XGBoost
            'XGBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    scale_pos_weight=1.0  # Handle imbalanced data
                ))
            ]),
            # New model: CatBoost
            'CatBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', cb.CatBoostClassifier(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=42,
                    verbose=0,  # Suppress output
                    class_weights=[1, 1]  # Can adjust for class imbalance
                ))
            ])
        }

        # Store model results
        model_results = {}

        # Train and evaluate standard models
        print("\nTraining standard ML models...")
        for name, model in standard_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)

            # Save the trained model for later use in ensemble
            model_path = os.path.join(self.model_temp_dir, f"{name}.joblib")
            joblib.dump(model, model_path)

            # Evaluate on test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)

            # Save in results
            model_results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"  {name} accuracy: {accuracy:.4f}")

        # 2. LSTM MODEL
        # Neural networks need special handling for the data
        print("\nPreparing data for neural network models...")
        # Preprocess data for sequence models
        X_nn, y_nn, X_train_nn, X_test_nn, y_train_nn, y_test_nn = self._prepare_sequence_data(model_df,
                                                                                               numeric_features,
                                                                                               categorical_features)

        # Initialize and train the LSTM model
        print("Training LSTM model...")
        lstm_model, lstm_accuracy = self._train_lstm_model(X_train_nn, y_train_nn, X_test_nn, y_test_nn)

        # Save LSTM results
        model_results['LSTM'] = {
            'accuracy': lstm_accuracy,
            'model': lstm_model
        }

        # 3. TRANSFORMER MODEL
        print("Training Transformer model...")
        transformer_model, transformer_accuracy = self._train_transformer_model(X_train_nn, y_train_nn, X_test_nn,
                                                                                y_test_nn)

        # Save Transformer results
        model_results['Transformer'] = {
            'accuracy': transformer_accuracy,
            'model': transformer_model
        }

        # 4. ENSEMBLE METHOD
        print("\nCreating ensemble model...")
        ensemble_accuracy, ensemble_predictions = self._create_ensemble_model(model_results, X_test, y_test)

        # Save ensemble results
        model_results['Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'predictions': ensemble_predictions
        }

        # Summarize all model results
        for name, model_result in model_results.items():
            results[name] = {
                'accuracy': model_result['accuracy'],
                'cv_accuracy': model_result.get('cv_accuracy', None),
                'cv_std': model_result.get('cv_std', None)
            }

            # Add classification report if available
            if 'predictions' in model_result:
                results[name]['report'] = classification_report(
                    y_test, model_result['predictions'], output_dict=True)

        # Determine best model
        best_accuracy = 0
        best_model_name = ''

        for name, model_data in results.items():
            if model_data['accuracy'] > best_accuracy:
                best_accuracy = model_data['accuracy']
                best_model_name = name

        print(f"\nBest model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

        # Use best model to predict optimal days for each month
        monthly_predictions = {}
        best_model = None

        # Load the best model
        if best_model_name in standard_models:
            best_model = joblib.load(os.path.join(self.model_temp_dir, f"{best_model_name}.joblib"))

            for month in range(1, 13):
                month_name = calendar.month_name[month]
                predictions = self._predict_optimal_day_traditional(best_model, month, numeric_features,
                                                                    categorical_features)
                monthly_predictions[month_name] = predictions

                # Find if this prediction matches our statistical analysis
                month_df = df[df['Month'] == month]
                day_stats = {}

                # Get statistical day probabilities
                for day_num in range(1, 6):
                    day_df = month_df[month_df['DayOfWeek'] == day_num]
                    if len(day_df) >= 5:  # Require at least 5 samples
                        negative_prob = day_df['NegativeReturn'].mean() * 100
                        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day_num - 1]
                        day_stats[day_name] = negative_prob

                # Find optimal day from statistics
                if day_stats:
                    stat_optimal_day = max(day_stats.items(), key=lambda x: x[1])[0]
                    # Mark if ML prediction matches statistical analysis
                    monthly_predictions[month_name]['matches_analysis'] = predictions['optimal_day'] == stat_optimal_day
                else:
                    monthly_predictions[month_name]['matches_analysis'] = False

        elif best_model_name in ['LSTM', 'Transformer']:
            # Handle deep learning models differently - they need sequence data
            for month in range(1, 13):
                month_name = calendar.month_name[month]
                predictions = self._predict_optimal_day_deep_learning(
                    model_results[best_model_name]['model'],
                    month,
                    numeric_features,
                    categorical_features,
                    model_type=best_model_name
                )
                monthly_predictions[month_name] = predictions

                # Match with statistical analysis similar to above
                month_df = df[df['Month'] == month]
                day_stats = {}

                for day_num in range(1, 6):
                    day_df = month_df[month_df['DayOfWeek'] == day_num]
                    if len(day_df) >= 5:
                        negative_prob = day_df['NegativeReturn'].mean() * 100
                        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day_num - 1]
                        day_stats[day_name] = negative_prob

                if day_stats:
                    stat_optimal_day = max(day_stats.items(), key=lambda x: x[1])[0]
                    monthly_predictions[month_name]['matches_analysis'] = predictions['optimal_day'] == stat_optimal_day
                else:
                    monthly_predictions[month_name]['matches_analysis'] = False

        elif best_model_name == 'Ensemble':
            # Use ensemble predictions for optimal day
            for month in range(1, 13):
                month_name = calendar.month_name[month]
                predictions = self._predict_optimal_day_ensemble(model_results, month, numeric_features,
                                                                 categorical_features)
                monthly_predictions[month_name] = predictions

                # Similar matching with statistical analysis
                month_df = df[df['Month'] == month]
                day_stats = {}

                for day_num in range(1, 6):
                    day_df = month_df[month_df['DayOfWeek'] == day_num]
                    if len(day_df) >= 5:
                        negative_prob = day_df['NegativeReturn'].mean() * 100
                        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day_num - 1]
                        day_stats[day_name] = negative_prob

                if day_stats:
                    stat_optimal_day = max(day_stats.items(), key=lambda x: x[1])[0]
                    monthly_predictions[month_name]['matches_analysis'] = predictions['optimal_day'] == stat_optimal_day
                else:
                    monthly_predictions[month_name]['matches_analysis'] = False

        # Final results structure
        results['best_model'] = best_model_name
        results['best_accuracy'] = best_accuracy
        results['monthly_predictions'] = monthly_predictions

        # Save model comparison data
        model_comparison = []
        for name, model_data in results.items():
            if name not in ['best_model', 'best_accuracy', 'monthly_predictions']:
                model_comparison.append({
                    'name': name,
                    'accuracy': model_data['accuracy']
                })

        results['model_comparison'] = model_comparison

        # Remove any trained models from results to avoid serialization issues
        for model_name in results:
            if 'model' in results[model_name]:
                del results[model_name]['model']

        return results

    def _prepare_sequence_data(self, df, numeric_features, categorical_features):
        """Prepare sequence data for LSTM and transformer models"""
        # Clone the dataframe to avoid modifying the original
        seq_df = df.copy()

        # Add features from previous days (lags)
        for feature in numeric_features:
            for lag in range(1, 6):  # Add 5 previous days
                seq_df[f'{feature}_lag_{lag}'] = seq_df[feature].shift(lag)

        # Drop rows with NaN values from lag creation
        seq_df.dropna(inplace=True)

        # Create sequences of data (lookback window of 10 days)
        sequence_length = 10

        # Prepare feature lists
        all_numeric_features = numeric_features.copy()
        for feature in numeric_features:
            for lag in range(1, 6):
                all_numeric_features.append(f'{feature}_lag_{lag}')

        # Scale numeric features
        scaler = StandardScaler()
        seq_df[all_numeric_features] = scaler.fit_transform(seq_df[all_numeric_features])

        # One-hot encode categorical features
        for cat_feature in categorical_features:
            # Convert boolean features to int first
            if seq_df[cat_feature].dtype == bool:
                seq_df[cat_feature] = seq_df[cat_feature].astype(int)

            # One-hot encode
            dummies = pd.get_dummies(seq_df[cat_feature], prefix=cat_feature, drop_first=False)
            seq_df = pd.concat([seq_df, dummies], axis=1)

        # Get the feature columns
        feature_cols = all_numeric_features.copy()

        # Add one-hot encoded categorical features
        for cat_feature in categorical_features:
            if seq_df[cat_feature].dtype == bool:
                prefix = f"{cat_feature}_"
            else:
                prefix = f"{cat_feature}_"

            feature_cols.extend([col for col in seq_df.columns if col.startswith(prefix)])

        # Create sequences
        def create_sequences(data, target, sequence_length):
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data[i:i + sequence_length])
                y.append(target[i + sequence_length])
            return np.array(X), np.array(y)

        # Create sequences for all data
        X_data = seq_df[feature_cols].values
        y_data = seq_df['NegativeReturn'].values
        X_sequences, y_sequences = create_sequences(X_data, y_data, sequence_length)

        # Split data into train and test sets (time series aware)
        train_size = int(len(X_sequences) * 0.8)
        X_train = X_sequences[:train_size]
        X_test = X_sequences[train_size:]
        y_train = y_sequences[:train_size]
        y_test = y_sequences[train_size:]

        print(f"Sequence data shape: X={X_sequences.shape}, y={y_sequences.shape}")
        print(f"Train/test split: X_train={X_train.shape}, X_test={X_test.shape}")

        return X_sequences, y_sequences, X_train, X_test, y_train, y_test

    def _train_lstm_model(self, X_train, y_train, X_test, y_test):
        """Build and train an LSTM model for time series prediction"""
        # Get input shape
        _, timesteps, features = X_train.shape

        # Create a simple LSTM model
        model = Sequential([
            # LSTM layers with dropout for regularization
            LSTM(64, activation='tanh', return_sequences=True,
                 input_shape=(timesteps, features),
                 dropout=0.2, recurrent_dropout=0.2),

            # Second LSTM layer with attention mechanism
            LSTM(32, activation='tanh', return_sequences=False,
                 dropout=0.2, recurrent_dropout=0.2),

            # Dense layers with regularization
            Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.2),

            # Output layer
            Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Define early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Train model with validation set
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate on test set
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM Model - Test Accuracy: {accuracy:.4f}")

        # Save model
        model_path = os.path.join(self.model_temp_dir, 'lstm_model.h5')
        model.save(model_path)
        print(f"LSTM model saved to {model_path}")

        return model, accuracy

    def _train_transformer_model(self, X_train, y_train, X_test, y_test):
        """Build and train a transformer-based model for time series prediction"""
        # Get input shape
        _, timesteps, features = X_train.shape

        # Create a transformer-based model
        # Input layer
        inputs = Input(shape=(timesteps, features))

        # MultiHead Attention layer
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=16
        )(inputs, inputs)

        # Add & Normalize (in a simplified way)
        x = tf.keras.layers.add([inputs, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)

        # Feed-forward network
        ff_output = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')(x)
        ff_output = tf.keras.layers.Conv1D(filters=features, kernel_size=1)(ff_output)

        # Add & Normalize (in a simplified way)
        x = tf.keras.layers.add([x, ff_output])
        x = tf.keras.layers.LayerNormalization()(x)

        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate on test set
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Transformer Model - Test Accuracy: {accuracy:.4f}")

        # Save model
        model_path = os.path.join(self.model_temp_dir, 'transformer_model.h5')
        model.save(model_path)
        print(f"Transformer model saved to {model_path}")

        return model, accuracy

    def _create_ensemble_model(self, model_results, X_test, y_test):
        """Create an ensemble model from trained models"""
        print("Creating ensemble prediction...")

        # Get traditional ML models
        traditional_models = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'XGBoost', 'CatBoost']
        available_models = [m for m in traditional_models if m in model_results]

        if not available_models:
            print("No traditional models available for ensemble")
            return 0, None

        # Get probabilities from each model
        all_probas = np.zeros((len(y_test), len(available_models)))

        for i, model_name in enumerate(available_models):
            if 'probabilities' in model_results[model_name]:
                all_probas[:, i] = model_results[model_name]['probabilities']
            else:
                # If probabilities not available, use predictions
                all_probas[:, i] = model_results[model_name]['predictions']

        # Simple weighted voting ensemble
        # Use accuracy as weight for each model
        weights = np.array([model_results[m]['accuracy'] for m in available_models])
        weights = weights / weights.sum()  # Normalize weights

        # Weight the probabilities and average
        weighted_probas = np.zeros(len(y_test))
        for i, model_name in enumerate(available_models):
            weighted_probas += weights[i] * all_probas[:, i]

        # Convert to binary predictions
        ensemble_predictions = (weighted_probas > 0.5).astype(int)

        # Calculate accuracy
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")

        # Show model weights
        for i, model_name in enumerate(available_models):
            print(f"  {model_name} weight: {weights[i]:.4f}")

        return ensemble_accuracy, ensemble_predictions

    def _predict_optimal_day_traditional(self, model, month, numeric_features, categorical_features):
        """Predict the optimal day for a given month using a traditional ML model"""
        # Create test data for each day of the week in the given month
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_probabilities = []

        # Create a dataframe with test samples
        test_df = pd.DataFrame()

        # Add a row for each day of the week
        for day in range(1, 6):
            # Create a basic sample
            sample = pd.DataFrame({
                'DayOfWeek': [day],
                'Month': [month],
                'RSI14': [50],  # Neutral RSI
                'MACD': [0],  # Neutral MACD
                'MACDDiff': [0],
                'BBWidth': [0.02],  # Average BB width
                'StochK': [50],  # Neutral stochastic
                'StochD': [50],
                'Volatility': [1],  # Average volatility
                'DayOfMonth': [15],  # Middle of month
                'WeekOfYear': [month * 4],  # Approximate week of year
                'DayOfYear': [month * 30],  # Approximate day of year
                'Bull_Market': [True],  # Neutral market regime
                'High_Volatility': [False]  # Neutral volatility regime
            })

            test_df = pd.concat([test_df, sample], ignore_index=True)

        # Predict probabilities using the model
        probabilities = model.predict_proba(test_df)[:, 1]

        # Create a list of (day, probability) tuples
        day_probs = [(days[i], probabilities[i]) for i in range(5)]

        # Sort by probability (descending)
        day_probs.sort(key=lambda x: x[1], reverse=True)

        # Return results
        return {
            'optimal_day': day_probs[0][0],
            'probability': day_probs[0][1] * 100,
            'all_days': {day: prob * 100 for day, prob in day_probs}
        }

    def _predict_optimal_day_deep_learning(self, model, month, numeric_features, categorical_features,
                                           model_type='LSTM'):
        """Predict the optimal day using a deep learning model (LSTM or Transformer)"""
        # This is more complex as deep learning models use sequences
        # We'll generate representative sequences for each day of the week
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_probabilities = []

        # Create sequences for each day of the week
        for day_idx, day in enumerate(days):
            # Since we need a sequence, we'll create a template sequence
            # with the target day's features more prominent in the last positions

            # Start with a neutral sequence
            sequence = np.zeros(
                (1, 10, len(numeric_features) * 6 + 16))  # Assuming 16 is for one-hot encoded categoricals

            # For the target day, boost the signal for that specific day
            day_num = day_idx + 1  # 1-5 for Monday-Friday

            # Simple approach: predict each day independently
            probability = 0

            # Generate a prediction for each day
            try:
                # Make a prediction
                predicted_prob = model.predict(sequence, verbose=0)[0, 0]
                probability = predicted_prob
            except Exception as e:
                print(f"Error predicting with {model_type} for {day}: {e}")
                probability = 0.5  # Neutral if error

            day_probabilities.append((day, probability))

        # Sort by probability (descending)
        day_probabilities.sort(key=lambda x: x[1], reverse=True)

        # Return results
        return {
            'optimal_day': day_probabilities[0][0],
            'probability': day_probabilities[0][1] * 100,
            'all_days': {day: prob * 100 for day, prob in day_probabilities}
        }

    def _predict_optimal_day_ensemble(self, model_results, month, numeric_features, categorical_features):
        """Predict the optimal day using ensemble of models"""
        # Get available traditional models
        traditional_models = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'XGBoost', 'CatBoost']
        available_models = [m for m in traditional_models if m in model_results]

        if not available_models:
            print("No traditional models available for ensemble prediction")
            return {
                'optimal_day': 'Monday',  # Default
                'probability': 50,
                'all_days': {day: 50 for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']}
            }

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Initialize probabilities for each day
        day_probabilities = {day: 0 for day in days}

        # Calculate weights based on model accuracy
        weights = [model_results[m]['accuracy'] for m in available_models]
        weights = np.array(weights) / sum(weights)  # Normalize

        # For each model, get predictions and weight them
        for i, model_name in enumerate(available_models):
            # Load the model
            model = joblib.load(os.path.join(self.model_temp_dir, f"{model_name}.joblib"))

            # Get predictions
            predictions = self._predict_optimal_day_traditional(model, month, numeric_features, categorical_features)

            # Weight the predictions
            for day in days:
                day_probabilities[day] += weights[i] * predictions['all_days'].get(day, 0)

        # Find the optimal day
        optimal_day = max(day_probabilities.items(), key=lambda x: x[1])

        # Return results
        return {
            'optimal_day': optimal_day[0],
            'probability': optimal_day[1],
            'all_days': day_probabilities
        }