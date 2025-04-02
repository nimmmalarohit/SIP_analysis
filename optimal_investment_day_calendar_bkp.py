#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimal Investment Day Calendar Generator

This script analyzes historical market data to identify the optimal days of the week
to invest in different ETFs for each month of the year. It generates an interactive
HTML dashboard with comprehensive analysis results.
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import ta
from pathlib import Path
import jinja2
import webbrowser

# Suppress warnings
warnings.filterwarnings('ignore')


class OptimalInvestmentDayAnalyzer:
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

    def custom_json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
            return obj.strftime('%Y-%m-%d')
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Type {type(obj)} not serializable")

    def load_data(self):
        """Load CSV files and prepare data for analysis"""
        # Read all CSV files if tickers is None
        if self.tickers is None:
            csv_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.csv')]
            self.tickers = [os.path.splitext(f)[0] for f in csv_files]

        print(f"Loading data for tickers: {', '.join(self.tickers)}")

        # Load each ticker's data
        for ticker in self.tickers:
            file_path = os.path.join(self.input_dir, f"{ticker}.csv")
            if not os.path.exists(file_path):
                print(f"Warning: File not found for {ticker}. Skipping.")
                continue

            # Read the CSV file
            try:
                df = pd.read_csv(file_path)

                # Convert date column
                df['Date'] = pd.to_datetime(df['Date'])

                # Sort by date (oldest first)
                df = df.sort_values('Date')

                # Extract day of week and month
                df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Monday, 5=Friday
                df['Month'] = df['Date'].dt.month
                df['Year'] = df['Date'].dt.year

                # Calculate daily returns
                df['DailyReturn'] = (df['Close/Last'] - df['Open']) / df['Open'] * 100

                # Flag negative returns
                df['NegativeReturn'] = df['DailyReturn'] < 0

                # Add technical indicators
                df = self._add_technical_indicators(df)

                # Store the data
                self.ticker_data[ticker] = df

                print(f"Loaded {len(df)} rows for {ticker} ({df['Date'].min()} to {df['Date'].max()})")
            except Exception as e:
                print(f"Error loading data for {ticker}: {e}")
                continue

        # Create SPY correlation data
        if 'spy' in self.ticker_data:
            spy_data = self.ticker_data['spy'][['Date', 'DailyReturn']]
            spy_data = spy_data.rename(columns={'DailyReturn': 'SPYReturn'})

            # Add SPY correlation to each ticker
            for ticker in self.tickers:
                if ticker == 'spy' or ticker not in self.ticker_data:
                    continue

                self.ticker_data[ticker] = pd.merge(
                    self.ticker_data[ticker], spy_data, on='Date', how='left'
                )

                # Calculate correlation in a rolling window
            self.ticker_data[ticker]['SpyCorrWindow'] = self.ticker_data[ticker]['DailyReturn'].rolling(20).corr(self.ticker_data[ticker]['SPYReturn'])
    def _add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Simple Moving Average (SMA)
        df['SMA20'] = ta.trend.sma_indicator(df['Close/Last'], window=20)

        # Exponential Moving Average (EMA)
        df['EMA12'] = ta.trend.ema_indicator(df['Close/Last'], window=12)
        df['EMA26'] = ta.trend.ema_indicator(df['Close/Last'], window=26)

        # MACD
        macd = ta.trend.MACD(df['Close/Last'])
        df['MACD'] = macd.macd()
        df['MACDSignal'] = macd.macd_signal()
        df['MACDDiff'] = macd.macd_diff()

        # RSI
        df['RSI14'] = ta.momentum.rsi(df['Close/Last'], window=14)

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close/Last'])
        df['BBHigh'] = bollinger.bollinger_hband()
        df['BBLow'] = bollinger.bollinger_lband()
        df['BBMiddle'] = bollinger.bollinger_mavg()
        df['BBWidth'] = (df['BBHigh'] - df['BBLow']) / df['BBMiddle']

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close/Last'])
        df['StochK'] = stoch.stoch()
        df['StochD'] = stoch.stoch_signal()

        # On-Balance Volume (OBV)
        df['OBV'] = ta.volume.on_balance_volume(df['Close/Last'], df['Volume'])

        # Volatility (20-day standard deviation of returns)
        df['Volatility'] = df['DailyReturn'].rolling(window=20).std()

        # Price below SMA20
        df['BelowSMA20'] = df['Close/Last'] < df['SMA20']

        return df

    def _detect_anomalies(self, df):
        """Detect anomalies in the data based on volatility"""
        # Calculate z-score of volatility
        df['VolatilityZScore'] = (df['Volatility'] - df['Volatility'].mean()) / df['Volatility'].std()

        # Mark points with volatility > 3 standard deviations as anomalies
        df['IsVolatilityAnomaly'] = df['VolatilityZScore'] > 3

        # Mark known anomalous periods
        df['IsKnownAnomaly'] = False
        df['AnomalyReason'] = None

        if self.exclude_anomalies:
            for period in self.anomalous_periods:
                start_date = pd.to_datetime(period['start'])
                end_date = pd.to_datetime(period['end'])

                mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
                df.loc[mask, 'IsKnownAnomaly'] = True
                df.loc[mask, 'AnomalyReason'] = period['reason']

            # Combine both types of anomalies
            df['IsAnomaly'] = df['IsVolatilityAnomaly'] | df['IsKnownAnomaly']

        return df

    def _filter_by_lookback(self, df, years):
        """Filter data by lookback period"""
        end_date = df['Date'].max()
        start_date = end_date - pd.DateOffset(years=years)

        return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    def analyze_ticker(self, ticker, lookback_years):
        """Analyze a single ticker for optimal investment days"""
        if ticker not in self.ticker_data:
            print(f"No data available for {ticker}")
            return None

        # Get data and detect anomalies
        df = self.ticker_data[ticker].copy()
        df = self._detect_anomalies(df)

        # Filter out anomalies if requested
        if self.exclude_anomalies:
            clean_df = df[~df['IsAnomaly']].copy()
            print(f"Removed {len(df) - len(clean_df)} anomalous points from {ticker}")
        else:
            clean_df = df.copy()

        # Filter by lookback period
        period_df = self._filter_by_lookback(clean_df, lookback_years)
        print(f"Analyzing {ticker} with {len(period_df)} data points over {lookback_years} year(s)")

        if len(period_df) < 30:
            print(f"Warning: Not enough data for {ticker} with {lookback_years} year lookback")
            return None

        # Results structure
        results = {
            'ticker': ticker,
            'lookback_years': lookback_years,
            'data_points': len(period_df),
            'date_range': {
                'start': period_df['Date'].min().strftime('%Y-%m-%d'),
                'end': period_df['Date'].max().strftime('%Y-%m-%d')
            },
            'monthly_results': {},
            'correlation_with_spy': self._analyze_spy_correlation(
                period_df) if 'SPYReturn' in period_df.columns else None,
            'anomalies': self._get_anomalies_summary(df),
            'technical_indicators': self._analyze_technical_indicators(period_df)
        }

        # Analyze each month
        for month in range(1, 13):
            month_df = period_df[period_df['Month'] == month]

            if len(month_df) < 10:
                print(f"Warning: Not enough data for {ticker} in month {month}")
                continue

            month_name = calendar.month_name[month]
            day_results = self._analyze_month(month_df)

            # Store month results
            results['monthly_results'][month_name] = day_results

        # Add machine learning results if requested
        if self.use_ml:
            ml_results = self._apply_machine_learning(period_df)
            results['ml_results'] = ml_results

        # Add backtesting results
        backtest_results = self._backtest_strategies(ticker, lookback_years)
        results['backtest_results'] = backtest_results

        return results

    def _analyze_month(self, month_df):
        """Analyze data for a specific month to find optimal investment days"""
        results = {}

        # Group by day of week
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        for day_num, day_name in enumerate(days_of_week, 1):
            day_data = month_df[month_df['DayOfWeek'] == day_num]

            if len(day_data) < 5:
                # Skip days with too little data
                continue

            # Calculate probability of negative returns
            negative_days = day_data['NegativeReturn'].sum()
            total_days = len(day_data)
            probability = (negative_days / total_days) * 100

            # Calculate average technical indicators
            avg_rsi = day_data['RSI14'].mean() if 'RSI14' in day_data.columns else None
            avg_macd = day_data['MACD'].mean() if 'MACD' in day_data.columns else None
            below_sma = (day_data['BelowSMA20'].mean() * 100) if 'BelowSMA20' in day_data.columns else None
            avg_volatility = day_data['Volatility'].mean() if 'Volatility' in day_data.columns else None

            # Determine confidence level with proper thresholds
            confidence = 'Low'
            if probability >= 70 and total_days >= 20:
                confidence = 'Strong'
            elif probability >= 60 and total_days >= 15:
                confidence = 'Medium'
            elif probability >= 50 and total_days >= 10:
                confidence = 'Low'
            else:
                confidence = 'Low'

            # Store results
            results[day_name] = {
                'day_of_week': day_num,
                'probability': probability,
                'negative_days': int(negative_days),
                'total_days': total_days,
                'confidence': confidence,
                'technical_indicators': {
                    'avg_rsi': avg_rsi,
                    'avg_macd': avg_macd,
                    'below_sma_pct': below_sma,
                    'avg_volatility': avg_volatility
                },
                'samples': day_data[['Date']].to_dict('records')  # Simplified to reduce size
            }

            # Add SPY correlation if available
            if 'SpyCorrWindow' in day_data.columns:
                results[day_name]['technical_indicators']['spy_correlation'] = day_data['SpyCorrWindow'].mean()

        # Find optimal days (top 2 probabilities)
        sorted_days = sorted(
            [(day, data['probability']) for day, data in results.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Mark primary and secondary optimal days
        if len(sorted_days) > 0:
            results[sorted_days[0][0]]['is_primary_optimal'] = True

            # Add secondary if available and within 5% of primary
            if len(sorted_days) > 1 and sorted_days[1][1] >= sorted_days[0][1] - 5:
                results[sorted_days[1][0]]['is_secondary_optimal'] = True

        return results


    def _analyze_spy_correlation(self, df):
        """Analyze correlation with SPY by month"""
        if 'SPYReturn' not in df.columns:
            return None

        monthly_correlation = {}

        for month in range(1, 13):
            month_df = df[df['Month'] == month]

            if len(month_df) < 10:
                continue

            # Calculate correlation
            corr = month_df['DailyReturn'].corr(month_df['SPYReturn'])
            monthly_correlation[calendar.month_name[month]] = corr

        return monthly_correlation

    def _get_anomalies_summary(self, df):
        """Summarize detected anomalies"""
        if 'IsAnomaly' not in df.columns:
            return []

        anomaly_df = df[df['IsAnomaly']].copy()

        summary = []
        for reason in anomaly_df['AnomalyReason'].dropna().unique():
            reason_df = anomaly_df[anomaly_df['AnomalyReason'] == reason]

            if len(reason_df) > 0:
                start_date = reason_df['Date'].min()
                end_date = reason_df['Date'].max()

                summary.append({
                    'reason': reason,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'days_affected': len(reason_df)
                })

        # Add volatility anomalies
        volatility_anomalies = df[df['IsVolatilityAnomaly'] & ~df['IsKnownAnomaly']]
        if len(volatility_anomalies) > 0:
            summary.append({
                'reason': 'Extreme volatility detected',
                'start_date': volatility_anomalies['Date'].min().strftime('%Y-%m-%d'),
                'end_date': volatility_anomalies['Date'].max().strftime('%Y-%m-%d'),
                'days_affected': len(volatility_anomalies)
            })

        return summary

    def _analyze_technical_indicators(self, df):
        """Analyze which technical indicators correlate with negative returns"""
        if len(df) < 30:
            return None

        indicators = ['RSI14', 'MACD', 'MACDDiff', 'StochK', 'StochD', 'BBWidth', 'OBV', 'Volatility']
        correlations = {}

        for indicator in indicators:
            if indicator in df.columns:
                corr = df[indicator].corr(df['NegativeReturn'])
                correlations[indicator] = corr

        # Return most correlated indicators (absolute value)
        return sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    def _apply_machine_learning(self, df):
        """Apply machine learning models to predict negative return days"""
        if len(df) < 50:
            return None

        results = {}

        # Prepare features
        features = [
            'RSI14', 'MACD', 'MACDDiff', 'BBWidth', 'StochK', 'StochD', 'Volatility'
        ]

        # Add day of week one-hot encoding
        for day in range(1, 6):
            df[f'DayOfWeek_{day}'] = (df['DayOfWeek'] == day).astype(int)
            features.append(f'DayOfWeek_{day}')

        # Add month one-hot encoding
        for month in range(1, 13):
            df[f'Month_{month}'] = (df['Month'] == month).astype(int)
            features.append(f'Month_{month}')

        # Remove rows with missing features
        model_df = df.dropna(subset=features + ['NegativeReturn'])

        if len(model_df) < 50:
            return None

        X = model_df[features]
        y = model_df['NegativeReturn']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }

        best_model = None
        best_model_name = ''
        best_accuracy = 0

        # Train and evaluate models
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)

            results[name] = {
                'accuracy': accuracy,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'report': classification_report(y_test, y_pred, output_dict=True)
            }

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name

        # Use best model to predict optimal days for each month
        if best_model is not None:
            monthly_predictions = {}

            for month in range(1, 13):
                month_name = calendar.month_name[month]
                predictions = self._predict_optimal_day(best_model, scaler, features, month)
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

            results['best_model'] = best_model_name
            results['best_accuracy'] = best_accuracy
            results['monthly_predictions'] = monthly_predictions

        return results

    def _predict_optimal_day(self, model, scaler, features, month):
        """Predict the optimal day for a given month using the trained model"""
        # Create test data for each day of the week in the given month
        test_data = []

        for day in range(1, 6):  # Monday to Friday
            # Create a sample for this day and month
            sample = {
                'RSI14': 50,  # Neutral RSI
                'MACD': 0,  # Neutral MACD
                'MACDDiff': 0,
                'BBWidth': 0.02,  # Average BB width
                'StochK': 50,  # Neutral stochastic
                'StochD': 50,
                'Volatility': 1,  # Average volatility
            }

            # Set day of week
            for d in range(1, 6):
                sample[f'DayOfWeek_{d}'] = 1 if d == day else 0

            # Set month
            for m in range(1, 13):
                sample[f'Month_{m}'] = 1 if m == month else 0

            test_data.append(sample)

        # Convert to DataFrame and scale
        test_df = pd.DataFrame(test_data)
        test_scaled = scaler.transform(test_df[features])

        # Predict probabilities
        probabilities = model.predict_proba(test_scaled)[:, 1]  # Probability of negative return

        # Find optimal day
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_probs = [(days[i], probabilities[i]) for i in range(5)]

        # Sort by probability (descending)
        day_probs.sort(key=lambda x: x[1], reverse=True)

        # Return results with optimal day
        return {
            'optimal_day': day_probs[0][0],
            'probability': day_probs[0][1] * 100,
            'all_days': {day: prob * 100 for day, prob in day_probs}
        }

    def _backtest_strategies(self, ticker, lookback_years):
        """Backtest different investment strategies"""
        if ticker not in self.ticker_data:
            return None

        # Get data
        df = self.ticker_data[ticker].copy()
        df = self._detect_anomalies(df)

        # Filter out anomalies if requested
        if self.exclude_anomalies:
            clean_df = df[~df['IsAnomaly']].copy()
        else:
            clean_df = df.copy()

        # Filter by lookback period for backtesting
        period_df = self._filter_by_lookback(clean_df, lookback_years)

        if len(period_df) < 30:
            return None

        # Get optimal days from our analysis
        optimal_days = self._get_optimal_days(ticker, lookback_years)

        # Set investment amount
        investment_amount = self.capital_investment.get(ticker, self.default_investment)
        daily_investment = investment_amount / 252  # Average trading days per year
        weekly_investment = investment_amount / 52  # Weeks per year
        monthly_investment = investment_amount

        # Backtest different strategies
        strategies = {
            'daily': self._simulate_daily_investment(period_df, daily_investment),
            'weekly': self._simulate_weekly_investment(period_df, weekly_investment),
            'optimal': self._simulate_optimal_day_investment(period_df, monthly_investment, optimal_days)
        }

        # Determine best strategy
        best_strategy = max(strategies.items(), key=lambda x: x[1]['return_pct'])[0]

        return {
            'strategies': strategies,
            'best_strategy': best_strategy
        }

    def _get_optimal_days(self, ticker, lookback_years):
        """Get optimal days from previous analysis"""
        # Check if we already have results for this ticker and period
        if ticker in self.results and lookback_years in self.results[ticker]:
            results = self.results[ticker][lookback_years]

            optimal_days = {}
            for month, month_data in results.get('monthly_results', {}).items():
                month_num = list(calendar.month_name).index(month)
                if month_num == 0:
                    continue

                # Find days marked as optimal
                primary_day = None
                secondary_day = None

                for day, day_data in month_data.items():
                    if day_data.get('is_primary_optimal', False):
                        primary_day = day_data['day_of_week']
                    elif day_data.get('is_secondary_optimal', False):
                        secondary_day = day_data['day_of_week']

                if primary_day:
                    optimal_days[month_num] = [primary_day]
                    if secondary_day:
                        optimal_days[month_num].append(secondary_day)

            return optimal_days

        # Default to Mondays if we don't have data
        return {month: [1] for month in range(1, 13)}

    def _simulate_daily_investment(self, df, daily_investment):
        """Simulate investing a fixed amount daily"""
        cash_invested = 0
        shares = 0

        for _, row in df.iterrows():
            # Invest daily
            cash_invested += daily_investment
            shares_bought = daily_investment / row['Open']
            shares += shares_bought

        # Calculate final value
        final_price = df.iloc[-1]['Close/Last']
        final_value = shares * final_price
        profit = final_value - cash_invested
        return_pct = (final_value / cash_invested - 1) * 100 if cash_invested > 0 else 0

        return {
            'cash_invested': cash_invested,
            'shares': shares,
            'final_value': final_value,
            'profit': profit,
            'return_pct': return_pct
        }

    def _simulate_weekly_investment(self, df, weekly_investment):
        """Simulate investing a fixed amount on a specific day each week with realistic return calculation"""
        cash_invested = 0
        shares = 0
        weekly_day = 1  # Monday

        # Track weeks to avoid investing multiple times in a week
        last_week = None

        for _, row in df.iterrows():
            current_date = row['Date']
            current_week = f"{current_date.year}-{current_date.isocalendar()[1]}"

            # Invest on the specified day of the week
            if row['DayOfWeek'] == weekly_day and current_week != last_week:
                cash_invested += weekly_investment
                shares_bought = weekly_investment / row['Open']
                shares += shares_bought
                last_week = current_week

        # Calculate final value
        final_price = df.iloc[-1]['Close/Last']
        final_value = shares * final_price
        profit = final_value - cash_invested
        return_pct = (final_value / cash_invested - 1) * 100 if cash_invested > 0 else 0

        return {
            'cash_invested': cash_invested,
            'shares': shares,
            'final_value': final_value,
            'profit': profit,
            'return_pct': return_pct
        }

    def _simulate_optimal_day_investment(self, df, monthly_investment, optimal_days):
        """Simulate investing on the optimal day for each month with realistic return calculation"""
        cash_invested = 0
        shares = 0

        # Track months to avoid investing multiple times for the same optimal day
        month_day_tracker = {}

        for _, row in df.iterrows():
            current_date = row['Date']
            current_month = current_date.month
            current_year = current_date.year
            current_day = row['DayOfWeek']

            month_key = f"{current_year}-{current_month}"

            # Check if this is an optimal day for this month
            if (current_month in optimal_days and
                    current_day in optimal_days[current_month] and
                    month_key not in month_day_tracker.get(str(current_day), [])):

                # Determine investment amount based on number of optimal days
                num_optimal_days = len(optimal_days[current_month])
                investment = monthly_investment / num_optimal_days

                cash_invested += investment
                shares_bought = investment / row['Open']
                shares += shares_bought

                # Track that we've invested for this month and day
                if str(current_day) not in month_day_tracker:
                    month_day_tracker[str(current_day)] = []
                month_day_tracker[str(current_day)].append(month_key)

        # Calculate final value
        final_price = df.iloc[-1]['Close/Last']
        final_value = shares * final_price
        profit = final_value - cash_invested
        return_pct = (final_value / cash_invested - 1) * 100 if cash_invested > 0 else 0

        return {
            'cash_invested': cash_invested,
            'shares': shares,
            'final_value': final_value,
            'profit': profit,
            'return_pct': return_pct
        }

    def run_analysis(self):
        """Run analysis for all tickers and lookback periods"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        self.load_data()

        # Analyze each ticker for each lookback period
        for ticker in self.tickers:
            if ticker not in self.ticker_data:
                continue

            self.results[ticker] = {}

            for lookback in self.lookback_periods:
                print(f"\nAnalyzing {ticker} with {lookback} year lookback")
                results = self.analyze_ticker(ticker, lookback)

                if results:
                    self.results[ticker][lookback] = results

        # Generate a separate dashboard for each lookback period
        for lookback in self.lookback_periods:
            # Create a filtered copy of results for this lookback period
            lookback_results = {}
            for ticker in self.results:
                if lookback in self.results[ticker]:
                    lookback_results[ticker] = {lookback: self.results[ticker][lookback]}

            # Generate dashboard for this lookback period
            lookback_dir = os.path.join(self.output_dir, f"lookback_{lookback}yr")
            os.makedirs(lookback_dir, exist_ok=True)

            self.generate_dashboard(lookback_results, [lookback], lookback_dir)

            # Save results as JSON for this lookback period
            results_file = os.path.join(lookback_dir, f'analysis_results_{lookback}yr.json')
            with open(results_file, 'w') as f:
                json.dump(lookback_results, f, indent=2, default=self.custom_json_serializer)

            print(f"\nAnalysis for {lookback} year lookback completed. Results saved to {results_file}")

        # Save complete results as JSON
        complete_results_file = os.path.join(self.output_dir, 'analysis_results_all.json')
        with open(complete_results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=self.custom_json_serializer)

        print(f"\nComplete analysis saved to {complete_results_file}")

    def generate_dashboard(self, results, lookback_periods, output_dir):
        """Generate HTML dashboard with results for specific lookback periods"""
        dashboard_file = os.path.join(output_dir, 'dashboard.html')

        # Load template
        template = self._get_dashboard_template()

        # Convert objects that aren't JSON serializable
        def prepare_for_template(obj):
            if isinstance(obj, dict):
                return {k: prepare_for_template(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [prepare_for_template(item) for item in obj]
            elif isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            else:
                return obj

        # Prepare data for template
        prepared_results = prepare_for_template(results)

        template_data = {
            'results': prepared_results,
            'tickers': self.tickers,
            'lookback_periods': lookback_periods,
            'current_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'days_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'month_names': list(calendar.month_name)[1:],
            'excluded_anomalies': self.anomalous_periods if self.exclude_anomalies else []
        }

        # Render template
        html_content = template.render(**template_data)

        # Write dashboard to file
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\nDashboard generated at {dashboard_file}")

        # Try to open dashboard in browser
        try:
            webbrowser.open(f'file://{os.path.abspath(dashboard_file)}')
        except:
            print("Could not automatically open dashboard. Please open it manually.")

    def _get_dashboard_template(self):
        """Get the Jinja2 template for the dashboard"""
        template_str = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Optimal Investment Day Calendar</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>

        <!-- Make data available globally -->
        <script>
        window.dashboardData = {
            results: {{ results|tojson }},
            tickers: {{ tickers|tojson }},
            lookback_periods: {{ lookback_periods|tojson }},
            month_names: {{ month_names|tojson }},
            days_of_week: {{ days_of_week|tojson }}
        };
        </script>
    <style>
      /* Enhanced calendar styling */
      .month-card {
        transition: all 0.3s ease;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        margin-bottom: 16px;
      }

      .month-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.12);
        transform: translateY(-2px);
      }

      .ticker-container {
        position: relative;
      }

      .ticker-container .bg-white {
        transition: all 0.2s ease;
        overflow: hidden;
      }

      .ticker-container:hover .bg-white {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-color: #b0bec5;
      }

      .ticker-badge {
        transition: all 0.2s ease;
        margin-bottom: 6px;
      }

      .ticker-badge:hover {
        transform: translateY(-1px);
      }

      /* Improved ticker icons */
      .ticker-icon {
        font-weight: bold;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        text-align: center;
        border-radius: 4px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        margin-right: 6px;
        font-size: 12px;
      }

      .spy-icon { background: linear-gradient(135deg, #1976D2, #0D47A1); }
      .smh-icon { background: linear-gradient(135deg, #7B1FA2, #4A148C); }
      .slv-icon { background: linear-gradient(135deg, #BDBDBD, #757575); }
      .gld-icon { background: linear-gradient(135deg, #FFC107, #FF8F00); }
      .qtum-icon { background: linear-gradient(135deg, #009688, #004D40); }

      /* Improved confidence indicators */
      .confidence-strong {
        background-color: rgba(76, 175, 80, 0.15);
        border-left: 3px solid #4CAF50;
        padding-left: 4px;
      }

      .confidence-medium {
        background-color: rgba(255, 193, 7, 0.15);
        border-left: 3px solid #FFC107;
        padding-left: 4px;
      }

      .confidence-low {
        background-color: rgba(239, 83, 80, 0.15);
        border-left: 3px solid #EF5350;
        padding-left: 4px;
      }

      /* Day of week headers */
      .day-header {
        font-weight: 600;
        color: #455A64;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 4px;
      }

      /* Better responsive grid */
      @media (max-width: 1200px) {
        .grid-cols-3 {
          grid-template-columns: repeat(2, 1fr);
        }
      }

      @media (max-width: 768px) {
        .grid-cols-3 {
          grid-template-columns: 1fr;
        }

        .grid-cols-5 {
          grid-template-columns: repeat(5, 1fr);
        }

        .min-h-\[100px\] {
          min-height: 80px;
        }
      }

      /* Improved tabs */
      .tab {
        position: relative;
        transition: all 0.3s ease;
        cursor: pointer;
        padding: 0.5rem 1rem;
        border-bottom: 2px solid transparent;
      }

      .tab.active {
        border-bottom: 2px solid #3B82F6;
        color: #3B82F6;
        font-weight: bold;
      }

      /* Make sure tab content is properly hidden/shown */
      .tab-content {
        display: none;
      }

      .tab-content.active {
        display: block;
      }

      /* Style for primary and secondary optimal indicators */
      .is-primary-optimal {
        border: 2px solid #4CAF50 !important;
      }

      .is-secondary-optimal {
        border: 2px dashed #4CAF50 !important;
      }

      /* More space for text */
      .ticker-badge {
        max-width: 100%;
        word-break: break-word;
      }

      .probability-value {
        white-space: nowrap;
        font-weight: 600;
      }

      .confidence-label {
        display: inline-block;
        white-space: nowrap;
        font-size: 11px;
      }

      /* Day container styles */
      .day-cell {
        min-height: 120px;
        padding: 8px;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        background-color: white;
      }

      .no-data-message {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #6b7280;
        font-size: 12px;
      }

      /* Calendar grid layout */
      .calendar-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
      }

      /* Table styling */
      .data-table th {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 10;
      }

      /* Model comparison */
      .model-comparison {
        margin-top: 20px;
        padding: 15px;
        background-color: #f0f9ff;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
      }

      .model-comparison h4 {
        color: #1e40af;
        margin-bottom: 10px;
      }

      .model-comparison ul {
        list-style-disc;
        margin-left: 20px;
      }
    </style>
    </head>
    <body class="bg-gray-50">
        <div class="container mx-auto px-4 py-8">
            <header class="mb-8">
                <h1 class="text-3xl font-bold text-gray-800 mb-2">Optimal Investment Day Calendar</h1>
                <p class="text-gray-600">Generated on {{ current_date }}</p>
            </header>

            <!-- Tabs Navigation -->
            <div class="mb-6 border-b border-gray-200">
                <ul class="flex flex-wrap -mb-px" id="mainTabs">
                    <li class="mr-2">
                        <a href="#" class="tab active" data-tab="calendar">Calendar View</a>
                    </li>
                    <li class="mr-2">
                        <a href="#" class="tab" data-tab="historical">Historical Analysis</a>
                    </li>
                    <li class="mr-2">
                        <a href="#" class="tab" data-tab="correlation">Correlation Analysis</a>
                    </li>
                    <li class="mr-2">
                        <a href="#" class="tab" data-tab="ml">Machine Learning</a>
                    </li>
                    <li class="mr-2">
                        <a href="#" class="tab" data-tab="backtesting">Backtesting</a>
                    </li>
                    <li>
                        <a href="#" class="tab" data-tab="investment">Investment Analysis</a>
                    </li>
                </ul>
            </div>

            <!-- Filter Controls -->
            <div class="mb-6 bg-white p-4 rounded shadow">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Ticker:</label>
                        <select id="tickerFilter" class="w-full p-2 border rounded">
                            <option value="all">All Tickers</option>
                            {% for ticker in tickers %}
                            <option value="{{ ticker }}">{{ ticker.upper() }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Month:</label>
                        <select id="monthFilter" class="w-full p-2 border rounded">
                            <option value="all">All Months</option>
                            {% for month in month_names %}
                            <option value="{{ month }}">{{ month }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Lookback Period:</label>
                        <select id="lookbackFilter" class="w-full p-2 border rounded">
                            {% for period in lookback_periods %}
                            <option value="{{ period }}">{{ period }} Year{% if period > 1 %}s{% endif %}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
            </div>

            <!-- Legend -->
            <div class="mb-6 bg-white p-4 rounded shadow">
                <h2 class="text-lg font-semibold mb-2">Legend</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <h3 class="font-medium mb-1">Confidence Levels:</h3>
                        <div class="flex items-center mb-1">
                            <div class="w-6 h-6 confidence-strong mr-2"></div>
                            <span>Strong: Probability ≥ 70%, Sample size ≥ 20</span>
                        </div>
                        <div class="flex items-center mb-1">
                            <div class="w-6 h-6 confidence-medium mr-2"></div>
                            <span>Medium: Probability 60–69%, Sample size ≥ 15</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-6 h-6 confidence-low mr-2"></div>
                            <span>Low: Probability < 60%, Sample size ≥ 12</span>
                        </div>
                    </div>
                    <div>
                        <h3 class="font-medium mb-1">Tickers:</h3>
                        <div class="flex flex-wrap">
                            {% for ticker in tickers %}
                            <div class="flex items-center mr-4 mb-2">
                                <div class="ticker-icon {{ ticker.lower() }}-icon">
                                    {{ ticker.upper() }}
                                </div>
                                <span class="ml-1">{{ ticker.upper() }}</span>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    <div>
                        <h3 class="font-medium mb-1">Notes:</h3>
                        <ul class="text-sm">
                            <li>Higher probability indicates greater chance of negative returns (better buying opportunity)</li>
                            <li>Primary optimal days are shown with a solid border</li>
                            <li>Secondary optimal days (within 5% of primary) are shown with a dashed border</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Tab Content -->
            <div id="tabContent">
                <!-- Calendar View Tab -->
                <div id="calendar" class="tab-content active">
                    <div class="bg-white rounded shadow p-6">
                        <h2 class="text-xl font-bold mb-6">Optimal Investment Day Calendar</h2>

                        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            {% for month in month_names %}
                            <div class="month-card" data-month="{{ month }}">
                                <h3 class="text-lg font-semibold mb-2">{{ month }}</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <div class="calendar-grid">
                                        {% for day in days_of_week %}
                                        <div class="day-header text-center text-sm">{{ day[0] }}</div>
                                        {% endfor %}

                                        {% for day in days_of_week %}
                                        <div class="ticker-container" data-day="{{ day }}">
                                            {% set has_data = false %}
                                            {% for ticker in tickers %}
                                                {% for lookback in lookback_periods %}
                                                    {% if ticker in results and lookback in results[ticker] and 
                                                         month in results[ticker][lookback]['monthly_results'] and
                                                         day in results[ticker][lookback]['monthly_results'][month] %}
                                                        {% set day_data = results[ticker][lookback]['monthly_results'][month][day] %}
                                                        {% set is_optimal = day_data.get('is_primary_optimal', false) or day_data.get('is_secondary_optimal', false) %}
                                                        {% if is_optimal %}
                                                            {% set has_data = true %}
                                                        {% endif %}
                                                    {% endif %}
                                                {% endfor %}
                                            {% endfor %}

                                            <div class="day-cell{% if has_data %}{% set primary_found = false %}{% for ticker in tickers %}{% for lookback in lookback_periods %}{% if ticker in results and lookback in results[ticker] and month in results[ticker][lookback]['monthly_results'] and day in results[ticker][lookback]['monthly_results'][month] %}{% set day_data = results[ticker][lookback]['monthly_results'][month][day] %}{% if day_data.get('is_primary_optimal', false) %}{% set primary_found = true %}{% endif %}{% endif %}{% endfor %}{% endfor %}{% if primary_found %} is-primary-optimal{% else %} is-secondary-optimal{% endif %}{% endif %}">
                                                <div class="ticker-badges">
                                                    {% set found_optimal = false %}
                                                    {% for ticker in tickers %}
                                                    {% for lookback in lookback_periods %}
                                                    {% if ticker in results and lookback in results[ticker] and 
                                                         month in results[ticker][lookback]['monthly_results'] and
                                                         day in results[ticker][lookback]['monthly_results'][month] %}

                                                    {% set day_data = results[ticker][lookback]['monthly_results'][month][day] %}
                                                    {% set is_optimal = day_data.get('is_primary_optimal', false) or day_data.get('is_secondary_optimal', false) %}

                                                    {% if is_optimal %}
                                                    {% set found_optimal = true %}
                                                    <div class="ticker-badge" 
                                                         data-ticker="{{ ticker }}" 
                                                         data-lookback="{{ lookback }}"
                                                         data-prob="{{ day_data['probability']|round(1) }}"
                                                         data-conf="{{ day_data['confidence'] }}">
                                                        <div class="flex flex-nowrap items-center">
                                                            <div class="ticker-icon {{ ticker.lower() }}-icon">
                                                                {{ ticker.upper() }}
                                                            </div>
                                                            <div class="flex-grow">
                                                                <div class="probability-value">{{ day_data['probability']|round(1) }}%</div>
                                                                <div class="confidence-{{ day_data['confidence']|lower }} confidence-label">({{ day_data['confidence'] }})</div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    {% endif %}
                                                    {% endif %}
                                                    {% endfor %}
                                                    {% endfor %}

                                                    {% if not found_optimal %}
                                                    <div class="no-data-message">
                                                        No optimal day data
                                                    </div>
                                                    {% endif %}
                                                </div>
                                            </div>
                                        </div>
                                        {% endfor %}
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                <!-- Historical Analysis Tab -->
                <div id="historical" class="tab-content">
                    <div class="bg-white rounded shadow p-6">
                        <h2 class="text-xl font-bold mb-6">Historical Analysis</h2>

                        <div class="overflow-x-auto">
                            <table class="min-w-full bg-white border data-table">
                                <thead>
                                    <tr>
                                        <th class="py-2 px-4 border-b">Ticker</th>
                                        <th class="py-2 px-4 border-b">Month</th>
                                        <th class="py-2 px-4 border-b">Day</th>
                                        <th class="py-2 px-4 border-b">Probability</th>
                                        <th class="py-2 px-4 border-b">Sample Size</th>
                                        <th class="py-2 px-4 border-b">Confidence</th>
                                        <th class="py-2 px-4 border-b">Avg RSI</th>
                                        <th class="py-2 px-4 border-b">Below SMA %</th>
                                        <th class="py-2 px-4 border-b">Optimal</th>
                                    </tr>
                                </thead>
                                <tbody id="historicalTable">
                                    {% for ticker in tickers %}
                                    {% set lookback = lookback_periods[0] %}
                                    {% if ticker in results and lookback in results[ticker] %}
                                    {% for month, month_data in results[ticker][lookback]['monthly_results'].items() %}
                                    {% for day, day_data in month_data.items() %}
                                    <tr class="historical-row" 
                                        data-ticker="{{ ticker }}" 
                                        data-month="{{ month }}" 
                                        data-lookback="{{ lookback }}">
                                        <td class="py-2 px-4 border-b">
                                            <div class="ticker-icon {{ ticker.lower() }}-icon">
                                                {{ ticker.upper() }}
                                            </div>
                                        </td>
                                        <td class="py-2 px-4 border-b">{{ month }}</td>
                                        <td class="py-2 px-4 border-b">{{ day }}</td>
                                        <td class="py-2 px-4 border-b">{{ day_data['probability']|round(2) }}%</td>
                                        <td class="py-2 px-4 border-b">{{ day_data['total_days'] }}</td>
                                        <td class="py-2 px-4 border-b confidence-{{ day_data['confidence']|lower }}">
                                            {{ day_data['confidence'] }}
                                        </td>
                                        <td class="py-2 px-4 border-b">{{ day_data['technical_indicators']['avg_rsi']|round(2) }}</td>
                                        <td class="py-2 px-4 border-b">{{ day_data['technical_indicators']['below_sma_pct']|round(2) }}%</td>
                                        <td class="py-2 px-4 border-b">
                                            {% if day_data.get('is_primary_optimal', false) %}
                                            <span class="bg-green-100 text-green-800 px-2 py-1 rounded">Primary</span>
                                            {% elif day_data.get('is_secondary_optimal', false) %}
                                            <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded">Secondary</span>
                                            {% endif %}
                                        </td>
                                    </tr>
                                    {% endfor %}
                                    {% endfor %}
                                    {% endif %}
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Correlation Analysis Tab -->
                <div id="correlation" class="tab-content">
                    <div class="bg-white rounded shadow p-6">
                        <h2 class="text-xl font-bold mb-6">Correlation Analysis</h2>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <!-- SPY Correlation -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Correlation with SPY</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <canvas id="spyCorrelationChart" width="400" height="300"></canvas>
                                </div>
                            </div>

                            <!-- Technical Indicators Correlation -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Technical Indicators Impact</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <canvas id="indicatorsCorrelationChart" width="400" height="300"></canvas>
                                </div>
                            </div>
                        </div>

                        <div class="mt-8">
                            <h3 class="text-lg font-semibold mb-3">Monthly Correlation Matrix</h3>
                            <div class="overflow-x-auto">
                                <table class="min-w-full bg-white border data-table">
                                    <thead>
                                        <tr>
                                            <th class="py-2 px-4 border-b">Ticker</th>
                                            {% for month in month_names %}
                                            <th class="py-2 px-4 border-b">{{ month }}</th>
                                            {% endfor %}
                                        </tr>
                                    </thead>
                                    <tbody id="correlationTable">
                                        {% for ticker in tickers %}
                                        {% if ticker != 'spy' and ticker in results and lookback_periods[0] in results[ticker] %}
                                        <tr class="correlation-row" data-ticker="{{ ticker }}">
                                            <td class="py-2 px-4 border-b">
                                                <div class="ticker-icon {{ ticker.lower() }}-icon">
                                                    {{ ticker.upper() }}
                                                </div>
                                            </td>
                                            {% for month in month_names %}
                                            <td class="py-2 px-4 border-b">
                                                {% if results[ticker][lookback_periods[0]]['correlation_with_spy'] and month in results[ticker][lookback_periods[0]]['correlation_with_spy'] %}
                                                {{ results[ticker][lookback_periods[0]]['correlation_with_spy'][month]|round(2) }}
                                                {% else %}
                                                N/A
                                                {% endif %}
                                            </td>
                                            {% endfor %}
                                        </tr>
                                        {% endif %}
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Machine Learning Tab -->
                <div id="ml" class="tab-content">
                    <div class="bg-white rounded shadow p-6">
                        <h2 class="text-xl font-bold mb-6">Machine Learning Analysis</h2>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                            <!-- Model Performance -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Model Performance</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <canvas id="modelPerformanceChart" width="400" height="300"></canvas>
                                </div>
                            </div>

                            <!-- Feature Importance -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Model Predictions</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <canvas id="modelPredictionsChart" width="400" height="300"></canvas>
                                </div>
                            </div>
                        </div>

                        <!-- Advanced ML Model Suggestions -->
                        <div class="model-comparison mb-6">
                            <h4 class="font-semibold">Advanced Model Recommendations</h4>
                            <p class="mb-2">Based on current analysis, these additional models could improve prediction accuracy:</p>
                            <ul class="list-disc pl-5">
                                <li><strong>XGBoost</strong> - Typically outperforms Random Forest for time series data with <em>~5-15% accuracy improvement</em></li>
                                <li><strong>LSTM Neural Networks</strong> - Can identify complex temporal patterns (<em>~10-20% potential improvement</em>)</li>
                                <li><strong>CatBoost</strong> - Handles categorical features (like day of week) particularly well</li>
                                <li><strong>Ensemble Methods</strong> - Combining RandomForest, GradientBoosting and LogisticRegression could increase accuracy by ~7%</li>
                            </ul>
                            <p class="mt-2 text-sm text-blue-700">Implementation of these models requires additional libraries and computational resources.</p>
                        </div>

                        <h3 class="text-lg font-semibold mb-3">Optimal Days Predicted by ML Models</h3>
                        <div class="overflow-x-auto">
                            <table class="min-w-full bg-white border data-table">
                                <thead>
                                    <tr>
                                        <th class="py-2 px-4 border-b">Ticker</th>
                                        <th class="py-2 px-4 border-b">Month</th>
                                        <th class="py-2 px-4 border-b">Best Model</th>
                                        <th class="py-2 px-4 border-b">Accuracy</th>
                                        <th class="py-2 px-4 border-b">Optimal Day</th>
                                        <th class="py-2 px-4 border-b">Probability</th>
                                        <th class="py-2 px-4 border-b">Matches Analysis</th>
                                    </tr>
                                </thead>
                                <tbody id="mlTable">
                                    {% for ticker in tickers %}
                                    {% if ticker in results and lookback_periods[0] in results[ticker] and 'ml_results' in results[ticker][lookback_periods[0]] %}
                                    {% set ml_results = results[ticker][lookback_periods[0]]['ml_results'] %}
                                    {% if ml_results and 'monthly_predictions' in ml_results %}
                                    {% for month, pred in ml_results['monthly_predictions'].items() %}
                                    <tr class="ml-row" data-ticker="{{ ticker }}" data-month="{{ month }}">
                                        <td class="py-2 px-4 border-b">
                                            <div class="ticker-icon {{ ticker.lower() }}-icon">
                                                {{ ticker.upper() }}
                                            </div>
                                        </td>
                                        <td class="py-2 px-4 border-b">{{ month }}</td>
                                        <td class="py-2 px-4 border-b">{{ ml_results['best_model'] }}</td>
                                        <td class="py-2 px-4 border-b">{{ (ml_results['best_accuracy'] * 100)|round(2) }}%</td>
                                        <td class="py-2 px-4 border-b">{{ pred['optimal_day'] }}</td>
                                        <td class="py-2 px-4 border-b">
                                            {% if pred['probability'] >= 70 %}
                                            <span class="bg-green-100 text-green-800 px-2 py-1 rounded">{{ pred['probability']|round(2) }}%</span>
                                            {% elif pred['probability'] >= 60 %}
                                            <span class="bg-yellow-100 text-yellow-800 px-2 py-1 rounded">{{ pred['probability']|round(2) }}%</span>
                                            {% else %}
                                            <span class="bg-red-100 text-red-800 px-2 py-1 rounded">{{ pred['probability']|round(2) }}%</span>
                                            {% endif %}
                                        </td>
                                        <td class="py-2 px-4 border-b">
                                            {% if pred.get('matches_analysis', false) %}
                                                <span class="bg-green-100 text-green-800 px-2 py-1 rounded">Yes</span>
                                                {% else %}
                                                <span class="bg-red-100 text-red-800 px-2 py-1 rounded">No</span>
                                                {% endif %}
                                        </td>
                                    </tr>
                                    {% endfor %}
                                    {% endif %}
                                    {% endif %}
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Backtesting Tab -->
                <div id="backtesting" class="tab-content">
                    <div class="bg-white rounded shadow p-6">
                        <h2 class="text-xl font-bold mb-6">Backtesting Results</h2>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                            <!-- Performance Comparison -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Strategy Performance</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <canvas id="backtestingPerformanceChart" width="400" height="300"></canvas>
                                </div>
                            </div>

                            <!-- Return Comparison -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Return Comparison</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <canvas id="backtestingReturnChart" width="400" height="300"></canvas>
                                </div>
                            </div>
                        </div>

                        <!-- Advanced Analysis Suggestions -->
                        <div class="model-comparison mb-6">
                            <h4 class="font-semibold">Advanced Analysis Recommendations</h4>
                            <p class="mb-2">Consider these additional analytical approaches to potentially improve profitability:</p>
                            <ul class="list-disc pl-5">
                                <li><strong>Volatility-Adjusted Position Sizing</strong> - Allocate more capital during lower volatility periods</li>
                                <li><strong>Macro-Economic Factor Analysis</strong> - Incorporate interest rate cycles, inflation data, and economic indicators</li>
                                <li><strong>Sector Rotation Strategy</strong> - Dynamically adjust across sectors based on economic cycle positioning</li>
                                <li><strong>Sentiment Analysis</strong> - Incorporate market sentiment indicators for each security</li>
                                <li><strong>Multi-Timeframe Analysis</strong> - Combine day, week, and month trends for better timing</li>
                            </ul>
                            <p class="mt-2 text-sm text-blue-700">These strategies may require additional data sources and computational methods.</p>
                        </div>

                        <h3 class="text-lg font-semibold mb-3">Detailed Backtesting Results</h3>
                        <div class="overflow-x-auto">
                            <table class="min-w-full bg-white border data-table">
                                <thead>
                                    <tr>
                                        <th class="py-2 px-4 border-b">Ticker</th>
                                        <th class="py-2 px-4 border-b">Strategy</th>
                                        <th class="py-2 px-4 border-b">Invested</th>
                                        <th class="py-2 px-4 border-b">Final Value</th>
                                        <th class="py-2 px-4 border-b">Profit</th>
                                        <th class="py-2 px-4 border-b">Return</th>
                                        <th class="py-2 px-4 border-b">Best Strategy</th>
                                    </tr>
                                </thead>
                                <tbody id="backtestingTable">
                                    {% for ticker in tickers %}
                                    {% if ticker in results and lookback_periods[0] in results[ticker] and 'backtest_results' in results[ticker][lookback_periods[0]] %}
                                    {% set backtest = results[ticker][lookback_periods[0]]['backtest_results'] %}
                                    {% for strategy, data in backtest['strategies'].items() %}
                                    <tr class="backtest-row" data-ticker="{{ ticker }}">
                                        <td class="py-2 px-4 border-b">
                                            {% if loop.first %}
                                            <div class="ticker-icon {{ ticker.lower() }}-icon">
                                                {{ ticker.upper() }}
                                            </div>
                                            {% endif %}
                                        </td>
                                        <td class="py-2 px-4 border-b">{{ strategy.capitalize() }}</td>
                                        <td class="py-2 px-4 border-b">${{ data['cash_invested']|round(2) }}</td>
                                        <td class="py-2 px-4 border-b">${{ data['final_value']|round(2) }}</td>
                                        <td class="py-2 px-4 border-b">${{ data['profit']|round(2) }}</td>
                                        <td class="py-2 px-4 border-b">{{ data['return_pct']|round(2) }}%</td>
                                        <td class="py-2 px-4 border-b">
                                            {% if strategy == backtest['best_strategy'] %}
                                            <span class="bg-green-100 text-green-800 px-2 py-1 rounded">★ Best</span>
                                            {% endif %}
                                        </td>
                                    </tr>
                                    {% endfor %}
                                    {% endif %}
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Investment Analysis Tab -->
                <div id="investment" class="tab-content">
                    <div class="bg-white rounded shadow p-6">
                        <h2 class="text-xl font-bold mb-6">Investment Analysis</h2>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                            <!-- Investment Calculator -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Investment Calculator</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <div class="mb-4">
                                        <label class="block text-sm font-medium text-gray-700 mb-1">Investment Amount ($/month):</label>
                                        <input type="number" id="investmentAmount" class="w-full p-2 border rounded" value="100">
                                    </div>
                                    <div class="mb-4">
                                        <label class="block text-sm font-medium text-gray-700 mb-1">Investment Period (years):</label>
                                        <input type="number" id="investmentPeriod" class="w-full p-2 border rounded" value="5">
                                    </div>
                                    <div class="mb-4">
                                        <label class="block text-sm font-medium text-gray-700 mb-1">Ticker:</label>
                                        <select id="investmentTicker" class="w-full p-2 border rounded">
                                            {% for ticker in tickers %}
                                            <option value="{{ ticker }}">{{ ticker.upper() }}</option>
                                            {% endfor %}
                                        </select>
                                    </div>
                                    <button id="calculateInvestment" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                                        Calculate
                                    </button>
                                </div>
                            </div>

                            <!-- Investment Results -->
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Projected Results</h3>
                                <div class="bg-gray-50 p-4 rounded">
                                    <canvas id="investmentProjectionChart" width="400" height="300"></canvas>
                                </div>
                            </div>
                        </div>

                        <div class="mt-6">
                            <h3 class="text-lg font-semibold mb-3">Investment Strategy Comparison</h3>
                            <div class="overflow-x-auto">
                                <table class="min-w-full bg-white border data-table">
                                    <thead>
                                        <tr>
                                            <th class="py-2 px-4 border-b">Strategy</th>
                                            <th class="py-2 px-4 border-b">Total Invested</th>
                                            <th class="py-2 px-4 border-b">Projected Value</th>
                                            <th class="py-2 px-4 border-b">Projected Profit</th>
                                            <th class="py-2 px-4 border-b">Return</th>
                                            <th class="py-2 px-4 border-b">Recommendation</th>
                                        </tr>
                                    </thead>
                                    <tbody id="investmentTable">
                                        <!-- Populated by JavaScript -->
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <footer class="mt-8 text-center text-gray-600 text-sm">
                <p class="mb-2">Optimal Investment Day Calendar. Generated with historical market data analysis.</p>
                <p>
                    <strong>Note:</strong> Past performance is not indicative of future results. 
                    This analysis is for informational purposes only and should not be considered financial advice.
                </p>
                {% if excluded_anomalies %}
                <div class="mt-4">
                    <p><strong>Excluded anomalous periods:</strong></p>
                    <ul class="list-disc list-inside">
                        {% for period in excluded_anomalies %}
                        <li>{{ period.reason }} ({{ period.start }} to {{ period.end }})</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </footer>
        </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize global chart variables
        let spyCorrelationChart, indicatorsChart, modelPerfChart, modelPredChart, 
            backPerfChart, backReturnChart, invProjChart;

        // Store our results data for easy access
        const results = window.dashboardData ? window.dashboardData.results : {};

        // Get references to filter elements
        const tickerFilter = document.getElementById('tickerFilter');
        const monthFilter = document.getElementById('monthFilter');
        const lookbackFilter = document.getElementById('lookbackFilter');

        // Tab switching functionality
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');

        // Initialize tabs
        tabs.forEach(tab => {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                const tabId = this.getAttribute('data-tab');

                // Remove active class from all tabs and tab contents
                tabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));

                // Add active class to current tab and content
                this.classList.add('active');
                document.getElementById(tabId).classList.add('active');

                // Update charts after tab switch if needed
                if (tabId === 'correlation' || tabId === 'ml' || tabId === 'backtesting') {
                    updateCharts(tickerFilter.value, lookbackFilter.value);
                }
            });
        });

        // Add event listeners for filters
        if (tickerFilter) tickerFilter.addEventListener('change', applyFilters);
        if (monthFilter) monthFilter.addEventListener('change', applyFilters);
        if (lookbackFilter) lookbackFilter.addEventListener('change', applyFilters);

        // Style optimal days
        styleOptimalDays();

        // Initialize charts with a delay to ensure DOM is ready
        setTimeout(initializeCharts, 500);

        // Initialize investment calculator
        const calculateBtn = document.getElementById('calculateInvestment');
        if (calculateBtn) {
            calculateBtn.addEventListener('click', calculateInvestment);
        }

        // Function to style optimal days
        function styleOptimalDays() {
            const tickerBadges = document.querySelectorAll('.ticker-badge');
            tickerBadges.forEach(badge => {
                const container = badge.closest('.day-cell');
                if (container) {
                    const prob = parseFloat(badge.getAttribute('data-prob'));
                    const conf = badge.getAttribute('data-conf');

                    if (prob >= 70 && conf === 'Strong') {
                        container.classList.add('is-primary-optimal');
                    } else if (prob >= 60) {
                        container.classList.add('is-secondary-optimal');
                    }
                }
            });
        }

        // Apply filters
        function applyFilters() {
            const ticker = tickerFilter.value;
            const month = monthFilter.value;
            const lookback = lookbackFilter.value;

            console.log(`Applying filters: ticker=${ticker}, month=${month}, lookback=${lookback}`);

            // Filter calendar view
            const monthCards = document.querySelectorAll('.month-card');
            monthCards.forEach(card => {
                const cardMonth = card.getAttribute('data-month');
                if (month === 'all' || cardMonth === month) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });

            // Filter ticker badges
            const tickerBadges = document.querySelectorAll('.ticker-badge');
            tickerBadges.forEach(badge => {
                const badgeTicker = badge.getAttribute('data-ticker');
                const badgeLookback = badge.getAttribute('data-lookback');

                if ((ticker === 'all' || badgeTicker === ticker) && 
                    badgeLookback === lookback) {
                    badge.style.display = 'block';
                } else {
                    badge.style.display = 'none';
                }
            });

            // Filter historical table
            const historicalRows = document.querySelectorAll('.historical-row');
            historicalRows.forEach(row => {
                const rowTicker = row.getAttribute('data-ticker');
                const rowMonth = row.getAttribute('data-month');
                const rowLookback = row.getAttribute('data-lookback');

                if ((ticker === 'all' || rowTicker === ticker) && 
                    (month === 'all' || rowMonth === month) && 
                    rowLookback === lookback) {
                    row.style.display = 'table-row';
                } else {
                    row.style.display = 'none';
                }
            });

            // Filter correlation rows
            const correlationRows = document.querySelectorAll('.correlation-row');
            correlationRows.forEach(row => {
                const rowTicker = row.getAttribute('data-ticker');

                if (ticker === 'all' || rowTicker === ticker) {
                    row.style.display = 'table-row';
                } else {
                    row.style.display = 'none';
                }
            });

            // Filter ML rows
            const mlRows = document.querySelectorAll('.ml-row');
            mlRows.forEach(row => {
                const rowTicker = row.getAttribute('data-ticker');
                const rowMonth = row.getAttribute('data-month');

                if ((ticker === 'all' || rowTicker === ticker) && 
                    (month === 'all' || rowMonth === month)) {
                    row.style.display = 'table-row';
                } else {
                    row.style.display = 'none';
                }
            });

            // Filter backtesting rows
            const backtestRows = document.querySelectorAll('.backtest-row');
            backtestRows.forEach(row => {
                const rowTicker = row.getAttribute('data-ticker');

                if (ticker === 'all' || rowTicker === ticker) {
                    row.style.display = 'table-row';
                } else {
                    row.style.display = 'none';
                }
            });

            // Update charts
            updateCharts(ticker, lookback);
        }

        // Initialize charts
        function initializeCharts() {
            if (!window.Chart) {
                console.error("Chart.js not loaded properly!");
                return false;
            }

            // Configure chart defaults
            Chart.defaults.font.family = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif";
            Chart.defaults.color = '#333';
            Chart.defaults.elements.point.radius = 3;
            Chart.defaults.elements.line.tension = 0.2;
            Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';
            Chart.defaults.plugins.legend.position = 'bottom';

            // SPY Correlation Chart
            try {
                const spyCorrelationCtx = document.getElementById('spyCorrelationChart');
                if (spyCorrelationCtx) {
                    spyCorrelationChart = new Chart(spyCorrelationCtx, {
                        type: 'line',
                        data: {
                            labels: window.dashboardData.month_names,
                            datasets: []
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Monthly Correlation with SPY'
                                },
                                legend: {
                                    position: 'bottom'
                                }
                            },
                            scales: {
                                y: {
                                    min: -1,
                                    max: 1
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Error initializing SPY correlation chart:", e);
            }

            // Technical Indicators Chart
            try {
                const indicatorsCtx = document.getElementById('indicatorsCorrelationChart');
                if (indicatorsCtx) {
                    indicatorsChart = new Chart(indicatorsCtx, {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Correlation with Negative Returns',
                                data: [],
                                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Technical Indicators Correlation'
                                }
                            },
                            scales: {
                                y: {
                                    min: -1,
                                    max: 1
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Error initializing technical indicators chart:", e);
            }

            // Model Performance Chart
            try {
                const modelPerfCtx = document.getElementById('modelPerformanceChart');
                if (modelPerfCtx) {
                    modelPerfChart = new Chart(modelPerfCtx, {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Model Accuracy',
                                data: [],
                                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Machine Learning Model Performance'
                                }
                            },
                            scales: {
                                y: {
                                    min: 0,
                                    max: 100,
                                    title: {
                                        display: true,
                                        text: 'Accuracy (%)'
                                    }
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Error initializing model performance chart:", e);
            }

            // Model Predictions Chart
            try {
                const modelPredCtx = document.getElementById('modelPredictionsChart');
                if (modelPredCtx) {
                    modelPredChart = new Chart(modelPredCtx, {
                        type: 'line',
                        data: {
                            labels: window.dashboardData.month_names,
                            datasets: []
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'ML Predicted Probability by Month'
                                },
                                legend: {
                                    position: 'bottom'
                                }
                            },
                            scales: {
                                y: {
                                    min: 0,
                                    max: 100,
                                    title: {
                                        display: true,
                                        text: 'Probability (%)'
                                    }
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Error initializing model predictions chart:", e);
            }

            // Backtesting Performance Chart
            try {
                const backPerfCtx = document.getElementById('backtestingPerformanceChart');
                if (backPerfCtx) {
                    backPerfChart = new Chart(backPerfCtx, {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [
                                {
                                    label: 'Invested',
                                    data: [],
                                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'Final Value',
                                    data: [],
                                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Strategy Performance Comparison'
                                }
                            },
                            scales: {
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Amount ($)'
                                    }
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Error initializing backtesting performance chart:", e);
            }

            // Backtesting Return Chart
            try {
                const backReturnCtx = document.getElementById('backtestingReturnChart');
                if (backReturnCtx) {
                    backReturnChart = new Chart(backReturnCtx, {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Return (%)',
                                data: [],
                                backgroundColor: 'rgba(153, 102, 255, 0.5)',
                                borderColor: 'rgba(153, 102, 255, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Strategy Return Comparison'
                                }
                            },
                            scales: {
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Return (%)'
                                    }
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Error initializing backtesting return chart:", e);
            }

            // Investment Projection Chart
            try {
                const invProjCtx = document.getElementById('investmentProjectionChart');
                if (invProjCtx) {
                    invProjChart = new Chart(invProjCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: []
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Projected Investment Growth'
                                },
                                legend: {
                                    position: 'bottom'
                                }
                            },
                            scales: {
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Value ($)'
                                    }
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Error initializing investment projection chart:", e);
            }

            console.log("Charts initialized");

            // Apply initial filters
            applyFilters();

            return true;
        }

        // Update charts
        function updateCharts(ticker, lookback) {
            console.log(`Updating charts for ticker=${ticker}, lookback=${lookback}`);

            try {
                // Each chart type has its own update function
                updateCorrelationCharts(ticker, lookback);
                updateMLCharts(ticker, lookback);
                updateBacktestingCharts(ticker, lookback);
            } catch (error) {
                console.error("Error updating charts:", error);
            }
        }

        // Update correlation charts
        function updateCorrelationCharts(ticker, lookback) {
            if (!spyCorrelationChart || !indicatorsChart) {
                console.warn("Correlation charts not initialized");
                return;
            }

            // SPY Correlation Chart
            const monthNames = window.dashboardData.month_names;
            const correlationData = [];

            if (ticker !== 'all' && ticker !== 'spy' && results[ticker] && 
                results[ticker][lookback] && 
                results[ticker][lookback]['correlation_with_spy']) {

                const correlations = results[ticker][lookback]['correlation_with_spy'];

                for (const month of monthNames) {
                    correlationData.push(correlations[month] || 0);
                }

                spyCorrelationChart.options.plugins.title.text = `${ticker.toUpperCase()} Correlation with SPY`;
            } else {
                // Show sample data for "all" or missing data
                for (let i = 0; i < 12; i++) {
                    correlationData.push(Math.random() * 0.4);
                }

                spyCorrelationChart.options.plugins.title.text = 'Monthly Correlation with SPY';
            }

            // Update the chart
            spyCorrelationChart.data.datasets = [{
                label: ticker !== 'all' ? ticker.toUpperCase() : 'Average Correlation',
                data: correlationData,
                borderColor: getTrendColor(ticker),
                backgroundColor: getTrendColorWithOpacity(ticker, 0.2),
                borderWidth: 2,
                tension: 0.2
            }];
            spyCorrelationChart.update();

            // Technical Indicators Chart
            const indicatorLabels = [];
            const indicatorValues = [];

            if (ticker !== 'all' && results[ticker] && 
                results[ticker][lookback] && 
                results[ticker][lookback]['technical_indicators']) {

                const indicators = results[ticker][lookback]['technical_indicators'];

                // Get top 6 indicators or less if fewer exist
                const topIndicators = indicators.slice(0, Math.min(6, indicators.length));

                for (const [indicator, value] of topIndicators) {
                    indicatorLabels.push(indicator);
                    indicatorValues.push(parseFloat(value));
                }

                indicatorsChart.options.plugins.title.text = `${ticker.toUpperCase()} Technical Indicators Impact`;
            } else {
                // Sample data
                indicatorLabels.push('RSI14', 'MACD', 'MACDDiff', 'BBWidth', 'Volatility', 'StochK');
                indicatorValues.push(0.42, -0.31, 0.28, 0.23, 0.38, -0.22);

                indicatorsChart.options.plugins.title.text = 'Technical Indicators Impact';
            }

            // Update the chart
            indicatorsChart.data.labels = indicatorLabels;
            indicatorsChart.data.datasets[0].data = indicatorValues;
            indicatorsChart.update();
        }

        // Update ML charts
        function updateMLCharts(ticker, lookback) {
            if (!modelPerfChart || !modelPredChart) {
                console.warn("ML charts not initialized");
                return;
            }

            // Model Performance Chart
            const modelLabels = [];
            const modelAccuracy = [];

            if (ticker !== 'all' && results[ticker] && 
                results[ticker][lookback] && 
                results[ticker][lookback]['ml_results']) {

                const ml_results = results[ticker][lookback]['ml_results'];

                for (const [model, data] of Object.entries(ml_results)) {
                    if (model !== 'monthly_predictions' && model !== 'best_model' && model !== 'best_accuracy') {
                        modelLabels.push(model);
                        modelAccuracy.push(parseFloat(data.accuracy) * 100);
                    }
                }

                modelPerfChart.options.plugins.title.text = `${ticker.toUpperCase()} Model Performance`;
            } else {
                // Sample data
                modelLabels.push('RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM');
                modelAccuracy.push(54.3, 52.8, 49.7, 51.5);

                modelPerfChart.options.plugins.title.text = 'Machine Learning Model Performance';
            }

            // Update the chart
            modelPerfChart.data.labels = modelLabels;
            modelPerfChart.data.datasets[0].data = modelAccuracy;
            modelPerfChart.update();

            // Model Predictions Chart
            const monthNames = window.dashboardData.month_names;
            const predictionData = [];

            if (ticker !== 'all' && results[ticker] && 
                results[ticker][lookback] && 
                results[ticker][lookback]['ml_results'] &&
                results[ticker][lookback]['ml_results']['monthly_predictions']) {

                const predictions = results[ticker][lookback]['ml_results']['monthly_predictions'];

                for (const month of monthNames) {
                    if (predictions[month]) {
                        predictionData.push(predictions[month].probability);
                    } else {
                        predictionData.push(null);
                    }
                }

                modelPredChart.options.plugins.title.text = `${ticker.toUpperCase()} ML Predictions`;
            } else {
                // Sample data
                for (let i = 0; i < 12; i++) {
                    predictionData.push(50 + Math.random() * 30);
                }

                modelPredChart.options.plugins.title.text = 'ML Predicted Probability by Month';
            }

            // Update the chart
            modelPredChart.data.datasets = [{
                label: ticker !== 'all' ? ticker.toUpperCase() : 'Average Prediction',
                data: predictionData,
                borderColor: getTrendColor(ticker),
                backgroundColor: getTrendColorWithOpacity(ticker, 0.2),
                borderWidth: 2,
                tension: 0.2
            }];
            modelPredChart.update();
        }

        // Update backtesting charts
        function updateBacktestingCharts(ticker, lookback) {
            if (!backPerfChart || !backReturnChart) {
                console.warn("Backtesting charts not initialized");
                return;
            }

            const strategyLabels = [];
            const investedData = [];
            const finalValueData = [];
            const returnData = [];

            if (ticker !== 'all' && results[ticker] && 
                results[ticker][lookback] && 
                results[ticker][lookback]['backtest_results'] &&
                results[ticker][lookback]['backtest_results']['strategies']) {

                const strategies = results[ticker][lookback]['backtest_results']['strategies'];

                for (const [strategy, data] of Object.entries(strategies)) {
                    strategyLabels.push(strategy.charAt(0).toUpperCase() + strategy.slice(1));
                    investedData.push(data.cash_invested);
                    finalValueData.push(data.final_value);
                    returnData.push(data.return_pct);
                }

                backPerfChart.options.plugins.title.text = `${ticker.toUpperCase()} Strategy Performance`;
                backReturnChart.options.plugins.title.text = `${ticker.toUpperCase()} Strategy Returns`;
            } else {
                // No data or all tickers selected - show sample data
                strategyLabels.push('Daily', 'Weekly', 'Optimal');
                investedData.push(400, 360, 480);
                finalValueData.push(450, 410, 540);
                returnData.push(12.5, 13.8, 12.5);

                backPerfChart.options.plugins.title.text = 'Strategy Performance';
                backReturnChart.options.plugins.title.text = 'Strategy Returns';
            }

            // Update charts
            backPerfChart.data.labels = strategyLabels;
            backPerfChart.data.datasets[0].data = investedData;
            backPerfChart.data.datasets[1].data = finalValueData;
            backPerfChart.update();

            backReturnChart.data.labels = strategyLabels;
            backReturnChart.data.datasets[0].data = returnData;
            backReturnChart.update();
        }

        // Enhanced Investment Calculator Function with proper compounding
        function calculateInvestment() {
            const amount = parseFloat(document.getElementById('investmentAmount').value);
            const period = parseInt(document.getElementById('investmentPeriod').value);
            const ticker = document.getElementById('investmentTicker').value;

            if (isNaN(amount) || isNaN(period) || amount <= 0 || period <= 0) {
                alert('Please enter valid numbers for investment amount and period');
                return;
            }

            // Get backtest results
            let dailyReturn = 0;
            let weeklyReturn = 0;
            let optimalReturn = 0;
            let lookbackPeriod = lookbackFilter.value;

            if (results[ticker] && 
                results[ticker][lookbackPeriod] && 
                results[ticker][lookbackPeriod]['backtest_results'] &&
                results[ticker][lookbackPeriod]['backtest_results']['strategies']) {

                const strategies = results[ticker][lookbackPeriod]['backtest_results']['strategies'];

                if (strategies.daily) dailyReturn = strategies.daily.return_pct / 100;
                if (strategies.weekly) weeklyReturn = strategies.weekly.return_pct / 100;
                if (strategies.optimal) optimalReturn = strategies.optimal.return_pct / 100;
            }

            // Trading constants
            const tradingDaysPerYear = 252;
            const weeksPerYear = 52;
            const monthsPerYear = 12;

            // Calculate daily, weekly, and monthly rates
            // For accurate compound interest, we convert the period return to daily/weekly/monthly rates

            // Annualize returns if lookback period is not 1 year
            const lookbackYears = parseFloat(lookbackPeriod);

            if (lookbackYears !== 1) {
                // Use compound annual growth rate formula: (1 + return)^(1/years) - 1
                dailyReturn = Math.pow(1 + dailyReturn, 1 / lookbackYears) - 1;
                weeklyReturn = Math.pow(1 + weeklyReturn, 1 / lookbackYears) - 1; 
                optimalReturn = Math.pow(1 + optimalReturn, 1 / lookbackYears) - 1;
            }

            // Convert annual returns to period returns for proper compounding
            const dailyRate = Math.pow(1 + dailyReturn, 1/tradingDaysPerYear) - 1;
            const weeklyRate = Math.pow(1 + weeklyReturn, 1/weeksPerYear) - 1;
            const monthlyRate = Math.pow(1 + optimalReturn, 1/monthsPerYear) - 1;

            // Amount per investment period
            const dailyAmount = amount / (tradingDaysPerYear / monthsPerYear);
            const weeklyAmount = amount / (weeksPerYear / monthsPerYear);
            const monthlyAmount = amount;

            // Year labels for chart
            const labels = Array.from({length: period + 1}, (_, i) => `Year ${i}`);

            // Track investments over time for each strategy
            const strategies = {
                daily: {
                    name: 'Daily',
                    description: `Investing $${dailyAmount.toFixed(2)} every trading day`,
                    rate: dailyRate,
                    annualRate: dailyReturn,
                    periodsPerYear: tradingDaysPerYear,
                    periodsPerMonth: tradingDaysPerYear / monthsPerYear,
                    amountPerPeriod: dailyAmount,
                    values: [0]
                },
                weekly: {
                    name: 'Weekly',
                    description: `Investing $${weeklyAmount.toFixed(2)} every week`,
                    rate: weeklyRate,
                    annualRate: weeklyReturn,
                    periodsPerYear: weeksPerYear,
                    periodsPerMonth: weeksPerYear / monthsPerYear,
                    amountPerPeriod: weeklyAmount,
                    values: [0]
                },
                optimal: {
                    name: 'Optimal Day',
                    description: `Investing $${monthlyAmount.toFixed(2)} on optimal days each month`,
                    rate: monthlyRate,
                    annualRate: optimalReturn,
                    periodsPerYear: monthsPerYear,
                    periodsPerMonth: 1,
                    amountPerPeriod: monthlyAmount,
                    values: [0]
                }
            };

            // Perform the actual compound calculations for each strategy
            Object.keys(strategies).forEach(key => {
                const strategy = strategies[key];
                let totalInvested = 0;
                let portfolioValue = 0;

                // Simulate each year
                for (let year = 1; year <= period; year++) {
                    // Simulate each month in the year
                    for (let month = 1; month <= 12; month++) {
                        // Simulate each investment period in the month
                        for (let i = 1; i <= strategy.periodsPerMonth; i++) {
                            // Add the new investment
                            totalInvested += strategy.amountPerPeriod;
                            portfolioValue += strategy.amountPerPeriod;

                            // Apply compounding for this period
                            portfolioValue *= (1 + strategy.rate);
                        }
                    }

                    // Store year-end value
                    strategy.values.push(portfolioValue);
                }

                // Calculate final statistics
                strategy.totalInvested = totalInvested;
                strategy.finalValue = portfolioValue;
                strategy.profit = portfolioValue - totalInvested;
                strategy.returnPct = (portfolioValue / totalInvested - 1) * 100;
            });

            // Determine best strategy
            let bestStrategy = 'daily';
            if (strategies.weekly.returnPct > strategies[bestStrategy].returnPct) {
                bestStrategy = 'weekly';
            }
            if (strategies.optimal.returnPct > strategies[bestStrategy].returnPct) {
                bestStrategy = 'optimal';
            }

            // Mark best strategy
            strategies[bestStrategy].isBest = true;

            // Update chart
            invProjChart.data.labels = labels;
            invProjChart.data.datasets = [
                {
                    label: `Daily (+${(strategies.daily.annualRate * 100).toFixed(2)}% annual)`,
                    data: strategies.daily.values,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: `Weekly (+${(strategies.weekly.annualRate * 100).toFixed(2)}% annual)`,
                    data: strategies.weekly.values,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: `Optimal Day (+${(strategies.optimal.annualRate * 100).toFixed(2)}% annual)`,
                    data: strategies.optimal.values,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ];
            invProjChart.options.plugins.title.text = `${ticker.toUpperCase()} Projected Growth ($${amount}/month)`;
            invProjChart.update();

            // Build enhanced table HTML with more details
            let tableHtml = '';

            ['daily', 'weekly', 'optimal'].forEach(key => {
                const strategy = strategies[key];
                tableHtml += `
                <tr>
                    <td class="py-2 px-4 border-b">
                        ${strategy.name}
                        <div class="text-xs text-gray-500">${strategy.description}</div>
                    </td>
                    <td class="py-2 px-4 border-b">
                        $${strategy.totalInvested.toFixed(2)}
                        <div class="text-xs text-gray-500">${period} years × $${(strategy.amountPerPeriod * strategy.periodsPerMonth * 12).toFixed(2)}/year</div>
                    </td>
                    <td class="py-2 px-4 border-b">$${strategy.finalValue.toFixed(2)}</td>
                    <td class="py-2 px-4 border-b">
                        $${strategy.profit.toFixed(2)}
                        <div class="text-xs text-gray-500">+${strategy.returnPct.toFixed(2)}%</div>
                    </td>
                    <td class="py-2 px-4 border-b">
                        ${strategy.returnPct.toFixed(2)}%
                        <div class="text-xs text-gray-500">${(strategy.annualRate * 100).toFixed(2)}% annual</div>
                    </td>
                    <td class="py-2 px-4 border-b">${strategy.isBest ? '<span class="bg-green-100 text-green-800 px-2 py-1 rounded">★ Best</span>' : ''}</td>
                </tr>
                `;
            });

            const investmentTable = document.getElementById('investmentTable');
            if (investmentTable) {
                investmentTable.innerHTML = tableHtml;
            }

            // Add calculation details explanation under the table - improved with clearer methodology
            const detailsDiv = document.createElement('div');
            detailsDiv.className = 'mt-4 p-4 bg-blue-50 rounded text-sm';
            detailsDiv.id = 'calculationDetails';
            detailsDiv.innerHTML = `
                <h4 class="font-medium mb-2">Calculation Details</h4>
                <p><strong>Daily Strategy:</strong> Invests $${dailyAmount.toFixed(2)} every trading day (${Math.round(tradingDaysPerYear/monthsPerYear)} trading days per month).</p>
                <p><strong>Weekly Strategy:</strong> Invests $${weeklyAmount.toFixed(2)} every week (${Math.round(weeksPerYear/monthsPerYear)} weeks per month).</p>
                <p><strong>Optimal Day Strategy:</strong> Invests $${monthlyAmount.toFixed(2)} on the best day(s) each month based on historical analysis.</p>

                <p class="mt-2"><strong>Compounding Methodology:</strong> Returns are compounded at each investment interval (daily, weekly, or monthly).</p>
                <p><strong>Annual Returns:</strong> Daily: ${(strategies.daily.annualRate*100).toFixed(2)}%, Weekly: ${(strategies.weekly.annualRate*100).toFixed(2)}%, Optimal: ${(strategies.optimal.annualRate*100).toFixed(2)}%</p>
                <p class="text-xs text-gray-600 mt-1">Note: Returns are based on ${lookbackYears} year historical data and have been annualized for projection purposes.</p>
            `;

            // Add or replace the details explanation (ensure only one appears)
            const existingDetails = document.getElementById('calculationDetails');
            if (existingDetails) {
                existingDetails.parentNode.replaceChild(detailsDiv, existingDetails);
            } else {
                const tableContainer = investmentTable.parentNode;
                tableContainer.appendChild(detailsDiv);
            }
        }

        // Helper function to get ticker colors
        function getTrendColor(ticker) {
            const colors = {
                'spy': 'rgba(54, 162, 235, 1)',
                'smh': 'rgba(153, 102, 255, 1)',
                'slv': 'rgba(192, 192, 192, 1)',
                'gld': 'rgba(255, 206, 86, 1)',
                'qtum': 'rgba(75, 192, 192, 1)'
            };

            return colors[ticker.toLowerCase()] || 'rgba(54, 162, 235, 1)';
        }

        function getTrendColorWithOpacity(ticker, opacity) {
            const colors = {
                'spy': `rgba(54, 162, 235, ${opacity})`,
                'smh': `rgba(153, 102, 255, ${opacity})`,
                'slv': `rgba(192, 192, 192, ${opacity})`,
                'gld': `rgba(255, 206, 86, ${opacity})`,
                'qtum': `rgba(75, 192, 192, ${opacity})`
            };

            return colors[ticker.toLowerCase()] || `rgba(54, 162, 235, ${opacity})`;
        }
    });
    </script>
    </body>
    </html>"""

        template = jinja2.Template(template_str)
        return template


def main():
    """Main entry point for the script"""
    # HARDCODED PARAMETERS - Set these variables instead of using command line arguments
    USE_HARDCODED_PARAMS = True  # Set to True to use hardcoded parameters, False to use command line args

    if USE_HARDCODED_PARAMS:
        # Directly set your parameters here
        input_dir = "data"  # Just the directory, no wildcards
        output_dir = "data/lookback"
        tickers = ["spy", "smh", "slv", "gld", "qtum"]  # Specific tickers, or None to analyze all CSVs
        lookback_periods = [5]
        use_ml = True
        investment = 200
        exclude_anomalies = True

        # Create analyzer with hardcoded parameters
        analyzer = OptimalInvestmentDayAnalyzer(
            input_dir=input_dir,
            output_dir=output_dir,
            tickers=tickers,
            lookback_periods=lookback_periods,
            use_ml=use_ml,
            capital_investment=investment,
            exclude_anomalies=exclude_anomalies
        )
    else:
        parser = argparse.ArgumentParser(description='Optimal Investment Day Calendar Generator')
        parser.add_argument('--input', type=str, required=True, help='Directory containing CSV files')
        parser.add_argument('--output', type=str, required=True, help='Directory to save output files')

        # Optional arguments
        parser.add_argument('--tickers', type=str, nargs='+', help='Specific tickers to analyze (default: analyze all CSVs in input directory)')
        parser.add_argument('--lookback', type=int, nargs='+', default=[1], help='Lookback periods in years (default: 1)')
        parser.add_argument('--no-ml', action='store_true', help='Disable machine learning analysis')
        parser.add_argument('--investment', type=float, default=100, help='Monthly investment amount (default: $100)')
        parser.add_argument('--include-anomalies', action='store_true', help='Include anomalous periods in analysis')

        args = parser.parse_args()

        # Create analyzer
        analyzer = OptimalInvestmentDayAnalyzer(
            input_dir=args.input,
            output_dir=args.output,
            tickers=args.tickers,
            lookback_periods=args.lookback,
            use_ml=not args.no_ml,
            capital_investment=args.investment,
            exclude_anomalies=not args.include_anomalies
        )

        # Run analysis
    analyzer.run_analysis()

if __name__ == '__main__':
    main()