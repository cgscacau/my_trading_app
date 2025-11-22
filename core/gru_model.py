import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import streamlit as st

class GRUModel:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, df, target_col='close'):
        data = df[target_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaled_data

    def build_model(self, units=50, dropout=0.2, layers=2):
        model = Sequential()
        # Camada de entrada
        model.add(GRU(units=units, return_sequences=(layers > 1), input_shape=(self.lookback, 1)))
        model.add(Dropout(dropout))
        
        # Camadas ocultas
        for i in range(1, layers):
            return_seq = (i < layers - 1)
            model.add(GRU(units=units, return_sequences=return_seq))
            model.add(Dropout(dropout))
            
        # Saída
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return history

    def predict(self, X):
        return self.model.predict(X)
