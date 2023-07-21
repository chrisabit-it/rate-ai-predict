import math
from pickle import dump
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model, Sequential
from pandas import read_json, DataFrame, Series
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from yfinance import Ticker

AI_PATH = 'ai'
CACHE_PATH = 'cache'

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})


class TrainRatePredict:
	_MODEL_NAME_PREFIX = 'rate_predict'

	def __init__(self,
	             rate_col='MATP',
	             neurons=20,
	             features=None,
	             epochs=100,
				 batch_size=32,
				 hidden_layer=2,
				 dropout=0.2,
				 stop_early=True,
				 optimizer='adam',
				 loss='mae',
				 validation_split=0.2,
				 train_data_ratio=0.8,
				 predict_gap_len=1,
				 bol_win=20,
				 rsi_win=14):
		self._rate_col = rate_col
		self._neurons = neurons
		if features is None:
			features = [rate_col, 'RSI', 'BOBW', 'BOLU-DELTA', 'BOLD-DELTA']
		self._features = features
		self._epochs = epochs
		self._batch_size = batch_size
		self._hidden_layer = hidden_layer
		self._dropout = dropout
		self._stop_early = stop_early
		self._optimizer = optimizer
		self._loss = loss
		self._validation_split = validation_split
		self._train_data_ratio = train_data_ratio
		self._predict_gap_len = predict_gap_len
		self._bol_win = bol_win
		self._rsi_win = rsi_win

		self._symbol: str = None
		self._rate_df: DataFrame = None
		self._data_scaler: MinMaxScaler = None
		self._pred_scaler: MinMaxScaler = None
		self._train_scaled_tuple = None
		self._test_scaled_tuple = None
		self._test_unscaled_tuple = None
		self._model: Model = None
		self._history = None
		self._y_pred_unscaled = None

	@property
	def rate_df(self) -> DataFrame:
		return self._rate_df

	@staticmethod
	def _calc_bolling(bol_df: DataFrame, window: int):
		bol_df['TP'] = (bol_df['Close'] + bol_df['Low'] + bol_df['High']) / 3
		bol_df['STD'] = bol_df['TP'].rolling(window).std()
		bol_df['MATP'] = bol_df['TP'].rolling(window).mean()

		bol_df['BOLU'] = bol_df['MATP'] + (2.0 * bol_df['STD'])
		bol_df['BOLD'] = bol_df['MATP'] - (2.0 * bol_df['STD'])
		bol_df['BOPB'] = (bol_df['TP'] - bol_df['BOLU']) / (bol_df['BOLU'] - bol_df['BOLD']) + 1.
		bol_df['BOPBU'] = 1.
		bol_df['BOPBD'] = 0.
		bol_df['BOBW'] = (bol_df['BOLU'] - bol_df['BOLD']) / bol_df['MATP']
		bol_df['BOBW'] = bol_df['BOBW']

	@staticmethod
	def _calc_rsi(candles_df: DataFrame, window: int):
		rsi_df = DataFrame(data=candles_df['Close'], index=candles_df.index)
		rsi_df['DIFF'] = rsi_df.diff(1)
		rsi_df['GAIN'] = rsi_df['DIFF'].clip(lower=0)
		rsi_df['LOSS'] = rsi_df['DIFF'].clip(upper=0).abs()
		rsi_df['AVG_GAIN'] = rsi_df['GAIN'].rolling(window=window, min_periods=window).mean()[:window + 1]
		rsi_df['AVG_LOSS'] = rsi_df['LOSS'].rolling(window=window, min_periods=window).mean()[:window + 1]
		for i, row in enumerate(rsi_df['AVG_GAIN'].iloc[window + 1:]):
			rsi_df['AVG_GAIN'].iloc[i + window + 1] = \
				(rsi_df['AVG_GAIN'].iloc[i + window] *
				 (window - 1) +
				 rsi_df['GAIN'].iloc[i + window + 1]) \
				/ window

		for i, row in enumerate(rsi_df['AVG_LOSS'].iloc[window + 1:]):
			rsi_df['AVG_LOSS'].iloc[i + window + 1] = \
				(rsi_df['AVG_LOSS'].iloc[i + window] *
				 (window - 1) +
				 rsi_df['LOSS'].iloc[i + window + 1]) \
				/ window

		rsi_df['RS'] = rsi_df['AVG_GAIN'] / rsi_df['AVG_LOSS']
		rsi_df['RSI'] = 100 - (100 / (1.0 + rsi_df['RS']))
		candles_df['RSI'] = rsi_df['RSI']

	@staticmethod
	def _gen_lstm_model(input_shape: tuple, in_units: int, out_units: int,
	                    preout_units: int,
	                    dropout: float = None,
	                    hidden_layer=1) -> Model:
		model = Sequential()

		model.add(LSTM(in_units, return_sequences=hidden_layer > 0, input_shape=input_shape))

		if dropout is not None:
			model.add(Dropout(dropout))

		for i in range(1, hidden_layer + 1):
			model.add(LSTM(units=in_units, return_sequences=i < hidden_layer))
			if dropout is not None:
				model.add(Dropout(dropout))

		if preout_units is not None:
			model.add(Dense(preout_units))
		model.add(Dense(out_units))

		return model

	def fetch_candles(self, symbol: str, begin: str, end: str):
		rate_file = Path(CACHE_PATH, f'{symbol}_{self._rsi_win}_{self._bol_win}_{begin}_{end}.json')

		if Path(rate_file).exists():
			rate_df: DataFrame = read_json(rate_file)
		else:
			rate_df = Ticker(symbol).history(start=begin, end=end)
			self._calc_bolling(rate_df, window=self._bol_win)
			self._calc_rsi(rate_df, window=self._rsi_win)
			if not rate_file.parent.exists():
				rate_file.parent.mkdir()
			rate_df.to_json(rate_file)
		rate_df = rate_df.sort_index()
		rate_df['BOLU-DELTA'] = rate_df['BOLU'] - rate_df['Close']
		rate_df['BOLD-DELTA'] = rate_df['Close'] - rate_df['BOLD']

		self._symbol = symbol

		self._rate_df = rate_df

	def plot_candles(self, tail_len=500):
		assert self._symbol is not None
		assert self._rate_df is not None

		plot_df = self._rate_df.tail(tail_len)
		filtered_plot_df = plot_df[self._features]
		fig, ax = plt.subplots(nrows=filtered_plot_df.shape[1] + 1, sharex=1, figsize=(16, 8))
		plt.ylabel(self._symbol, fontsize=18)
		sns.set_palette(["#090364", "#1960EF", "#EF5919"])

		for i, ax in enumerate(fig.axes):
			if i == 0:
				sns.lineplot(data=plot_df[[self._rate_col, 'BOLU', 'BOLD']], ax=ax)
			else:
				sns.lineplot(data=filtered_plot_df.iloc[:, i - 1], ax=ax)
			ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
			ax.xaxis.set_major_locator(mdates.AutoDateLocator())

		fig.tight_layout()
		plt.show()

	@staticmethod
	def _partition_dataset(sequence_len: int, gap_len: int, index_rate: int, data_np: np.ndarray) \
			-> (np.ndarray, np.ndarray):
		x, y = [], []
		data_len = data_np.shape[0] - gap_len
		for i in range(sequence_len, data_len):
			x.append(data_np[i - sequence_len:i])
			y.append(data_np[i + gap_len - 1, index_rate])

		x = np.array(x)
		y = np.array(y)

		return x, y

	def _init_scaler(self):
		data_df = self._rate_df[self._features].dropna()
		index_rate = data_df.columns.get_loc(self._rate_col)

		self._data_scaler = MinMaxScaler()
		self._pred_scaler = MinMaxScaler()
		self._pred_scaler.fit(DataFrame(data_df[self._rate_col]))

		data_np_unscaled = data_df.to_numpy()
		data_np_scaled = self._data_scaler.fit_transform(data_np_unscaled)

		train_data_len = math.ceil(data_np_scaled.shape[0] * self._train_data_ratio)

		train_data_scaled = data_np_scaled[0:train_data_len, :]
		test_data_scaled = data_np_scaled[train_data_len - self._neurons:, :]
		test_data_unscaled = data_np_unscaled[train_data_len - self._neurons:, :]

		self._train_scaled_tuple = self._partition_dataset(sequence_len=self._neurons,
		                                                   gap_len=self._predict_gap_len,
		                                                   index_rate=index_rate,
		                                                   data_np=train_data_scaled)
		self._test_scaled_tuple = self._partition_dataset(sequence_len=self._neurons,
		                                                  gap_len=self._predict_gap_len,
		                                                  index_rate=index_rate,
		                                                  data_np=test_data_scaled)
		self._test_unscaled_tuple = self._partition_dataset(sequence_len=self._neurons,
		                                                    gap_len=self._predict_gap_len,
		                                                    index_rate=index_rate,
		                                                    data_np=test_data_unscaled)

	def gen_model(self):
		assert self._symbol is not None
		self._init_scaler()

		x_train, _ = self._train_scaled_tuple
		in_units = x_train.shape[1] * x_train.shape[2]
		input_shape = (x_train.shape[1], x_train.shape[2])

		self._model = self._gen_lstm_model(input_shape=input_shape, in_units=in_units, out_units=1, preout_units=5,
		                                   dropout=self._dropout, hidden_layer=self._hidden_layer)

		self._model.compile(optimizer=self._optimizer, loss=self._loss)

	def train(self):
		assert self._symbol is not None
		assert self._model is not None
		assert self._train_scaled_tuple is not None

		x_train, y_train = self._train_scaled_tuple

		model_name = f'{self._MODEL_NAME_PREFIX}_{self._symbol}'

		callbacks = [
			ReduceLROnPlateau(
				monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
			)
		]
		if self._stop_early:
			callbacks.append(EarlyStopping(monitor='loss', patience=10, verbose=1))

		self._history = self._model.fit(x_train, y_train,
		                                batch_size=self._batch_size,
		                                epochs=self._epochs,
		                                callbacks=callbacks,
		                                validation_split=self._validation_split)

		model_path = Path(AI_PATH, f'{model_name}.keras')
		if not model_path.parent.exists():
			model_path.parent.mkdir()
		self._model.save(filepath=model_path, save_format='keras', overwrite=True)

	def plot_model_loss(self):
		assert self._symbol is not None
		assert self._model is not None

		fig, ax = plt.subplots(figsize=(30, 5), sharex=True)
		sns.lineplot(data=self._history.history['loss'])
		plt.title(f'Model loss {self._symbol}')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		# ax.xaxis.set_major_locator(plt.MaxNLocator(self._epochs))
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.grid()
		plt.show()

	def test_model(self):
		assert self._symbol is not None
		assert self._model is not None
		assert self._test_scaled_tuple is not None
		assert self._test_unscaled_tuple is not None
		assert self._pred_scaler is not None

		x_test_scaled, _ = self._test_scaled_tuple
		_, y_test_unscaled = self._test_unscaled_tuple
		y_test_unscaled = y_test_unscaled.reshape(-1, 1).reshape(-1, 1)

		y_pred_scaled = self._model.predict(x_test_scaled)
		self._y_pred_unscaled = self._pred_scaler.inverse_transform(y_pred_scaled)

		mae = mean_absolute_error(y_test_unscaled, self._y_pred_unscaled)
		print(f'Median Absolute Error (MAE): {np.round(mae, 2)}')

		mape = np.mean((np.abs(np.subtract(y_test_unscaled, self._y_pred_unscaled) / y_test_unscaled))) * 100
		print(f'Mean Absolute Percentage Error (MAPE): {np.round(mape, 2)} %')

		mdape = np.median((np.abs(np.subtract(y_test_unscaled, self._y_pred_unscaled) / y_test_unscaled))) * 100
		print(f'Median Absolute Percentage Error (MDAPE): {np.round(mdape, 2)} %')

	@staticmethod
	def _plot_bar(plot_df: DataFrame, col_name: str):
		plot_series = plot_df[col_name].dropna()
		_, ax_res = plt.subplots(figsize=(16, 2))
		color = ['#2BC97A' if d > 0 else '#C92B2B' for d in plot_series]
		ax_res.bar(height=plot_series, x=plot_series.index, label=col_name, color=color)

	def plot_metrics(self, tail_len=500):
		assert self._symbol is not None
		assert self._rate_df is not None
		assert self._test_unscaled_tuple is not None
		assert self._y_pred_unscaled is not None

		data_df = self._rate_df[self._features]
		index_rate = data_df.columns.get_loc(self._rate_col)

		x_test, y_test = self._test_unscaled_tuple
		y_pred = self._y_pred_unscaled

		test_df: DataFrame = DataFrame(y_test).rename(columns={0: 'y_test'})
		test_df['y_pred'] = y_pred
		test_df['delta'] = (test_df['y_pred'] / test_df['y_test'] - 1.) * 100.
		test_df['x_last'] = Series(x_test[:, -1, index_rate])
		test_df['trend'] = (test_df['y_pred'] - test_df['x_last']) * (test_df['y_test'] - test_df['x_last'])
		test_df['signum'] = np.sign(test_df['trend'])

		plot_df = test_df.tail(tail_len).copy()
		_, ax = plt.subplots(figsize=(16, 8))
		plt.title(f'y_pred vs y_test')
		plt.ylabel(self._symbol, fontsize=18)
		sns.set_palette(["#090364", "#1960EF"])
		sns.lineplot(data=plot_df[['y_pred', 'y_test']], linewidth=1.0, dashes=False, ax=ax)

		for col_name in ['delta', 'trend', 'signum']:
			self._plot_bar(plot_df, col_name)

		plt.legend()
		plt.show()

		print(f'Mean Signum: {test_df["signum"].mean():.2f}')

	def persist(self):
		assert self._symbol is not None
		assert self._model is not None
		assert self._data_scaler is not None
		assert self._pred_scaler is not None

		model_name = f'{self._MODEL_NAME_PREFIX}_{self._symbol}'
		data_pkl_path = Path(AI_PATH, f'{model_name}_data.pkl')
		pred_pkl_path = Path(AI_PATH, f'{model_name}_pred.pkl')
		dump(self._data_scaler, open(data_pkl_path, 'wb'))
		dump(self._pred_scaler, open(pred_pkl_path, 'wb'))