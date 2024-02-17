#%%
#1. Setup - maily importing packages
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#%%
#2. Load the data as dataframe
df=pd.read_csv('Google_Stock_Price.csv')

# %%
df.head()
# %%
df.info()

# %%
#3. Data cleaning
#Convert the datatype obejct to float
df['Close'] = df['Close'].str.replace(',','').astype(float)
df['Volume'] = df['Volume'].str.replace(',','').astype(float)

# %%
#Pop date out of the dataframe
date = pd.to_datetime(df.pop('Date'), format="%m/%d/%Y")


#%%
#4. Data Inspection
plot_cols = ['Open', 'High', 'Low']
plot_features = df[plot_cols]
plot_features.index = date
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date[:480]
_ = plot_features.plot(subplots=True)

# %%
df.describe().transpose()

# %%
#5. Data Splitting - 70:20:10 splits for train-validation-test
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

# %%
#6. Data normalization -standard scaling
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# %%
#7. Create a class to perform data windowing
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def plot(self, model=None, plot_col='Open', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue
        
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')
    
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
   return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result
  
# %%
# Example window 1
window_1 = WindowGenerator(input_width=30, label_width=30, shift=1,
                     label_columns=['Open'])
window_1  

# %%
# Example window 2
w2 = WindowGenerator(input_width=30, label_width=1, shift=1,
                     label_columns=['Open'])
w2

# %%
#Try out the split window method
# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

# %%
#Try out the plot method
w2.example = example_inputs, example_labels
w2.plot()

# %%
#8. Model development
#8.1 Create single-step model
wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    label_columns=['Open'])

wide_window

# %%
#8.2 Create LSTM Model
lstm_single_step = tf.keras.Sequential()
lstm_single_step.add(tf.keras.layers.LSTM(32, return_sequences=True))
lstm_single_step.add(tf.keras.layers.Dense(1))


# %%
MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

# %%
#9. Model training 
history_single_step = compile_and_fit(lstm_single_step, wide_window)

# %%
#Plot graphs to display training result
#(A) Loss graph
plt.figure()
plt.plot(history_single_step.history['loss'])
plt.plot(history_single_step.history['val_loss'])
plt.legend(['Training Loss', 'Validaton Loss'])
plt.show()

# %%
#10. Plot the result
wide_window.plot(lstm_single_step, plot_col = 'Open')

# %%
#11. Multi step model
OUT_STEPS = 30
multi_window = WindowGenerator(input_width=30,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
multi_window

# %%
#12. Create the model 
lstm_multi_step = tf.keras.Sequential()
#lstm_multi_step.add(tf.keras.layers.LSTM(64, return_sequences=False))
lstm_multi_step.add(tf.keras.layers.LSTM(32, return_sequences=False))
lstm_multi_step.add(tf.keras.layers.Dense(OUT_STEPS*num_features))
lstm_multi_step.add(tf.keras.layers.Reshape([OUT_STEPS, num_features]))
#lstm_multi_step.add(tf.keras.layers.Dropout(0.3))
#lstm_multi_step.add(tf.keras.layers.Dropout(0.5))
history_multi_step = compile_and_fit(lstm_multi_step, multi_window)

# %%
multi_window.plot(lstm_multi_step, plot_col='Open')

# %%
#Plot graphs to display training result
#(A) Loss graph
plt.figure()
plt.plot(history_multi_step.history['loss'])
plt.plot(history_multi_step.history['val_loss'])
plt.legend(['Training Loss', 'Validaton Loss'])
plt.show()
# %%
