import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('data_for_forecast.csv').drop(columns = ['Unnamed: 0', 'date'])

features = data.drop(columns = ['sales']).copy()
target = data['sales'].copy()

features_train, features_test, target_train, target_test = train_test_split(np.array(features), np.array(target), shuffle = False)
import tensorflow as tf

class CNNmodel(tf.keras.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, el_drop_out, num_outputs, k_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.el_drop_out = el_drop_out
        self.num_outputs = num_outputs
        self.k_size = k_size
        
        self.add(tf.keras.Input(shape = (input_dim)))
        self.add(tf.keras.layers.Reshape((input_dim, 1)))
        self.add(tf.keras.layers.Conv1D(filters = num_outputs, kernel_size = k_size, padding = 'same', activation = 'relu'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(self.el_drop_out))
        
        for i in range(hidden_dim):
            if self.el_drop_out < 0.5:
                self.el_drop_out=+0.2
            
            if i % 2 != 0:
                self.num_outputs=self.num_outputs*2
            
            self.add(tf.keras.layers.Conv1D(filters = self.num_outputs, kernel_size = self.k_size, activation = 'relu'))
            self.add(tf.keras.layers.BatchNormalization())
            self.add(tf.keras.layers.Dropout(self.el_drop_out))
            
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(self.num_outputs, activation = 'relu'))
        self.add(tf.keras.layers.Dropout(self.el_drop_out))
        self.add(tf.keras.layers.Dense(units = 1))
        
    def build(self, loss, lr, metrics):
        self.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate = lr), metrics = metrics)

model = CNNmodel(20, 5, 1, 0.1, 16, 3)
model.build('mse', 0.001, ['mae'])
model.summary()

checpoin_model = tf.keras.callbacks.ModelCheckpoint(filepath = 'models/tensorflow_CNN_model.h5',
                                                   save_wietghts_only = True,
                                                   save_best_only = True,
                                                   mode = 'min',
                                                   monitor = 'val_loss',
                                                   verbose = 1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                 factor = 0.1, 
                                                 patience = 5, 
                                                 verbose = 1, 
                                                 mode = 'min', 
                                                 min_lr = 0.001)


history = model.fit(x = features_train, 
                    y = target_train, 
                    epochs = 35, 
                    validation_split = 0.2, 
                    callbacks = [checpoin_model, reduce_lr],
                    shuffle = False, 
                    batch_size=4096)                                                 

history_df = pd.DataFrame(history.history)
plt.plot(history_df['loss'], label = 'loss')
plt.plot(history_df['val_loss'], label = 'val_loss')
print("val_mae = ", list(history_df["val_mae"])[-1])
plt.xlabel("epochs")
plt.legend()
plt.show()

model.load_weights('models/tensorflow_CNN_model.h5')
y_pred = np.array(model.predict(features_test)).flatten()
mn = 0
mx = 100000
plt.figure(figsize=(7,7))
a = plt.axes(aspect='equal')
plt.scatter(target_test, y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Actual vs predicted values')
plt.xlim([mn, mx])
plt.ylim([mn, mx])
plt.grid()
plt.plot([mn, mx], [mn, mx])
plt.show()

plt.plot(target_test[:300], linewidth = 1.7, color = 'red')
plt.show()
plt.plot(y_pred[:300])
plt.show()