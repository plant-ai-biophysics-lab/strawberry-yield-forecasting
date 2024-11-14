import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Attention, LeakyReLU, Concatenate, Flatten, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K

#tf.config.run_functions_eagerly(False)

# def custom_loss_wrapper():
#     def custom_loss(y_true, y_pred_and_features): 
#         # Split the combined output tensor back into y_pred and input_features
#         y_pred = y_pred_and_features[:, 0:1]
        
#         # Compute the differences
#         #diff_pred = K.abs(y_true - y_pred)
#         #diff_phen = K.abs(y_true - _phen_y_tf)   

#         # Compute the penalization factor
#         #penalization_factor = K.mean(K.cast(diff_phen < diff_pred, dtype=tf.float32) - 0.5)
#         # Adjust the MSE based on the penalization factor
#         mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
#         #adjusted_mse = mse * (1 + penalization_factor)

#         # Calculate weights based on the difference
#         #weights = 1 + diff_phen / (diff_phen + diff_pred + K.epsilon())
#         # Compute weighted MSE
#         #weighted_mse = K.mean(weights * K.square(y_true - y_pred))

#         #return weighted_mse
#         #return adjusted_mse#mse # + reg_term (if used)
#         return mse
#     return custom_loss

def train_LSTM(epochs,batch_size,optimizer,samples_dim,fold,train_X,train_y,year,block_size,phenological,loss='mse'):
    model_type='LSTM'
    inputs = Input(shape=(samples_dim[0], samples_dim[2]))
    if phenological:
        phen_str = 'XT'
    else:
        phen_str = 'X'
    x = LSTM(80, activation='relu', return_sequences=True)(inputs)
    #x = Dropout(0.1)(x)
    x = LSTM(40, activation='relu', return_sequences=True)(x)
    #x = Dropout(0.1)(x)
    x = LSTM(40, activation='relu', return_sequences=True)(x)
    #x = Dropout(0.1)(x)
    x = LSTM(20, activation='relu')(x)
    #x = Dropout(0.1)(x) 
    x = Dense(20, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
    # Save the model architecture and weights
    checkpoint = ModelCheckpoint('/Users/andres/Documents/strawberry-forecasting/models/'+'best_LSTM'+'_'+str(year)+'_'+str(block_size)+'_'+str(samples_dim[0])+str(samples_dim[1])+'_'+str(fold)+'_'+phen_str+'.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    #wandb.config.epochs = epochs
    #wandb.config.batch_size = batch_size
    #wandb.config.optimizer = optimizer
    history = model.fit(train_X, train_y, shuffle=False, epochs=epochs, batch_size=batch_size, validation_split=0.25, callbacks=[checkpoint])#, WandbCallback()])
    return model_type

def train_GRU(epochs,batch_size,optimizer,samples_dim,fold,train_X,train_y,year,block_size,phenological,loss='mse'):
    model_type='GRU'
    inputs = Input(shape=(samples_dim[0], samples_dim[2]))
    if phenological:
        phen_str = 'XT'
    else:
        phen_str = 'X'
    x = GRU(60, activation='relu', return_sequences=True)(inputs)
    #x = Dropout(0.01)(x)
    x = GRU(30, activation='relu', return_sequences=True)(x)
    #x = Dropout(0.01)(x)
    x = GRU(30, activation='relu', return_sequences=True)(x)
    #x = Dropout(0.01)(x)
    x = GRU(10, activation='relu')(x)
    #x = Dropout(0.01)(x)
    x = Dense(20, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
    checkpoint = ModelCheckpoint('/Users/andres/Documents/strawberry-forecasting/models/'+'best_GRU'+'_'+str(year)+'_'+str(block_size)+'_'+str(samples_dim[0])+str(samples_dim[1])+'_'+str(fold)+'_'+phen_str+'.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    history = model.fit(train_X, train_y, shuffle=False, epochs=epochs, batch_size=batch_size, validation_split=0.25, callbacks=[checkpoint])
    return model_type
    