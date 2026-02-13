import tensorflow as tf
import numpy as np
from tensorflow import keras 
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, GlobalAveragePooling1D, Concatenate
from abc import ABC, abstractmethod
from tensorflow.keras.models import Model
import mlflow




class BuildModel:
    def build(self, vocab_size, max_sub_len, max_body_len):
        subject = Input(shape=(max_sub_len, ))
        body = Input(shape=(max_body_len, ))

        subject_emb = Embedding(vocab_size, 128)(subject)
        body_emb = Embedding(vocab_size, 128)(body)

        pool_sub = GlobalAveragePooling1D()(subject_emb)
        pool_body = GlobalAveragePooling1D()(body_emb)

        merged = Concatenate()([pool_sub, pool_body])

        x = Dense(128, activation='relu')(merged)
        x = Dropout(0.5)(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)

        output = Dense(6, activation="softmax")(x)  

        model = Model(inputs =[subject, body], outputs=output)

        return model
    

class TrainModel:
    def train(self, model, data, epoch=10, batch_size=64):
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        X_sub = np.stack(data['subject_seq'].values)
        X_body = np.stack(data['body_seq'].values)

        history = model.fit([X_sub, X_body], data['target'], epochs=epoch, batch_size=batch_size)
        
        return history.history['loss'][-1]