import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
import pickle
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
import keras.backend as K
from keras.utils import to_categorical
from keras.layers import LeakyReLU, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Dense, Input, Lambda, Conv2D, Conv2DTranspose, RepeatVector, Dropout, Flatten, Activation, BatchNormalization, regularizers, UpSampling2D, Concatenate
from keras.layers import Activation, ZeroPadding2D, concatenate, add

number_of_classes = 100

def resnet_block(input_tensor):
    ## Defining the Resnet Block
    x = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)

    x = add([x,input_tensor])
    return x

class GATH():
    def __init__(self):
        optimizer = keras.optimizers.Adam(lr=1e-4)
        self.auc = keras.models.load_model('../action_unit_model.h5')
        self.auc.name='auc'
        for layer in self.auc.layers:    
            layer.trainable=False
        
        self.discriminator = self.discriminator_network()
        self.discriminator.compile(loss = discriminator_loss, optimizer = optimizer)
        self.generator = self.generator_network()
        face = Input(shape = (100,100,3))
        au_coeff = Input(shape = (25,25,18))
        generated = self.generator([face,au_coeff])
        self.discriminator.trainable=False
        valid = self.discriminator(generated)
        auc_gen = self.auc(generated)
        self.combined = Model([face, au_coeff], [valid, generated, auc_gen])
        self.combined.summary()
        self.combined.compile(loss = [combined_adv,combined_recon, 'mean_squared_error'], optimizer = optimizer)

    def generator_network(self):    
        face = Input(shape = (100,100,3))
        au_coeff = Input(shape = (25,25,18))
        ## Layer 1
        
        x = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding='same', activation=None)(face)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ## Layer 2
        x = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ## Layer 3
        x = Conv2D(256, kernel_size=(5,5), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ## Layer 4
        x = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ############# Concatenate block of AU coefficients
        x = concatenate([x, au_coeff], axis=-1)
        ## Layer 5
        x = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        ## Layer 6 (6 Residual Blocks)
        x = resnet_block(x)
        x = resnet_block(x)
        x = resnet_block(x)
        x = resnet_block(x)
        x = resnet_block(x)
        x = resnet_block(x)

        ## Layer 7 (Upsampling Blocks)
        x = Conv2DTranspose(128, kernel_size=(5,5), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2DTranspose(128, kernel_size=(5,5), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(3, kernel_size=(5,5), strides=(1,1), padding='same', activation=None)(x)
        x = Activation('tanh')(x)

        model = Model(inputs = [face, au_coeff], outputs = [x])
        model.summary()
        return model

    def discriminator_network(self):
        face = Input(shape=(100, 100, 3))
        ## Layer 1
        x = Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='same', activation=None, input_shape=(100,100,3))(face)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ## Layer 2
        x = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ## Layer 3
        x = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ## Layer 4
        x = Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        ## Layer 5
        x = Conv2D(512, kernel_size=(3,3), strides=(2,2), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Flatten()(x)

        x = Dense(1024,activation=None)(x)
        x = LeakyReLU(alpha=0.1)(x)

        c = Dense(number_of_classes,activation=None)(x)
        c = Activation('softmax')(c)

        rf = Dense(1,activation=None)(x)
        rf = Activation('softmax')(rf)

        x = concatenate([c,rf])
        model = Model(inputs = [face], outputs = [x])
        model.summary()
        return model

    def train(self, epochs, data_gen):
        valid = np.ones((512, 1))
        G_LOSS0 = []
        G_LOSS1 = []
        G_LOSS2 = []
        G_LOSS3 = []
        D_LOSS = []
        fake = np.zeros((512, 1))
        for epoch in range(epochs):
            data_gen = createGenerator(image_gen, target_gen, auc)

            for batch_no in range(10):
                X,Y = next(data_gen)
                gen_X = self.generator.predict([X[0], X[2]])

                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(X[0], np.hstack((Y[0],valid)))
                d_loss_fake = self.discriminator.train_on_batch(gen_X, np.hstack((Y[0],fake)))
                d_loss_real = self.discriminator.train_on_batch(X[0], np.hstack((Y[0],valid)))
                d_loss_fake = self.discriminator.train_on_batch(gen_X, np.hstack((Y[0],fake)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                self.discriminator.trainable = False

                g_loss = self.combined.evaluate([X[0],X[2]], [np.hstack((Y[0],valid)),Y[1], Y[2]])            
                print ("Epoch : %d [D loss: %f] [G loss total: %f, G loss adverserial: %f, G loss reconstruction: %f, G loss au: %f]" % (epoch, d_loss, g_loss[0], g_loss[1], g_loss[2],g_loss[3]))
                D_LOSS.append(d_loss)
                G_LOSS0.append(g_loss[0])
                G_LOSS1.append(g_loss[1])
                G_LOSS2.append(g_loss[2])
                G_LOSS3.append(g_loss[3])
            
            g = gen_X[0,:,:,:]
            g = (g - g.min())/(g.max()-g.min())
            plt.imshow(g)
            plt.savefig('gens/' + str(epoch) + '.jpg')
            plt.close()
            plt.plot(D_LOSS)
            plt.savefig('plots/' + str(epoch) + 'd_loss.jpg')
            plt.close()
            plt.plot(G_LOSS0, label='Total')
            plt.plot(G_LOSS1, label='Adverserial')
            plt.plot(G_LOSS2, label='Reconstruction')
            plt.plot(G_LOSS3, label='AU loss')
            plt.legend()
            plt.savefig('plots/' + str(epoch) + 'g_loss.jpg')
            plt.close()
        self.discriminator.save('models/D' + str(epoch) + '.md5')
        self.generator.save('models/G' + str(epoch) + '.md5')
        self.combined.save('models/C' + str(epoch) + '.md5')


def discriminator_loss(y_true, y_pred):
    true_real = y_true[0][-1]
    pred_real = y_pred[0][-1]
    loss = 0.05 * K.square(pred_real - true_real)
    loss += true_real * 0.05 * keras.losses.binary_crossentropy(y_true[0][:-1], y_pred[0][:-1])
    return loss

def combined_adv(y_true, y_pred):
    Ladv = K.mean(K.square(y_pred[-1]))
    Lcls = keras.losses.binary_crossentropy(y_true[:,:-1], y_pred[:,:-1])
    return 0.05 * Lcls - 0.05 * Ladv

def combined_recon(y_true, y_pred):
    Lrec = keras.losses.mean_absolute_error(y_true, y_pred)
    # Lrec = K.mean(K.sum(K.sum(K.sum(K.abs(y_pred - y_true),axis=1),axis=1),axis=1))
    Ltv = K.mean(tf.image.total_variation(y_pred)/30000)
    return Lrec + Ltv


def createGenerator(image_gen, target_gen, auc):
    while True:
        idx = np.random.permutation(df.shape[0])
        df_df = df.iloc[idx]
        batches = image_gen.flow_from_dataframe(df_df, x_col = 'face_path',
                                    y_col = 'name',
                                    directory = '/scratch/aditya/cv_data/faces_dataset/',
                                    target_size = (100,100),
                                    color_mode = 'rgb',
                                    batch_size =512,
                                    drop_duplicates = False,
                                    shuffle = False)
        target_batches = target_gen.flow_from_dataframe(df_df, x_col = 'target_face',
                                    y_col = 'target_name',
                                    directory = '/scratch/aditya/cv_data/ravdess_faces/',
                                    target_size = (100,100),
                                    color_mode = 'rgb',
                                    batch_size = 512,
                                    drop_duplicates = False,
                                    # class_mode = 'other',
                                    shuffle = False)
        
        idx0 = 0
        for batch,tbatch in zip(batches,target_batches):
            idx1 = idx0 + batch[0].shape[0]
            au_coeff = df['au'].values[idx[idx0:idx1]]
            au_final = np.zeros((512,25,25,18))
            for i in range(512):
                au_final[i,:,:,:] = au_coeff[i]
            fau_real = auc.predict(tbatch[0])
            yield [batch[0], tbatch[0], au_final], [batch[1], batch[0], fau_real]
            idx0 = idx1
            if idx1 >= df.shape[0]:
                break

if __name__ == '__main__':
    df = pd.read_csv('final_df_small.csv')
    df['target_face'] = df['target_face'].apply(lambda x: x[:-3]+'bmp')
    df['target_name'] = df['target_face'].apply(lambda x: x[38:46])
    df['target_face'] = df['target_face'].apply(lambda x: x[38:])
    df['au'] = df['au'].apply(lambda x: eval(x))
    
    auc = keras.models.load_model('../action_unit_model.h5')
    for layer in auc.layers:    
        layer.trainable=False
    
    image_gen = ImageDataGenerator(rescale=1./255)
    target_gen = ImageDataGenerator(rescale=1./255)
    data_gen = createGenerator(image_gen, target_gen, auc)
    
    # gen.compile(loss = generator_loss, optimizer = optimizer)
    gath = GATH()
    gath.train(100, data_gen)



