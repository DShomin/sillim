import numpy as np
from keras.utils import Sequence
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
from utils import uni_len


class TrainGenerator(Sequence):
    def __init__(self,
                 mel_files,
                 pca_data,
                 y_one_hot,
                 batch_size=64,
                 alpha=0.5,
                 datagen=None,
                 mel_data=None):

        self.mel_files = mel_files
        self.y_one_hot = y_one_hot
        self.batch_size = batch_size
        self.alpha = alpha
        self.datagen = datagen
        self.mel_data = mel_data
        self.pca_data = pca_data
        self.on_epoch_end()

    def load_one_mel(self, filename):
        x = self.mel_data[filename].copy()
        x = uni_len(x, self.req_mel_len)
        x = x[..., np.newaxis]
        if self.datagen is not None:
            x = self.datagen.random_transform(x)
        return x

    def load_mels_for_batch(self, filelist):
        this_batch_data = [self.load_one_mel(x) for x in filelist]
        return np.array(this_batch_data)

    def __len__(self):
        return int(np.ceil(len(self.mel_files) / self.batch_size))

    def on_epoch_end(self):
        # initialize the indices
        self.indices = np.arange(len(self.mel_files))
        self.mixup_indices = np.arange(len(self.mel_files))

        # shuffle the indices
        np.random.shuffle(self.indices)
        np.random.shuffle(self.mixup_indices)

        # sample points for mixup
        self.mixup_vals = np.random.beta(self.alpha, self.alpha, len(self.mel_files))

    def __getitem__(self, index):
        this_batch_indices = self.indices[
            (index * self.batch_size):((index + 1) * self.batch_size)]
        this_batch_mixup_indices = self.mixup_indices[
            (index * self.batch_size):((index + 1) * self.batch_size)]
        this_batch_mixup_vals = self.mixup_vals[
            (index * self.batch_size):((index + 1) * self.batch_size)]

        return self.__data_generation(this_batch_indices,
                                      this_batch_mixup_indices,
                                      this_batch_mixup_vals)

    def __data_generation(self,
                          this_batch_indices,
                          this_batch_mixup_indices,
                          this_batch_mixup_vals):

        self.req_mel_len = np.random.randint(263, 763)

        this_batch_mel1 = self.load_mels_for_batch([
                self.mel_files[i] for i in this_batch_indices
            ])
        this_batch_mel2 = self.load_mels_for_batch([
                self.mel_files[i] for i in this_batch_mixup_indices
            ])
        this_batch_mixup_vals_mel = this_batch_mixup_vals.reshape(
            this_batch_mixup_vals.shape[0], 1, 1, 1
        )
        this_batch_mel = (this_batch_mel1 * this_batch_mixup_vals_mel) +\
            (this_batch_mel2 * (1 - this_batch_mixup_vals_mel))

        this_batch_pca1 = self.pca_data[this_batch_indices, :]
        this_batch_pca2 = self.pca_data[this_batch_mixup_indices, :]
        this_batch_mixup_vals_pca = this_batch_mixup_vals.reshape(
            this_batch_mixup_vals.shape[0], 1
        )
        this_batch_pca = (this_batch_pca1 * this_batch_mixup_vals_pca) +\
            (this_batch_pca2 * (1 - this_batch_mixup_vals_pca))

        this_batch_y1 = self.y_one_hot[this_batch_indices, :]
        this_batch_y2 = self.y_one_hot[this_batch_mixup_indices, :]
        this_batch_mixup_vals_y = this_batch_mixup_vals.reshape(
            this_batch_mixup_vals.shape[0], 1
        )
        this_batch_y = (this_batch_y1 * this_batch_mixup_vals_y) +\
            (this_batch_y2 * (1 - this_batch_mixup_vals_y))

        return ({'mel': this_batch_mel,
                 'pca': this_batch_pca},
                this_batch_y)


class ValGenerator(Sequence):
    def __init__(self,
                 mel_files,
                 pca_data,
                 y_one_hot,
                 batch_size=64,
                 mel_data=None):

        self.mel_files = mel_files
        self.y_one_hot = y_one_hot
        self.batch_size = batch_size
        self.mel_data = mel_data
        self.pca_data = pca_data

        self.one_set_size = int(np.ceil(len(self.mel_files) / self.batch_size))

        self.req_mel_len_list = [263, 363, 463, 563, 663, 763]
        self.on_epoch_end()

    def load_one_mel(self, filename):
        x = self.mel_data[filename].copy()
        x = uni_len(x, self.req_mel_len)
        x = x[..., np.newaxis]
        return x

    def load_mels_for_batch(self, filelist):
        this_batch_data = [self.load_one_mel(x) for x in filelist]
        return np.array(this_batch_data)

    def __len__(self):
        return 6*self.one_set_size

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        # initialize the indices
        self.indexes = np.arange(len(self.mel_files))

        # create y array(s)
        tmp = []
        for _ in range(6):
            tmp.append(self.y_one_hot[self.indexes, :])
        self.y_this_epoch = tmp

        # create pca array(s)
        tmp = []
        for _ in range(6):
            tmp.append(self.pca_data[self.indexes, :])
        self.pca_this_epoch = tmp

        # create mel array(s)
        tmp = []
        for one_req_len in self.req_mel_len_list:
            self.req_mel_len = one_req_len
            tmp.append(self.load_mels_for_batch([
                  self.mel_files[i] for i in np.arange(len(self.mel_files))
            ]))
        self.mel_this_epoch = tmp

    def __data_generation(self, batch_num):

        this_set = int(batch_num / self.one_set_size)
        this_index = batch_num % self.one_set_size
        this_indices = self.indexes[this_index*self.batch_size:(this_index+1)*self.batch_size]

        this_batch_mel = self.mel_this_epoch[this_set][this_indices, :, :]
        this_batch_pca = self.pca_this_epoch[this_set][this_indices, :]
        this_batch_y = self.y_this_epoch[this_set][this_indices, :]

        return ({'mel': this_batch_mel,
                 'pca': this_batch_pca},
                this_batch_y)


def create_mel_and_pca_model():
    inp1 = Input(shape=(64, None, 1), name='mel')
    x = BatchNormalization()(inp1)
    x = Conv2D(10, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(3, kernel_size=(1, 1), padding='same', activation='relu')(x)

    mn = MobileNetV2(include_top=False)
    mn.layers.pop(0)

    mn_out = mn(x)
    x = GlobalAveragePooling2D()(mn_out)

    inp2 = Input(shape=(350,), name='pca')
    y = BatchNormalization()(inp2)

    x = concatenate([x, y], axis=-1)
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(384, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(41, activation='softmax')(x)

    model = Model(inputs=[inp1, inp2], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model
