import numpy as np
from keras.utils import Sequence
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow as tf
from utils import uni_len


class TrainGenerator(Sequence):
    def __init__(self,
                 mel_files,
                 y_one_hot=None,
                 batch_size=64,
                 alpha=1,
                 datagen=None,
                 mel_data=None):

        self.mel_files = mel_files
        self.y_one_hot = y_one_hot
        self.batch_size = batch_size
        self.alpha = alpha
        self.datagen = datagen
        self.mel_data = mel_data
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

        this_batch_x1 = self.load_mels_for_batch([
                self.mel_files[i] for i in this_batch_indices
            ])
        this_batch_x2 = self.load_mels_for_batch([
                self.mel_files[i] for i in this_batch_mixup_indices
            ])
        this_batch_mixup_vals_x = this_batch_mixup_vals.reshape(
            this_batch_mixup_vals.shape[0], 1, 1, 1
        )
        this_batch_x = (this_batch_x1 * this_batch_mixup_vals_x) +\
            (this_batch_x2 * (1 - this_batch_mixup_vals_x))

        this_batch_y1 = self.y_one_hot[this_batch_indices, :]
        this_batch_y2 = self.y_one_hot[this_batch_mixup_indices, :]
        this_batch_mixup_vals_y = this_batch_mixup_vals.reshape(
            this_batch_mixup_vals.shape[0], 1
        )
        this_batch_y = (this_batch_y1 * this_batch_mixup_vals_y) +\
            (this_batch_y2 * (1 - this_batch_mixup_vals_y))

        return (this_batch_x, this_batch_y)


class ValGenerator(Sequence):
    def __init__(self,
                 mel_files,
                 y_one_hot,
                 batch_size=64,
                 mel_data=None):

        self.mel_files = mel_files
        self.y_one_hot = y_one_hot
        self.batch_size = batch_size
        self.mel_data = mel_data

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

        # create y array
        tmp = []
        for _ in range(6):
            tmp.append(self.y_one_hot[self.indexes, :])
        self.y_this_epoch = tmp

        # create x array(s)
        tmp = []
        for one_req_len in self.req_mel_len_list:
            self.req_mel_len = one_req_len
            tmp.append(self.load_mels_for_batch([
                  self.mel_files[i] for i in np.arange(len(self.mel_files))
            ]))
        self.x_this_epoch = tmp

    def __data_generation(self, batch_num):

        this_set = int(batch_num / self.one_set_size)
        this_index = batch_num % self.one_set_size
        this_indices = self.indexes[this_index*self.batch_size:(this_index+1)*self.batch_size]

        this_x = self.x_this_epoch[this_set][this_indices, :]
        this_y = self.y_this_epoch[this_set][this_indices, :]

        return (this_x, this_y)

def tf_one_sample_positive_class_precisions(y_true, y_pred) :
    num_samples, num_classes = y_pred.shape
    
    # find true labels
    pos_class_indices = tf.where(y_true > 0) 
    
    # put rank on each element
    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices
    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)
    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))
    sample_range = tf.transpose(sample_range)
    sample_range = tf.reshape(sample_range, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_classes = tf.reshape(retrieved_classes, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)
    retrieved_class_map = tf.transpose(retrieved_class_map)
    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))
    
    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)
    class_range = tf.add(class_range, tf.range(num_classes, delta=1))
    
    class_rankings = tf.scatter_nd(retrieved_class_map,
                                          class_range,
                                          tf.shape(y_pred))
    
    #pick_up ranks
    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)

    # add one for division for "presicion_at_hits"
    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1) 
    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)
    
    # generate tensor [num_sample, predict_rank], 
    # top-N predicted elements have flag, N is the number of positive for each sample.
    sample_label = pos_class_indices[:, 0]   
    sample_label = tf.reshape(sample_label, (-1, 1))
    sample_label = tf.cast(sample_label, tf.int32)
    
    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))
    retrieved_class_true_position = tf.concat((sample_label, 
                                               num_correct_until_correct), axis=1)
    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)
    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position, 
                                         retrieved_pos, 
                                         tf.shape(y_pred))
    # cumulate predict_rank
    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)

    # find positive position
    pos_ret_indices = tf.where(retrieved_class_true > 0)

    # find cumulative hits
    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)  
    correct_rank = tf.cast(correct_rank, tf.float32)

    # compute presicion
    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)

    return pos_class_indices, precision_at_hits

def tf_lwlrap(y_true, y_pred):
    num_samples, num_classes = y_pred.shape
    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))
    pos_flgs = tf.cast(y_true > 0, tf.int32)
    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)
    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),
                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))
    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)  
    class_label = pos_class_indices[:,1]
    sum_precisions_by_classes = tf.unsorted_segment_sum(precision_at_hits,
                                                        class_label,
                                                       num_classes)
    labels_per_class = tf.cast(labels_per_class, tf.float32)
    labels_per_class = tf.add(labels_per_class, 1e-7)
    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,
                                  tf.cast(labels_per_class, tf.float32))
    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)
    return out


def create_mel_model():
    mn = MobileNetV2(include_top=False)
    mn.layers.pop(0)
    inp = Input(shape=(64, None, 1))
    x = BatchNormalization()(inp)
    x = Conv2D(10, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(3, kernel_size=(1, 1), padding='same', activation='relu')(x)
    mn_out = mn(x)
    x = GlobalAveragePooling2D()(mn_out)
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(384, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(80, activation='softmax')(x)
    model = Model(inputs=[inp], outputs=x)
    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=0.0001),
                  metrics=[tf_lwlrap])
    return model
