import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras as K
import tensorflow.keras.backend as Kback

def SAM_avg(x):
    batch, _, _, channel = x.shape
    x = K.layers.Conv2D(channel//2, kernel_size=1, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = K.layers.Conv2D(channel//2, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = K.layers.BatchNormalization()(x)
    x = CAM(x)
    ## Average Pooling
    x1 = tf.reduce_mean(x, axis=-1)
    x1 = tf.expand_dims(x1, axis=-1)
    ## Conv layer
    feats = K.layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid", kernel_initializer=tf.keras.initializers.HeNormal())(x1)
    feats = K.layers.Multiply()([x, feats])
    return feats

def SAM_max(x):
    batch, _, _, channel = x.shape
    x = K.layers.SeparableConv2D(channel, kernel_size=1, padding="same")(x)
    x = K.layers.SeparableConv2D(channel, kernel_size=3, padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = CAM(x)
    ## Max Pooling
    x2 = tf.reduce_max(x, axis=-1)
    x2 = tf.expand_dims(x2, axis=-1)
    ## Conv layer
    feats = K.layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(x2)
    feats = K.layers.Multiply()([x, feats])
    return feats

def CSSAM(x):
    x_avg = SAM_avg(x)
    x_max = SAM_max(x)
    x = K.layers.Concatenate()([x_avg, x_max])
    x = ChannelDropout(drop_ratio=0.5)(x)
    return x

def CAM(x, ratio=8):
    batch, _, _, channel = x.shape
    ## Shared layers
    l1 = K.layers.Dense(channel//ratio, activation="relu", use_bias=False)
    l2 = K.layers.Dense(channel, use_bias=False)
    ## Global Average Pooling
    x1 = K.layers.GlobalAveragePooling2D()(x)
    x1 = l1(x1)
    x1 = l2(x1)
    ## Global Max Pooling
    x2 = K.layers.GlobalMaxPooling2D()(x)
    x2 = l1(x2)
    x2 = l2(x2)
    ## Add both the features and pass through sigmoid
    feats = x1 + x2
    feats = K.layers.Activation("sigmoid")(feats)
    feats = K.layers.Multiply()([x, feats])
    return feats

class ChannelDropout(K.layers.Layer):
    def __init__(self, drop_ratio=0.2):
        super(ChannelDropout, self).__init__()
        self.drop_ratio = drop_ratio

    def build(self, input_shape):
        _, _, _, self.channels = input_shape
        # Initialize a trainable mask with ones
        self.mask = RichardsSigmoid(units=1)(self.add_weight("mask", shape=(1, 1, 1, self.channels), initializer="ones", trainable=True))

    def call(self, x):
        # Duplicate the mask to match the batch size
        mask = tf.tile(self.mask, [tf.shape(x)[0], 1, 1, 1])
        # Multiply the input by the mask
        x = x * mask
        num_channels_to_keep = int(self.channels // 1.25)
        sorted_x, indices = tf.nn.top_k(x, k=num_channels_to_keep, sorted=True)
        sorted_x = sorted_x[:,:,:,0:num_channels_to_keep]
        return sorted_x
    
class RichardsSigmoid(K.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super(RichardsSigmoid, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialize learnable parameters: A, Q, mu
        self.A = self.add_weight(name='A', shape=(self.units,), initializer='uniform', trainable=True)
        self.Q = self.add_weight(name='Q', shape=(self.units,), initializer='uniform', trainable=True)
        self.mu = self.add_weight(name='mu', shape=(self.units,), initializer='uniform', trainable=True)

        super(RichardsSigmoid, self).build(input_shape)

    def call(self, x):
        # Richards sigmoid function
        return 1 / (1 + tf.exp(-self.A * tf.exp(-self.Q * (x - self.mu))))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)
    
input_layer = K.Input(shape=(256,256,3))

deep_learner = K.applications.DenseNet169(include_top = False, weights = "imagenet", input_tensor = input_layer)
for layer in deep_learner.layers:
    layer.trainable = True

input_img = K.layers.Input(shape=(256,256,3)) 
feat_img = deep_learner(input_img)
feat_img = CSSAM(feat_img)
flat = K.layers.GlobalAveragePooling2D()(feat_img)
flat = K.layers.Dropout(0.2)(flat)
output = K.layers.Dense(3, activation='softmax')(flat)

model = K.Model(inputs=input_img, outputs=output)
optimizer = K.optimizers.AdamW(lr=0.0001)
model.compile(loss=["categorical_crossentropy"], metrics=METRICS, optimizer = optimizer)
model.summary()
