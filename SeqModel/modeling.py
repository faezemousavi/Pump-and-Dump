#-*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DNN(object):

    def __init__(self, tensor_dict, model_config):
        self.label = tensor_dict["label"]
        self.channel_embedding = tensor_dict["channel_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]
        self.target_features = tensor_dict["target_features"]

        # configuration
        self.model_config = {
            "hidden1": 32,
            "hidden2": 16,
            "hidden3": 32,
            "learning_rate": 0.0005
        }
        self.model_config.update(model_config)
        print("Model Configuaration:")
        print(self.model_config)

    def build(self):
        """
        build the architecture for the base DNN model.
        """
        self.inp = self.target_features
        self.build_fcn_net(self.inp)
        self.loss_op()

    def build_fcn_net(self, inp):
        with tf.name_scope("Fully_connected_layer"):

            dnn1 = tf.layers.dense(inp, self.model_config["hidden1"], activation=tf.nn.relu, name='f1')
            dnn2 = tf.layers.dense(dnn1, self.model_config["hidden2"], activation=tf.nn.relu, name='f2')
            
            self.logit = tf.squeeze(tf.layers.dense(dnn2, 1, activation=None, name='logit'))

        self.y_hat = tf.sigmoid(self.logit)

        return

    def loss_op(self):
        with tf.name_scope('Metrics'):

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"]).minimize(self.loss)

        return
        
class GRUModel(DNN):
    def __init__(self, tensor_dict, model_config):
        super(GRUModel, self).__init__(tensor_dict, model_config)

        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:,:,-9:]

    def gru_layer(self):
        # Define GRU cell
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=self.model_config["hidden_gru_units"])

        # Dynamic RNN for variable length sequences
        outputs, _ = tf.nn.dynamic_rnn(gru_cell, self.seq_embedding, dtype=tf.float32, sequence_length=self.length)

        # Aggregate sequence representations (e.g., mean pooling)
        output_mean = tf.reduce_mean(outputs, axis=1)

        return output_mean

    def build(self):
        """
        Override the build function
        """
        self.seq_embedding_mean = self.gru_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean, self.coin_embedding], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()

        
class BiGRUModel(DNN):
    def __init__(self, tensor_dict, model_config):
        super(BiGRUModel, self).__init__(tensor_dict, model_config)

        self.seq_embedding = tensor_dict["seq_embedding"]
        self.length = tensor_dict["length"]

    def bigru_layer(self):
        # Define BiGRU cell
        gru_forward = tf.keras.layers.GRU(units=self.model_config["hidden_gru_units"], return_sequences=True)
        gru_backward = tf.keras.layers.GRU(units=self.model_config["hidden_gru_units"], return_sequences=True, go_backwards=True)
        
        # Bidirectional BiGRU layer
        bigru_output = tf.keras.layers.Bidirectional(gru_forward, backward_layer=gru_backward, name='bigru')(self.seq_embedding)

        return bigru_output

    def build(self):
        """
        Override the build function to define the model architecture
        """
        self.seq_embedding_bigru = self.bigru_layer()

        # Flatten the seq_embedding_bigru to have rank 2
        batch_size = tf.shape(self.seq_embedding_bigru)[0]
        seq_length = tf.shape(self.seq_embedding_bigru)[1]
        hidden_units = self.model_config["hidden_gru_units"] * 2  # Because of bidirectionality
        seq_embedding_flat = tf.reshape(self.seq_embedding_bigru, [batch_size, seq_length * hidden_units])

        # Concatenate tensors
        self.inp = tf.concat([self.target_features, seq_embedding_flat, self.coin_embedding], axis=1)

        # Build the fully connected layers
        self.build_fcn_net(self.inp)

        # Define the loss operation
        self.loss_op()


class LSTMModel(DNN):
    def __init__(self, tensor_dict, model_config):
        super(LSTMModel, self).__init__(tensor_dict, model_config)

        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:, :, -9:]

    def lstm_layer(self):
        # Define LSTM cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.model_config["hidden_lstm_units"])

        # Dynamic RNN for variable length sequences
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, self.seq_embedding, dtype=tf.float32, sequence_length=self.length)

        # Aggregate sequence representations (e.g., mean pooling)
        output_mean = tf.reduce_mean(outputs, axis=1)

        return output_mean

    def build(self):
        """
        Override the build function
        """
        self.seq_embedding_mean = self.lstm_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean, self.coin_embedding], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()
        
class BiLSTMModel(DNN):
    def __init__(self, tensor_dict, model_config):
        super(BiLSTMModel, self).__init__(tensor_dict, model_config)
        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:, :, -9:]

    def bilstm_layer(self):
        lstm_units = self.model_config["hidden_lstm_units"]
        num_lstm_layers = self.model_config["num_lstm_layers"]

        # Define forward LSTM cell
        lstm_forward = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)

        # Define backward LSTM cell
        lstm_backward = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True, go_backwards=True)

        # Apply Bidirectional wrapper to LSTMs
        bilstm_output = tf.keras.layers.Bidirectional(lstm_forward, backward_layer=lstm_backward, name='bilstm')(self.seq_embedding)

        # Aggregate sequence representations (e.g., mean pooling)
        output_mean = tf.reduce_mean(bilstm_output, axis=1)

        return output_mean

    def build(self):
        self.seq_embedding_bilstm = self.bilstm_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_bilstm, self.coin_embedding], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()
        
class TCNModel(DNN):
    def __init__(self, tensor_dict, model_config):
        super(TCNModel, self).__init__(tensor_dict, model_config)
        
        self.seq_embedding = tensor_dict["seq_embedding"]  # Assuming this is your input sequence data

    def tcn_layer(self):
        # TCN configuration
        num_filters = 64
        num_layers = 3
        kernel_size = 3
        dropout_rate = self.model_config["dropout_rate"]

        # Define TCN layers
        tcn = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=1, activation='relu')(self.seq_embedding)
        for i in range(1, num_layers):
            tcn = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=2**i, activation='relu')(tcn)

        tcn = tf.keras.layers.Dropout(dropout_rate)(tcn)

        # Optional: Add pooling layer or additional convolutional layers here
        # Example: tcn = tf.keras.layers.MaxPooling1D(pool_size=2)(tcn)

        return tcn

    def build(self):
        """
        Override the build function
        """
        # Assuming you flatten self.seq_embedding_tcn
        seq_embedding_tcn_flat = tf.reshape(self.seq_embedding, [-1, 50 * 64])  # Flatten to [batch_size, 50 * 64]

        # Concatenate the tensors
        self.inp = tf.concat([self.target_features, seq_embedding_tcn_flat, self.coin_embedding], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()

class TransferLearningModel(DNN):
    def __init__(self, tensor_dict, model_config):
        super(TransferLearningModel, self).__init__(tensor_dict, model_config)
        
        self.tensor_dict = tensor_dict  # Store the entire tensor_dict
        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:,:,-9:]
        self.seq_coin_embedding = tensor_dict["seq_coin_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]
        self.is_training = model_config.get("is_training", True)
        self.checkpoint_path = model_config.get("checkpoint_path")

    def build_initial_model(self):
        # Define the initial model architecture
        with tf.variable_scope("initial_model"):
            # Process sequence embedding
            seq_flat = tf.reshape(self.seq_embedding, [-1, 50 * 9])  # Assuming max sequence length of 50
            seq_hidden = tf.layers.dense(seq_flat, units=64, activation=tf.nn.relu, name="seq_hidden")

            # Process sequence coin embedding
            seq_coin_flat = tf.reshape(self.seq_coin_embedding, [-1, 50 * self.seq_coin_embedding.shape[-1]])
            seq_coin_hidden = tf.layers.dense(seq_coin_flat, units=64, activation=tf.nn.relu, name="seq_coin_hidden")

            # Process coin embedding
            coin_hidden = tf.layers.dense(self.coin_embedding, units=32, activation=tf.nn.relu, name="coin_hidden")

            # Combine all features
            combined = tf.concat([seq_hidden, seq_coin_hidden, coin_hidden], axis=1)
            self.initial_output = tf.layers.dense(combined, units=128, activation=tf.nn.relu, name="initial_output")

    def build_final_layers(self):
        # Add new layers on top of the initial model
        with tf.variable_scope("final_layers"):
            hidden = tf.layers.dense(self.initial_output, units=64, activation=tf.nn.relu, name="hidden")
            self.final_layer = tf.layers.dense(hidden, units=1, activation=None, name="final_layer")
            self.logit = tf.squeeze(self.final_layer)
            self.y_hat = tf.sigmoid(self.logit)

    # def loss_op(self):
    #     # Define loss
    #     self.label = self.tensor_dict['label']
    #     self.loss = tf.losses.log_loss(self.label, self.y_hat)
        
    #     # Define optimizer
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"])
    #     self.train_op = self.optimizer.minimize(self.loss)
    def loss_op(self):
      self.label = self.tensor_dict['label']
      self.loss = tf.losses.log_loss(self.label, self.y_hat)
      
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"])
      self.train_op = self.optimizer.minimize(self.loss)

    def build(self):
        """
        Build the model architecture.
        """
        # Build the initial model
        self.build_initial_model()
        
        # Build the final layers
        self.build_final_layers()
        
        # Define loss and optimizer
        self.loss_op()


class SNN(DNN):
    def __init__(self, tensor_dict, model_config):
        super(SNN, self).__init__(tensor_dict, model_config)

        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:,:,-9:]
        self.seq_coin_embedding = tensor_dict["seq_coin_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]
    def positional_attention_layer(self):

        feat_num = self.seq_embedding.shape[2]
        self.length = tf.where(tf.less_equal(self.length, self.model_config["max_seq_length"]),
                               self.length, tf.Variable([self.model_config["max_seq_length"] for i in range(self.model_config["batch_size"])]))

        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")

        mask_2d = tf.expand_dims(self.sequence_mask, axis=2)
        masked_seq_embedding = self.seq_embedding * tf.cast(mask_2d, tf.float32) # convert bool to float

        # use f(x) to speed training
        pos_atten_list = []
        seq_pos_attent_embed_list = []
        num_head = 8

        for i in range(feat_num):

            x = tf.expand_dims(masked_seq_embedding[:, :, i], axis=2)
            inp = layer_norm(x, name="layer_norm_" + str(i)) # add layer norm

            # positional attention
            pos_atten = tf.get_variable("pos_atten_" + str(i), [1, 50, num_head],
                                        initializer = tf.truncated_normal_initializer(stddev=0.02))
            pos_atten = tf.layers.dense(pos_atten, 8, activation=tf.nn.relu, name="pos_" + str(i) + "_fx1")
            pos_atten = tf.layers.dense(pos_atten, num_head, activation=None, name="pos_" + str(i) + "_fx2")

            pos_atten = tf.tile(pos_atten, [self.model_config["batch_size"], 1, 1])
            paddings = tf.ones_like(pos_atten) * (-2 ** 32 + 1)

            mask_new = tf.tile(mask_2d, [1,1, num_head])
            pos_atten = tf.where(tf.logical_not(mask_new), paddings, pos_atten)
            pos_atten = tf.nn.softmax(pos_atten, axis=1)

            # inp = tf.expand_dims(self.masked_seq_embedding[:, :, i], axis=2)
            seq_pos_attent_embed = tf.reduce_sum(pos_atten * inp, axis=1)
            seq_pos_attent_embed_list.append(seq_pos_attent_embed)
            pos_atten_list.append(tf.expand_dims(pos_atten, axis=2))

        self.pos_atten = tf.concat(pos_atten_list, axis=2)
        self.seq_pos_attent_embeding = tf.concat(seq_pos_attent_embed_list, axis=1)

        # for coin embedding
        coin_mask_2d = tf.expand_dims(self.sequence_mask, axis=2)
        masked_seq_coin_embedding = self.seq_coin_embedding * tf.cast(coin_mask_2d, tf.float32)  # convert bool to float
        coin_inp = layer_norm(masked_seq_coin_embedding, name="layer_norm_coin_embedding")  # add layer norm

        num_head = 8
        pos_atten_coin = tf.get_variable("pos_atten_coin", [1, 50, num_head],
                                          initializer=tf.truncated_normal_initializer(stddev=0.02))

        pos_atten_coin = tf.layers.dense(pos_atten_coin, num_head, activation=tf.nn.relu, name="pos_coin_fx1")
        pos_atten_coin = tf.layers.dense(pos_atten_coin, num_head, activation=None, name="pos_coin_fx2")

        pos_atten_coin = tf.tile(pos_atten_coin, [self.model_config["batch_size"], 1, 1])
        paddings = tf.ones_like(pos_atten_coin) * (-2 ** 32 + 1)

        coin_mask_new = tf.tile(coin_mask_2d, [1, 1, num_head])
        pos_atten_coin = tf.where(tf.logical_not(coin_mask_new), paddings, pos_atten_coin)
        pos_atten_coin = tf.nn.softmax(pos_atten_coin, axis=1)

        seq_coin_pos_attent_embeding = tf.reduce_sum(tf.expand_dims(pos_atten_coin, axis=2) * tf.expand_dims(coin_inp, axis=3), axis=1)
        self.seq_coin_pos_attent_embeding = tf.reshape(seq_coin_pos_attent_embeding, [self.model_config["batch_size"], -1])
        output = tf.concat([self.seq_pos_attent_embeding, self.seq_coin_pos_attent_embeding], axis=1)

        return output

    def build(self):
        """
        override the build function
        """
        self.seq_embedding_mean = self.positional_attention_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean, self.coin_embedding], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()

class SNNTA(DNN):

    def __init__(self, tensor_dict, model_config):
        super(SNNTA, self).__init__(tensor_dict, model_config)

        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:,:,-9:]
        self.seq_coin_embedding = tensor_dict["seq_coin_embedding"]

        self.atten_strategy = "mlp"
        # self.atten_strategy = "dot"
    def positional_attention_layer(self):

        self.length = tf.where(tf.less_equal(self.length, self.model_config["max_seq_length"]), self.length,
                               tf.Variable([self.model_config["max_seq_length"] for i in range(self.model_config["batch_size"])]))
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")

        query = self.target_features[:,-9:]
        key = self.seq_embedding

        if self.atten_strategy == "mlp":
            query = tf.expand_dims(query, axis=1)
            queries = tf.tile(query, [1, key.shape[1], 1])
            attention_all = tf.concat([queries, key, queries - key, queries * key], axis=-1)

            d_layer_1_all = tf.layers.dense(attention_all, 32, activation=tf.nn.relu, name="f1_att")
            d_layer_2_all = tf.layers.dense(d_layer_1_all, 16, activation=tf.nn.relu, name="f2_att")
            d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name="f3_att")
            d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, key.shape[1]])

            scores = d_layer_3_all
            key_masks = tf.expand_dims(self.sequence_mask, 1)

            paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
            scores = tf.where(key_masks, scores, paddings)
            scores = tf.nn.softmax(scores, axis=2)
            output = tf.squeeze(tf.matmul(scores, key))

        elif self.atten_strategy == "dot":
            attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                     keys=key,
                                                     values=key,
                                                     num_heads=8,
                                                     key_masks=self.sequence_mask,
                                                     dropout_rate=self.model_config["dropout_rate"],
                                                     is_training=self.model_config["is_training"],
                                                     reuse=False)
            output = tf.squeeze(attended_embedding, axis=1)

        else:
            raise ValueError("Please choose the right attention strategy. Current is " + self.atten_strategy)

        return layer_norm(output, "layer_norm")

    def build(self):
        """
        override the build function
        """
        self.seq_embedding_mean = self.positional_attention_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()



def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    epsilon = 1e-6
    filters = input_tensor.get_shape()[-1]
    with tf.variable_scope(name):
        scale = tf.get_variable("layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [filters], initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(input_tensor, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=-1, keep_dims=True)
        input_tensor = (input_tensor - mean) * tf.rsqrt(variance + epsilon)
        input_tensor = input_tensor * scale + bias

        return input_tensor

def multihead_attention(queries,
                        keys,
                        values,
                        key_masks,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      values: A 3d tensor with shape of [N, T_v, C_v]
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.logical_not(key_masks), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

    return outputs
