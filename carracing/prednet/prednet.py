import os
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

def add_upscale(X):
    prev_shape = X.get_shape()
    size = [2 * int(s) for s in prev_shape[1:3]]
    return tf.image.resize_nearest_neighbor(X, size)

class PredNet():
    def __init__(self, batch_size, time_step, 
                    layer_loss_weights=np.array([1., 0, 0, 0], np.float32) ,
                    img_shape=(128, 160, 3),
                    stack_sizes=(3, 48, 96, 192), 
                    R_stack_sizes=(3, 48, 96, 192), 
                    A_filter_sizes=(3, 3 ,3), 
                    Ahat_filter_sizes=(3, 3, 3, 3), 
                    R_filter_sizes=(3, 3, 3, 3), 
                    pixel_max=1, 
                    output_mode='all', 
                    extrap_start_time=None):
        self.batch_size = batch_size
        self.time_step = time_step
        self.layer_loss_weights = layer_loss_weights
        self.img_shape = img_shape
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        self.R_stack_sizes = R_stack_sizes
        self.A_filter_sizes = A_filter_sizes
        self.Ahat_filter_sizes = Ahat_filter_sizes
        self.R_filter_sizes = R_filter_sizes
        self.pixel_max = pixel_max
        self.output_mode = output_mode
        
        default_output_mode = ['predition', 'error', 'all']
        layer_output_mode = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]

        if self.output_mode in default_output_mode:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None

        img_w, img_h, img_c = self.img_shape
        self.frame_shape = [self.batch_size, 1, img_h, img_w, img_c]
        self.error_shape = [self.batch_size, 1, self.nb_layers]
        self.input_shape = [self.batch_size, self.time_step, img_h, img_w, img_c]
        self.build_model()

    def build_model(self, hps):
        self.inputs = tf.placeholder(tf.float32, self.input_shape)
        frame_predictions, errors = self.forward(self.inputs)

        errors = tf.concat(axis=1, values=errors) # [b, t, nb_layers]
        self.frame_predictions = tf.concat(axis=1, values=frame_predictions) # [b, t, h, w, c]


        layer_loss_weights = np.expand_dims(self.layer_loss_weights, 1)
        time_loss_weights = 1. / (self.time_step - 1) * np.ones((self.time_step, 1))
        time_loss_weights[0] = 0
        time_loss_weights = np.array(time_loss_weights, np.float32)

        errors_ = tf.reshape(errors, [-1, self.nb_layers])
        errors_by_time = tf.matmul(errors_, layer_loss_weights) # [b * t, 1]
        errors_by_time = tf.reshape(errors_by_time, (self.batch_size, self.time_step))
        errors_by_time = errors[:, :, 0]
        final_error = flatten(tf.matmul(errors_by_time, time_loss_weights)) # [b]
        final_error = tf.reduce_mean(final_error)

        # training operation
        self.error = final_error
        self.loss_sum = tf.summary.scalar("error", self.error)
        self.t_vars = tf.trainable_variables()

        num_param = 0.0
        for var in self.t_vars:
            num_param += int(np.prod(var.get_shape()))
        print("Number of paramers: %d"%num_param)
        self.saver = tf.train.Saver(max_to_keep = 10)

    def forward(self, inputs):
        """
        inputs : [batch_size, t, h, w, c]
            batch_size : batch datasize
            t : time step (frame)
            h : image height size
            w : image width size
            c : image channel size
        """
        states = self.get_initial_state()
        errors = []
        frame_predictions = []
        t = inputs.get_shape().as_list()[1]

        reuse_step = False
        for ti in range(t):
            a = inputs[:, ti]
            output, states = self.step(a, states, reuse_step=reuse_step)
            frame_predictions.append(tf.reshape(output[0], self.frame_shape)) 
            errors.append(tf.reshape(output[1], self.error_shape)) 
            reuse_step = True

        return frame_predictions, errors

    def get_initial_state(self):
        initial_states = []
        img_h, img_w, img_c = self.img_shape
        for u in ["r", "c", "e"]:
            for l in range(self.nb_layers):
                ds_factor = 2 ** l
                if u in ['r', 'c']:
                    stack_size = self.R_stack_sizes[l]
                elif u == 'e':
                    stack_size = 2 * self.stack_sizes[l]
                output_size = stack_size * img_h * img_w
                initial_state = tf.zeros((batch, output_size))
                output_shape = (self.batch_size, img_h, img_w, stack_size)
                initial_state = tf.reshape(initial_state, output_shape)
                initial_states += [initial_state]
        return initial_states

    def step(self, a, states, reuse_step, scope_step='one_step'):        
        r_tm1 = states[: self.nb_layers]
        c_tm1 = states[self.nb_layers: 2 * self.nb_layers]
        e_tm1 = states[2 * self.nb_layers: 3 * self.nb_layers]
        r = []
        c = []
        e = []
        with tf.variable_scope(scope_step) as scope:
            if reuse_step:
                scope.reuse_variables()
            for l in reversed(range(self.nb_layers)):
                inputs = [r_tm1[l], e_tm1[l]]
                if l < self.nb_layers - 1:
                    inputs.append(r_up)
                inputs = tf.concat(inputs, axis=-1)
                new_c, new_r = self.convlstm(inputs, l, c_tm1[l], 'lstm' + str(l))
                c.insert(0, new_c)
                r.insert(0, new_r)
                if l > 0:
                    r_up = add_upscale(new_r)


            for l in range(self.nb_layers):
                with tf.variable_scope("conv_ahat"+str(l)):
                    input_ = r[l]
                    k_h = 3
                    k_w = 3
                    in_ch = input_.get_shape()[-1]
                    out_ch = self.stack_sizes[l]
                    w = tf.get_variable("weights", [k_h, k_w, in_ch, out_ch], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
                    b = tf.get_variable('biases', [out_ch], initializer=tf.constant_initializer(0.0))
                    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME')
                    conv = tf.nn.bias_add(conv, b)
                    ahat = tf.nn.relu(conv)

                if l == 0:
                    ahat = tf.minimum(ahat, self.pixel_max)
                    frame_prediction = ahat

                e_up = tf.nn.relu(ahat - a)
                e_down = tf.nn.relu(a - ahat)
                e.append(tf.concat([e_up, e_down], axis=-1))

                if self.output_layer_num == l:
                    if self.output_layer_type == 'A':
                        output = a
                    elif self.output_layer_type == 'Ahat':
                        output = ahat
                    elif self.output_layer_type == 'r':
                        output = r[l]
                    elif self.output_layer_type == 'e':
                        output = e[l]

                if l < self.nb_layers - 1:
                    with tf.variable_scope("conv_a"+str(l)):
                        input_ = e[l]
                        k_h = 3
                        k_w = 3
                        in_ch = input_.get_shape()[-1]
                        out_ch = self.stack_sizes[l+1]
                        w = tf.get_variable("weights", [k_h, k_w, in_ch, out_ch], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
                        b = tf.get_variable("biases", [out_ch], initializer=tf.constant_initializer(0.0))
                        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME')
                        conv = tf.nn.bias_add(conv, b)
                        a = tf.nn.relu(conv)
                        a = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            if self.output_layer_type is None:
                if self.output_mode == 'prediction':
                    output = frame_prediction
                else:
                    for l in range(self.nb_layers):
                        layer_error = tf.reduce_mean(flatten(e[l]), axis=-1, keep_dims=True)
                        if l == 0:
                            all_error = layer_error
                        else:
                            all_error = tf.concat([all_error, layer_error], axis=-1)

                    if self.output_mode == 'error':
                        output = all_error
                    else:
                        output = [frame_prediction, all_error]

            states = r + c + e
            return output, states

    def convlstm(self, inputs, li, c, scope_name="conv_lstm", reuse=False):
         with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            concat = conv2d(inputs, self.R_stack_sizes[li] * 4, self.R_filter_sizes[li], self.R_filter_sizes[li], name='lstm'+str(l))
            i, z, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)
            new_c = c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) * tf.nn.tanh(z)
            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
            return new_c, new_h
             
    def save(self, sess, checkpoint_dir, step):
        model_name = "PredNet"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: 
                model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: "+str(model_name))
            return True, model_name
        else:
            return False, None


