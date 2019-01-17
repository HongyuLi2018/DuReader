"""Descriptions"""
import tensorflow as tf

INF = 1e30


def dropout(args, keep_prob, is_train, mode="recurrent"):
    """Descriptions"""
    if is_train:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(tf.less(keep_prob, 1.), lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def softmax_mask(val, mask):
    """Descriptions"""
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def summ(memory, hidden, mask, init=None, keep_prob=1.0, is_train=None, scope="summ"):
    """Descriptions"""
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        if init is None:
            s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        else:
            init = tf.tile(tf.expand_dims(init, axis=1), [1, tf.shape(memory)[1], 1])
            d_init = dropout(init, keep_prob=keep_prob, is_train=is_train)
            s0 = tf.nn.tanh(
                dense(d_memory, hidden, scope="s0") + dense(d_init, hidden, scope="s1")
            )
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res


def dense(inputs, hidden, use_bias=True, scope="dense"):
    """Descriptions"""
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)]
        out_shape += [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
