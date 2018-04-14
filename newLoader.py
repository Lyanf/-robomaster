import tensorflow as tf
# h_fc1_drop, W_fc2) + b_fc2
# dropout/mul:0 Variable_6:0 Variable_7:0
# saver = tf.train.import_meta_graph('./new/model.meta')
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./new/model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./new'))
    print(sess.run('dropout/mul:0'))
    print(sess.run('Variable_6:0'))
    print(sess.run('Variable_7:0'))