import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data

mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# trainImages = mnist.train.images
# trainLabels = mnist.train.labels
# # 要通过这个把图片数据输入进去（数据是固定的）
# imagePlaceholder = tf.placeholder(tf.float32,[None,784])
# answerPlaceholder = tf.placeholder(tf.float32,[None,10])
# # 每个点对于每种可能情况的权重，要调教的参数
# weight = tf.Variable(tf.zeros([784,10]))
# #每种可能情况的误差，要调教的参数
# errorr = tf.Variable(tf.zeros([10]))
#
# model = tf.nn.softmax(tf.matmul(imagePlaceholder,weight)+errorr)
#
# # 这个东西用来衡量模型的不准确度（数学相关，交叉熵）
# jiaochashang = -tf.reduce_sum(answerPlaceholder*tf.log(model))
#
# # 可以看到，我们的参数有了，模型有了，然后评价体系也有了
# # TensorFlow的作用就是，运行，然后他帮你调参数，
# # 使得评价体系得到的结果是最优的
# # 在这里我们的评价体系是越低越好，所以代码这么写
# train_step  = tf.train.GradientDescentOptimizer(0.01).minimize(jiaochashang)
#
# init = tf.initialize_all_variables()
#
# sess =  tf.Session()
# sess.run(init)
# for i in range(1000):
#     batch_xs , batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step,feed_dict={imagePlaceholder:batch_xs,model:batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(answerPlaceholder,1))
# accuracy =  tf.reduce_mean(tf.cast(correct_prediction,"float"))
# print(sess.run(accuracy,feed_dict={imagePlaceholder:mnist.test.images,answerPlaceholder:mnist.test.labels}))

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
print(W.name)
print(b.name)
saver = tf.train.Saver()
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess,feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(session = sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(session = sess,feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


saver.save(sess,r"./check/Mymodel.ckpt")
with open('myNote.txt','w+') as fi:
    for i in b.eval(sess):
        fi.write(i+' ')
