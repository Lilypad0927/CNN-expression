import tensorflow
hello = tensorflow.constant('Hello, TensorFlow!')
sess = tensorflow.Session()
print sess.run(hello)
