import numpy as np
base_dir = '/Users/jinzhu/Google\ Drive/2017\ spring/visual\ learning\ recognition/Project/Tensor\ Flow/resnet-in-tensorflow/'
data_dir = base_dir+'cifar10_data'
full_data_dir = base_dir+'cifar10_data/cifar-10-batches-py/data_batch_'
vali_dir = base_dir+'cifar10_data/cifar-10-batches-py/test_batch'

#IMG_WIDTH = 32
#IMG_HEIGHT = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_DEPTH = 3
NUM_CLASS = 10

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?

def _read_one_batch(path, is_random_label):
    '''
    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays
    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''
    fo = open(path, 'rb')
    dicts = cPickle.load(fo)
    fo.close()

    data = dicts['data']
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label

def read_in_all_images(address_list, shuffle=True, is_random_label = False):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays
    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print 'Reading images from ' + address
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


    if shuffle is True:
        print 'Shuffling'
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label



def test(self, test_image_array):
    '''
    This function is used to evaluate the test data. Please finish pre-precessing in advance
    :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
    img_depth]
    :return: the softmax probability with shape [num_test_images, num_labels]
    '''
    num_test_images = len(test_image_array)
    print "FLAGS.test_batch_size"
    print (FLAGS.test_batch_size)
    num_batches = num_test_images // FLAGS.test_batch_size
    remain_images = num_test_images % FLAGS.test_batch_size
    print '%i test batches in total...' %num_batches

    # Create the test image and labels placeholders
    print "5----------"

    self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

    # Build the test graph
    print "6----------"
    logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
    print "7----------"
    predictions = tf.nn.softmax(logits)

    # Initialize a new session and restore a checkpoint
    print "8----------"
    saver = tf.train.Saver(tf.global_variables())
    print "9----------"
    sess = tf.Session()
    #saver.restore(sess, './model.ckpt')
    print "10----------"
    #ResNet 50 pretrained model 
    saver.restore(sess, '/Users/jinzhu/Google Drive/2017 spring/visual learning recognition/Project/Tensor Flow/tensorflow-resnet-MIT/data/tensorflow-resnet-pretrained-20160509/ResNet-L50.ckpt')
    #saver.restore(sess, FLAGS.test_ckpt_path)
    print 'Model restored from ', FLAGS.test_ckpt_path
    print "11----------"
    prediction_array = np.array([]).reshape(-1, NUM_CLASS)
    print "num class"
    print NUM_CLASS
    # Test by batches
    print "12----------"

    for step in range(num_batches):
        print "13----------"

        if step % 10 == 0:
            print '%i batches finished!' %step
        offset = step * FLAGS.test_batch_size
        print "14----------"

        test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]
        print "15----------"

        batch_prediction_array = sess.run(predictions,
                                    feed_dict={self.test_image_placeholder: test_image_batch})
        print "16----------"

        prediction_array = np.concatenate((prediction_array, batch_prediction_array))

    # If test_batch_size is not a divisor of num_test_images
    print "17----------"

    if remain_images != 0:
        print "18----------"

        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        # Build the test graph
        print "19----------"
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
        print "20----------"
        predictions = tf.nn.softmax(logits)
        print "21----------"
        test_image_batch = test_image_array[-remain_images:, ...]
        print "22----------"
        batch_prediction_array = sess.run(predictions, feed_dict={
            self.test_image_placeholder: test_image_batch})
        #print "batch image prediction array"
        #print (batch_prediction_array)
        print "23----------"

        prediction_array = np.concatenate((prediction_array, batch_prediction_array))
        print "argmax ---------"
        print np.argmax(prediction_array,axis=1)

    return prediction_array
# Start the training session
print "1----------"
#saver.restore(sess, './model.ckpt')
#print "2---------"
print "3----------"
test_image_array, test_labels = read_in_all_images([vali_dir],
                                                       is_random_label=VALI_RANDOM_LABEL)
print "test labels"
print test_labels[0:200]
#test_image_array = whitening_image(test_image_array)
print "test array shape"
print test_image_array.shape
test_image_array = test_image_array[0:200,:,:,:]
print test_image_array.shape
print "4----------"
#test_image_array = test_image_array[]
prediction_array = test(test_image_array)
print prediction_array
print "prediction array shape"
print (prediction_array.shape)
print "sum"
print np.sum(prediction_array[0])
