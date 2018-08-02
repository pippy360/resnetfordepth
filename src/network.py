import numpy as np
import tensorflow as tf
from PIL import Image
from imagenetValues import get_imagenet_values


IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
BATCH_SIZE = 8
TRAIN_FILE = 'training_data.csv'

def loadData(csv_file_path, batch_size, imageDim):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, tag = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        # input
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)
       
        # resize
        image = tf.image.resize_images(image, (imageDim['height'], imageDim['width']))
        # generate batch
        images, tags = tf.train.batch(
            [image, tag],
            batch_size=batch_size,
            num_threads=4,
            capacity= 50 + 3 * batch_size,
        )
        return images, tags

def loadTags():
	tags = {}
	vals = get_imagenet_values()
	for val in vals:
		tags[val[0]] = val[1]
	return tags

def train():
	with tf.Graph().as_default():
		tagsStr = loadTags()
		imageDim = { 'height': IMAGE_HEIGHT, 'width': IMAGE_WIDTH }
		images, tags = loadData(TRAIN_FILE, BATCH_SIZE, imageDim=imageDim)
		with tf.Session() as sess:

			threads = []
			coord = tf.train.Coordinator()
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

			images_output, tags_output = sess.run([images, tags])
			for i, im in enumerate(images_output):
				pilimg = Image.fromarray(np.uint8(im))
				image_name = "%05d_org.png" % i
				print( image_name + "_" + str(tagsStr[tags_output[i]]) )
				pilimg.save(image_name)
			print 'tags:' + str(tags_output)


#######################################################################################
#			network
#######################################################################################




def residualLayer(name, input_data, inputSize, outputSize, isResize, strides=None):
	with tf.variable_scope(name) as scope:
            if strides == None:
                if isResize:
                    strides = (2, 2)
                else:
                    strides = (1, 1)

            #conv 1x1 /1
            conv = input_data
            conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=1, strides=strides, padding='SAME', use_bias=False)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.nn.relu(conv)

            #conv 3x3 /1
            conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=3, padding='SAME', use_bias=False)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.nn.relu(conv)

            #conv 1x1 /1
            conv = tf.layers.conv2d(conv, filters=outputSize, kernel_size=1, padding='SAME', use_bias=False)
            conv = tf.layers.batch_normalization(conv)
            #NO RELU
            
            #residule stuff
            inputsToAdd = input_data
            if(isResize):		
                    inputsToAdd = tf.layers.conv2d(inputsToAdd, filters=outputSize, kernel_size=1, strides=strides, padding='SAME', use_bias=False)
                    inputsToAdd = tf.layers.batch_normalization(inputsToAdd)

            conv = tf.add(conv, inputsToAdd)
            conv = tf.nn.relu(conv)
            return conv


def inference_resnet_50(input_images):
	#input 304 x 228 x 3

	conv1_a = tf.layers.conv2d(input_images, filters=64, kernel_size=7, strides=(2, 2), padding='SAME', name='conv1', use_bias=False)

	#Using biases
	biases = tf.get_variable('biases1', [64], dtype='float32', trainable=True)
	conv1_b = tf.nn.bias_add(conv1_a, biases)
        
	conv2 = tf.layers.batch_normalization(conv1_b, name='bn_conv1')
	conv3 = tf.nn.relu(conv2)
	conv4 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=(2, 2), name='pool1', padding='SAME')
	
	conv5 = residualLayer("Res_resize_1_", conv4, inputSize=64, outputSize=256, isResize=True, strides=(1, 1))
	for i in range(2):
		conv5 = residualLayer("Res_1_"+str(i), conv5, inputSize=64, outputSize=256, isResize=False)

	conv6 = residualLayer("Res_resize_2_", conv5, inputSize=128, outputSize=512, isResize=True)
	for i in range(3):
		conv6 = residualLayer("Res_2_"+str(i), conv6, inputSize=128, outputSize=512, isResize=False)

	conv7 = residualLayer("Res_resize_3_", conv6, inputSize=256, outputSize=1024, isResize=True)
	for i in range(5):
		conv7 = residualLayer("Res_3_"+str(i), conv7, inputSize=256, outputSize=1024, isResize=False)

	conv8 = residualLayer("Res_resize_4_", conv7, inputSize=512, outputSize=2048, isResize=True)
	for i in range(2):
		conv8 = residualLayer("Res_4_"+str(i), conv8, inputSize=512, outputSize=2048, isResize=False)

	global_avg_pool = tf.reduce_mean(conv8, [1,2])

	return tf.contrib.layers.fully_connected(global_avg_pool, 1000)

train()


