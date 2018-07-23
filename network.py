import numpy as np
import tensorflow as tf


def loadData(filenamespath):
	tf.WholeFileReader()


        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        # input
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)       
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        #depth = tf.cast(depth, tf.int64)
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity= 50 + 3 * self.batch_size,
        )
        return images, depths, invalid_depths





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

print inference_resnet_50(tf.zeros([1,64,64,3]))

