import numpy as np
import tensorflow as tf
from PIL import Image
from imagenetValues import get_imagenet_values
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from pprint import pprint

import os, sys, argparse
import h5py
from numpy import array


IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
BATCH_SIZE = 8
TRAIN_FILE = 'training_data.csv'

everything = {}

def restore(sess, model):
	#TODO: handle case when no checkpoint

	# Restore the moving average version of the learned variables for eval.
	#variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
	#variables_to_restore = variable_averages.variables_to_restore()

	#is there one??
	model.load_weights(weights_path)
	ckpt = tf.train.get_checkpoint_state( '../20180601_resnet_v2_imagenet_checkpoint' )
	#print_tensors_in_checkpoint_file(file_name=ckpt.model_checkpoint_path, tensor_name='', all_tensors=False)
	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.train.Saver()
		# Restores from checkpoint
		saver.restore(sess, ckpt.model_checkpoint_path)
		# Assuming model_checkpoint_path looks something like:
		#   /my-favorite-path/cifar10_train/model.ckpt-0,
		# extract global_step from it.
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

		print('checkpoint loaded with global_step: ' + str(global_step))
		return global_step
	else:
		return None

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

def loadAllVariables():
	res = {}
	for x in tf.global_variables():
		res[x.name.replace('/', '_')] = x
	return res

def convertToList(inputVar):
	valof = []
	for vl in inputVar:
		valof.append(vl)
	return valof

def loadFromFile():
	loadedVarList = []
	loadedFile = h5py.File("resnet50_weights_tf_dim_ordering_tf_kernels.h5", 'r')
	print loadedFile
	for tfVar in loadedFile:
		out1 = loadedFile[str(tfVar)]
		for subVar in out1:
			out2 = out1[str(subVar)]
			for actualVar in subVar[str()]:
				print str(actualVar)
				loadedVarList.append(actualVar)
	
	convertedVars = {}
	for toConvert in loadedVarList:
		convertedVars[str(toConvert)] = convertToList(toConvert)
	return convertedVars


def train():
	with tf.Graph().as_default():

		imageDim = { 'height': IMAGE_HEIGHT, 'width': IMAGE_WIDTH }
		images, tags = loadData(TRAIN_FILE, BATCH_SIZE, imageDim=imageDim)
		network = inference_resnet_50(images)

		loadedFromFile = loadFromFile()
		for i in loadedFromFile:
			print i
		tfVars = loadAllVariables()
		assignOps = []
		for k, v in tfVars.items():
			assignOps.append( v.assign( loadedFromFile[k] ) )

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(assignOps)  # or `assign_op.op.run()`
			return

		with tf.Session() as sess:
			restore(sess)
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


def residualLayer(input_data, stage, block, inputSize, outputSize, isResize, strides=None):
	if strides == None:
		if isResize:
		    strides = (2, 2)
		else:
		    strides = (1, 1)
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	#conv 1x1 /1
	conv = input_data
	conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=1, strides=strides, padding='SAME', use_bias=False, 
		name=conv_name_base + '2a')
	everything[ conv_name_base + '2a' ] = conv
	conv = tf.layers.batch_normalization(conv, name=bn_name_base + '2a')
	everything[ bn_name_base + '2a' ] = conv
	conv = tf.nn.relu(conv)

	#conv 3x3 /1
	conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=3, padding='SAME', use_bias=False, 
		name=conv_name_base + '2b')
	everything[ conv_name_base + '2b' ] = conv
	conv = tf.layers.batch_normalization(conv, name=bn_name_base + '2b')
	everything[ bn_name_base + '2b' ] = conv
	conv = tf.nn.relu(conv)

	#conv 1x1 /1
	conv = tf.layers.conv2d(conv, filters=outputSize, kernel_size=1, padding='SAME', use_bias=False,
		name=conv_name_base + '2c')
	everything[ conv_name_base + '2c' ] = conv
	conv = tf.layers.batch_normalization(conv, name=bn_name_base + '2c')
	everything[ bn_name_base + '2c' ] = conv
	#NO RELU

	#residule stuff
	inputsToAdd = input_data
	if(isResize):		
		inputsToAdd = tf.layers.conv2d(inputsToAdd, filters=outputSize, kernel_size=1, strides=strides, 
			padding='SAME', use_bias=False, name=conv_name_base + '1')
		everything[ conv_name_base + '1' ] = conv
		inputsToAdd = tf.layers.batch_normalization(inputsToAdd, name=bn_name_base + '1')
		everything[ bn_name_base + '1' ] = conv

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
	
	conv5 = residualLayer(conv4, stage=2, block='a', inputSize=64, outputSize=256, isResize=True, strides=(1, 1))
	conv5 = residualLayer(conv5, stage=2, block='b', inputSize=64, outputSize=256, isResize=False)
	conv5 = residualLayer(conv5, stage=2, block='c', inputSize=64, outputSize=256, isResize=False)

	conv6 = residualLayer(conv5, stage=3, block='a', inputSize=128, outputSize=512, isResize=True)
	conv6 = residualLayer(conv6, stage=3, block='b', inputSize=128, outputSize=512, isResize=False)
	conv6 = residualLayer(conv6, stage=3, block='c', inputSize=128, outputSize=512, isResize=False)
	conv6 = residualLayer(conv6, stage=3, block='d', inputSize=128, outputSize=512, isResize=False)

	conv7 = residualLayer(conv6, stage=4, block='a', inputSize=256, outputSize=1024, isResize=True)
	conv7 = residualLayer(conv7, stage=4, block='b', inputSize=256, outputSize=1024, isResize=False)
	conv7 = residualLayer(conv7, stage=4, block='c', inputSize=256, outputSize=1024, isResize=False)
	conv7 = residualLayer(conv7, stage=4, block='d', inputSize=256, outputSize=1024, isResize=False)
	conv7 = residualLayer(conv7, stage=4, block='e', inputSize=256, outputSize=1024, isResize=False)
	conv7 = residualLayer(conv7, stage=4, block='f', inputSize=256, outputSize=1024, isResize=False)

	conv8 = residualLayer(conv7, stage=5, block='a', inputSize=512, outputSize=2048, isResize=True)
	conv8 = residualLayer(conv8, stage=5, block='b', inputSize=512, outputSize=2048, isResize=False)
	conv8 = residualLayer(conv8, stage=5, block='c', inputSize=512, outputSize=2048, isResize=False)

	global_avg_pool = tf.reduce_mean(conv8, [1,2], name='avg_pool')
	return tf.layers.dense(global_avg_pool, units=1000, activation='softmax', name='fc1000')

train()


