from os import listdir
from os.path import isfile, isdir, join
from random import randint


#get the classes 
classes = []
with open('./imagenet_classes.txt') as classesfile:
	for line in classesfile:
		classes.append(line)



mypath = './data'
trainingTestingDataSplit = .3 * 100 #30% testing data

training = open('training_data.csv', 'w+')
testing	 = open('testing_data.csv', 'w+')


count = 0
for out in listdir(mypath):
	fullPath = join(mypath, out)
	if isdir(fullPath):
		count = count + 1

		for f in listdir(fullPath):
			filepath = join(fullPath, f)
			if isfile(filepath):
				if randint(0, 99) < trainingTestingDataSplit:
					#write to testing file
					print( 'testing: ' + filepath )
					testing.write(filepath + ',' + str(count - 1) + '\n')
				else:
					#write to training file 
					print( 'training: ' + filepath )
					training.write(filepath + ',' + str(count - 1) + '\n')



