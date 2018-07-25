import random
import sys


filename = sys.argv[1]
lines = open(filename).readlines()
random.shuffle(lines)
open(filename, 'w').writelines(lines)
