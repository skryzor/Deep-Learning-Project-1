from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import tensorflow as tf
import sys

import os
repo_path = os.path.dirname(os.path.abspath(__file__))
test_output_path1 = repo_path + "/Software1.txt"
test_output_path2 = repo_path + "/Software2.txt"
model_path = "model/saved_model.h5"


def get_test_data():
	train_input = []
	train_output = []
	for i in range(1, 101):
		if i % 3 == 0 and i % 5 == 0:
			train_output.append([1, 0, 0, 0])
		elif i % 3 == 0:
			train_output.append([0, 1, 0, 0])
		elif i % 5 == 0:
			train_output.append([0, 0, 1, 0])
		else:
			train_output.append([0, 0, 0, 1])
		train_input.append( np.array([i >> d & 1 for d in range(16)]) )

	train_input = np.asarray(train_input)
	train_output = np.asarray(train_output)
	return train_input, train_output

def software1(test_input_path):
	f1 = open(test_input_path, 'r')
	nums = f1.readlines()
	nums = [int(i) for i in nums]
	nums = np.asarray(nums)
	f1.close()
	# print(nums)
	output = []
	for i in range(nums.shape[0]):
		if nums[i] % 3 == 0 and nums[i] % 5 == 0:
			output.append(str(nums[i]) + "\n")
			output.append("fizzbuzz" + "\n")
		elif nums[i] % 3 == 0:
			output.append(str(nums[i]) + "\n")
			output.append("fizz" + "\n")
		elif nums[i] % 5 == 0:
			output.append(str(nums[i]) + "\n")
			output.append("buzz" + "\n")
		else:
			output.append(str(nums[i]) + "\n")

	with open(test_output_path1, 'w') as f2:
		for item in output:
			f2.write("%s" % item)
	f2.close()
	print("software1 file saved")
	return

def software2(test_input_path):
	f1 = open(test_input_path, 'r')
	nums = f1.readlines()
	nums = [int(i) for i in nums]
	nums = np.asarray(nums)
	f1.close()
	test_input = []
	test_output = []
	for j in nums:
		if j % 3 == 0 and j % 5 == 0:
			test_output.append([1, 0, 0, 0])
		elif j % 3 == 0:
			test_output.append([0, 1, 0, 0])
		elif j % 5 == 0:
			test_output.append([0, 0, 1, 0])
		else:
			test_output.append([0, 0, 0, 1])
		test_input.append( np.array([j >> d & 1 for d in range(16)]) )

	test_input = np.asarray(test_input)
	test_output = np.asarray(test_output)
	
	model = neural_model()
	model.load_weights(model_path)
	loss,acc = model.evaluate(test_input,  test_output, verbose=2) 
	print("Restored model, accuracy: {:5.2f}%".format(100*acc))
	predictions = model.predict(test_input)
	# for i in range(test_output.shape[0]):
		# print(predictions[i], test_output[i])

	output = []
	for i in range(test_input.shape[0]):
		output.append(str(nums[i]) + "\n")
		j = np.argmax(predictions[i])
		if j == 0:
			output.append("fizzbuzz" + "\n")
		elif j == 1:
			output.append("fizz" + "\n")
		elif j == 2:
			output.append("buzz" + "\n")
		else:
			continue

	with open(test_output_path2, 'w') as f2:
		for item in output:
			f2.write("%s" % item)
	f2.close()
	print("software2 file saved")
	return

def test(test_path):
	print("inside test")
	print("test path is " + str(test_path))
	software1(test_path)
	software2(test_path)
	return

def neural_model():
	model = Sequential()
	model.add(Dense(256, activation=tf.nn.relu, input_shape=(16,)))
	# Afterwards, we do automatic shape inference:
	model.add(Dense(256, activation=tf.nn.relu))
	model.add(Dense(4, activation=tf.nn.softmax))
	model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
	return model

def train():
	print("inside train")
	train_input = []
	train_output = []
	for i in range(101, 1001):
		if i % 3 == 0 and i % 5 == 0:
			train_output.append([1, 0, 0, 0])
		elif i % 3 == 0:
			train_output.append([0, 1, 0, 0])
		elif i % 5 == 0:
			train_output.append([0, 0, 1, 0])
		else:
			train_output.append([0, 0, 0, 1])
		train_input.append( np.array([i >> d & 1 for d in range(16)]) )



	# print([int(x) for x in list('{0:0b}'.format(101))])
	# test_input, test_output = get_test_data()
	train_input = np.asarray(train_input)
	train_output = np.asarray(train_output)
	model = neural_model()
	model.summary()
	callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True,
                                                 verbose=1)
	
	model.fit(train_input, train_output, batch_size=32, epochs=100, callbacks=[callback])

	# metrics = model.evaluate(test_input, test_output)
	# predictions = model.predict(test_input)
	# for i in range(test_output.shape[0]):
	# 	print(test_output[i], predictions[i])

	return

def main():
	print("Nilesh Abhimanyu Kande")
	print("SR. NO. 15688")
	print("Computer Science and Automation")
	print("Indian Institute of Sciecne, Bangalore")
	args = len(sys.argv)
	print(args)
	if(args == 1):
		train()
	elif (args == 3 and sys.argv[1] == "--test-data"):
		test_input = sys.argv[2]
		test(test_input) 
	else:
		print("Invalid arguments count")
	return

if __name__ == "__main__":
	main()