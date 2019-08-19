

import numpy as ny
import string
import tensorflow as tf
from tensorflow.keras.utils import plot_model


def process_dataset(ip_file_name, op_file_name = "processed.data"):
	ip_file = open(ip_file_name)
	op_file = open(op_file_name, "w")
	for line in ip_file.readlines():
		new_line = line.replace("Iris-setosa", "1, 0, 0")
		new_line = new_line.replace("Iris-versicolor", "0, 1, 0")
		new_line = new_line.replace("Iris-virginica", "0, 0, 1")
		op_file.write(new_line)	
	ip_file.close()	
	op_file.close()
	

	arr = ny.genfromtxt(op_file_name, delimiter = ',')
	input_arr = arr[:len(arr),:4] 
	output_arr = arr[:len(arr), 4:]
	print(output_arr)
	output_arr = output_arr.astype(int)

	return input_arr, output_arr
	

def init_ann():
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(2, input_shape=(4, )))
	model.add(tf.keras.layers.Dense(3))
	plot_model(model, to_file='model.jpg', show_shapes=True)
	return model
	

#main
input_arr, output_arr = process_dataset("./iris.data")
model = init_ann()

SGD_optimizer = tf.keras.optimizers.SGD(momentum = 0.1)

model.compile(SGD_optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

model.fit(input_arr, output_arr, batch_size = None, epochs = 1000, verbose = 2, class_weight = {0:1, 1:1.5, 2:1}) #class weight assigned because 2nd 
																												  #category seemed underrepresented
																												  #(wasn't very accurate)

print('Successful training')

inputs = ny.genfromtxt("./test.data", delimiter = ',')
print(model.predict(inputs, verbose = 1))
