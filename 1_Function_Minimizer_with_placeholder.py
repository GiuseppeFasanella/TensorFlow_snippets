import numpy as np
import tensorflow as tf

##Ora la mia cost function dipende dai parametri in coefficient

coefficients = np.array([[1.],[-10.],[25.]])
#define a variable in tensorflow
w = tf.Variable(0,dtype=tf.float32) 

## List of parameters,  di shape [3,1]
## Un placeholder e' una variabile il cui valore verra'
## specificato dopo con feed_dict
x = tf.placeholder(tf.float32, [3,1])

## my cost function is defined as x00w^2 -x10w + x20
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
##learning_rate = 0.01
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

## Printa il valore iniziale w=0
print(session.run(w))

## Printa una iterazione del gradient descent
session.run(train, feed_dict={x:coefficients})
print(session.run(w))

## Ora facciamo 1000 iterazione del gradient descent
for i in range(1000):
  session.run(train, feed_dict={x:coefficients})
print(session.run(w)) #--> 4.9999. Noi stiamo minimizzando (w-5)^2. Giusto!
