import numpy as np
import tensorflow as tf

#define a variable in tensorflow
w = tf.Variable(0,dtype=tf.float32) 
## my cost function is defined as w^2 -10w + 25
cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
##learning_rate = 0.01
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

## Printa il valore iniziale w=0
print(session.run(w))

## Printa una iterazione del gradient descent
session.run(train)
print(session.run(w))

## Ora facciamo 1000 iterazione del gradient descent
for i in range(1000):
  session.run(train)
print(session.run(w)) #--> 4.9999. Noi stiamo minimizzando (w-5)^2. Giusto!
