from PIL import Image, ImageFilter,ImageEnhance
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy

import tensorflow as tf
import os
import datetime

# define the name of keyspace in cassandra
KEYSPACE="mnistkeyspace"

# connect to cassandra and create the keyspace
cluster = Cluster(contact_points=['mnist-cassandra'], port=9042)
session = cluster.connect()
session.execute("""
           CREATE KEYSPACE %s
           WITH replication = { 'class': 'SimpleStrategy',  'replication_factor': '2' }
           """ % KEYSPACE)

#enter the keyspace and create a table
session.set_keyspace(KEYSPACE)
session.execute("""
           CREATE TABLE predictrecord (
               time text,
               filename text,
               result text,
               PRIMARY KEY (time)
           )
           """)

#initalize the flask app
app = Flask(__name__)


@app.route('/upload')
def upload_file():
   """
   if the route is '/upload'
   show the html file that can be find in the 'template file folder' 
   which named 'handwriting_webpage.html'
   """
   return render_template('handwriting_webpage.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
        
        f = request.files['file']

        #find the path of current file 
        basepath = os.path.dirname(__file__)

        #find the path of the file that is used to save images
        upload_path = os.path.join(basepath, 'upload_file',secure_filename(f.filename))

        #convert the relative path to abusolute path
        upload_path = os.path.abspath(upload_path)

        #save the image to specific path
        f.save(upload_path)

        #read the images
        im = Image.open(upload_path)          

        #convert the image to grayscale image
        im = im.convert('L')

        #sharp the image
        im=ImageEnhance.Sharpness(im).enhance(3)

        #convert the image to 28*28 Pixels
        im=im.resize((28,28))

        #save the standardlized image to specific path
        im.save(upload_path)
        tv = list(im.getdata())

        result = [(255-x)*1.0/255.0 for x in tv]
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        def weight_variable(shape):
            initial = tf.truncated_normal(shape,stddev = 0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1,shape = shape)
            return tf.Variable(initial)

        def conv2d(x,W):
            return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x,[-1,28,28,1])

        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        #restore all the variables
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            #use the saved model
            saver.restore(sess, "trained_model/model.ckpt")
 
            prediction=tf.argmax(y_conv,1)
            predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)

            #get the predicted number
            number='%d'%predint[0]

            predict_number='%s'%number
            output='predict result:'+number      

            #get the current time     
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            #insert the record to cassandra with values of the current time, filename and the predicted number to TABLE predictrecord, KEYSPACE mnistkeyspace
            session.execute('insert into mnistkeyspace.predictrecord(time,filename,result) values(%s,%s,%s);',[now_time,f.filename,predict_number])
            return output   

if __name__ == '__main__':
   #run the app in port 80 which is correspond to the port that exposed by docker container
   app.run(host='0.0.0.0', port=80)


