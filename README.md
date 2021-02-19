Problem statement:   Live face recogninition Here the target is to identify face and label with the respective name.


Python version - 3.7.0 (due to uncompatability program is not running so i hav echanged the compatable version)

Here in this Opencv, Numpy, Pickle, Pillow and Os libraries are used.
cv2 version - 4.1.0
numpy version - 1.16.4
pickle version - 4.0

Here libraries are used with their compatable versions.


cascades are used here in order recognize specific parts of face in the frame.
In this API(facerecognize-train) data is trained in order to recognize the face. Before running the main program we need to run the 
training data set.

In respective path first need to setup with cascades, images with sample data with other set of images in order to train API for easy recognition. 
Here cascades : eyes, frontal face , full body, left eye, smile, right eye and frotal face etc
but for the requirement we used frontal face, eye and smile cascade especially for the recognizing the images.

In identifiers path we are just storing values of the live face detection as i can say these as my iteratives and these identifiers are stored in yml format. used example(face-trainner.yml).

Running steps:    collect no of images in set of data and store them with their respective name folder as this API stores the name of the person from folder name. for example in my data sets i stored my images with different pictures in a folder naming it as "praveen"
eg: C:\praveen\germany\Artificial intelligence\praveen\images\praveen

here training the images into numpy array using pil_image library. This is taking every pixel value into some sort of numpy array.
This is just to convert the every images into set of numbers.

The changes in code from the below reference link is versions i dealed with python and even the libraries which is effecting the results. In the reference the entire project is there in which i had simplified the query for my target state. As mention my source code is ony for recognizing the live face withh label name. The project is trimmed according my requirement and sort of variable storage for better understanding from scratch. In this cascades presented are not changed as we have standard requirement for recognition of face.
The sample results which i runned through are mentioned below.

Install mentioned libraries in the above 
eg; pip intsall numpy --upgrade

Please make sure that you have OpenCV Contrib installed, using the command pip install opencv-contrib-python 


==>  import the this source code and initially run the python training file :
python facerecognize-train.py

once the set of images place are trained by the API, then main progrograms comes into the picture. Now run the python main file:
python facerecognize.py

here the x and y axis values of face are stored and even the end of the streaming the image will be captured in to to train the more extensively the API.

samle result of images are stored. 
eg: https://github.com/praveenkumarappari/Face_recognition/test_image.png

Reference : https://github.com/codingforentrepreneurs/OpenCV-Python-Series/tree/master/src





