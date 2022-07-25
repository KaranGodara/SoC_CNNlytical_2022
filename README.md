# SoC_CNNlytical_2022
The goal of this project was to make you well-versed with CNNs, enough that one can write an entire network in PyTorch from scratch. It will kickstart one's journey into Deep Learning.

## Mentors
1. Akshat Kumar
2. Vaibhav Raj
3. Ashwin Ramachandran

## Work done under the SoC
### Mid April - First Week of May
In this phase everyone was given time to go through the first few lectures of the material provided as the reference to the SoC. The reference we followed intially was CS231 lectures of Stanford given by Andrej Karpathy.

The topics we studied and learnt about were,
1. Introduction and History context of computer vision.
2. Data-driven approach, kNNs, Linear classification 1.
3. Linear classification 2, optimization.
4. Backpropagation, neural networks 1.
5. Neural networks 2.
6. Neural networks 3, Intro to ConvNets.

The above topics broadly introduced us to the world of Computer Vision and made us realise how difficult it is traditionally to classify images as they are nothing but collection of numbers. But using this new strategy called data-driven approach we not only solve this issue but also use computers in a way which is different from the classical way.

We first of all studied how a trivial way of classification can be implemented using **Nearest Neighbour methods and k-Nearest Neighbours**. This idea do not help us in the long run but is a good example of what one tries to achieve using data driven approach and computer vision.

After this trivial start to computer vision, we moved on to good stuff one step at a time, the first being linear classification wherein we multiply and add matrices linearly to predict the outcome. We saw how that idea works and how one can go about implementing it.

The next natural direction one proceeds and so did we was to study about the scores and the loss functions we can use on them to see how good our model is performing. The two loss functions we studied were **Support Vector Machine (SVM) Loss** and **Softmax loss**. We saw how they work, what they try to represent and how can can use them to make a better model.

After that we moved on to see how can can use those loss functions to **back-propagate** the information of how the parameteres need to be changed. Also we saw what are the different ways in which one can optimise the model by different ways simple one being **Stochastic Gradient Descent method** also called as the Vanilla method.

After learning about linear classification, we moved on to very important portion of this learning curve, the Neural Networks.
In part 1, we studied how one can set up thr architecture of their model using those linear functions we studied already and combining with them various different types of non-linearity, thus making the model deep and complex to successfully grab the required information from the training data and set the model as perfect as possible.

In part 2, we set the data and the loss, that is we saw how one can pre-process the data given to him before training, how to initialise the weights, what is batch-normalisation, how does regularization helps in all this stuff and how that is important and finally setting up the loss functions we want to use.

In part 3, we studied about the learning and evaluation of model and how one can baby-sit the entire process. In this part we saw various other methods of optimization other than vanilla one such as ***momentum update***, ***Adagrad***, ***RMS-prop***, ***nesterov*** etc. Also another very important thing we tackled here was the hyperparameter optimization as it plays a crucial role in how are model is gonna work, some of these hyperparameters included learning rate, epochs, loss functions, intermediate layer sizes, number of layers etc.

Finally after all this we studied a toy example using these Neural Networks to predict something.

Another refrence which we needed to study in this period/phase was tutorial on **numpy** as that gives us a faster way in data-learning to deal with huge matrices such as those of images.

### First week of May to Mid-May
Here we were presented our **first Assignment** which we needed to complete. In this assignment we needed to use the MNIST dataset comprising of numbers 0 to 9 written in various styles. Our model was to be made using Neural Networks and its aim was to be able to predict the number written on them once the training was done with reasonably good accuracy.
We in this assignment implemented from scratch everything starting from loss functions, optimization methods to backward propagation. We were to only and only use numpy to all the above calculations and steps.

Few test images representing my model prediction were:

<img width="239" alt="Screenshot 2022-07-25 at 5 47 43 PM" src="https://user-images.githubusercontent.com/94215375/180781597-f66ea734-3043-441f-9571-375915596f0d.png">


### Mid-May to first week of June
In this phase we were to study about the **Convolutional Neural Networks** from the lectues we were following of CS231n by Andrej Karpathy and see how they are different from the Neural Networks and why and how they provide better results.
In it we saw how the model architechture is defined as how here we instead of using a fully-connected pathway as we did in NNs we only connect few neurons from the one layer to the neuron in following layer. We saw how here instead of working in 2D we also keep track of spatial 3D model of data.
We studied how can moves from one layer to other, what is stride, padding, max-pooling etc. We studied how there exists different kinds of layer and what are their roles.

Apart from this what we were required to learn and study about **pyTorch API** as it helps in easily implement all the kinds of loss, forward-backward propagation etc which we had to implement manually in assignment 1, using a simple function calling.
We had to see video tutorials and articles to study the same.

After learning all these things we were given **Assignment 2**, wherein using the same dataset MNIST but this time using pyTorch API, we were to implement our previous model. 

Few test images representing my model prediction were:

<img width="201" alt="Screenshot 2022-07-25 at 5 48 13 PM" src="https://user-images.githubusercontent.com/94215375/180781666-f005fab5-e809-4a12-8c79-6d5d41da081a.png">

### first week of June to end of June
Till now we have studied all the theory about NNs carefully and also implemented the same for 2 straight assignments. Now in this phase we were told to go over the CNN part carefully again, solve any query or doubt we may have yet. 
Also we were given time to study about pyTorch and TorchVision(used for computer vision problems). 

After going through this learning phase, we were given Assignment 3, this assignment was programming heavy and more complex as in it we were to apply the knowledge we had gained till now about CNNs on making our model and this time the dataset we used was CIFAR dataset, in which we classified images in 10 different labels.
We implemented entire working model of CNN and also saw how it gives significant difference in performance compared to linear classifier.

### Early July to end of SoC
After learning all the theory required to classify images using CNNs and linear classifiers. We were ready to step into the real world. So now in Assignment 4, we were told to implement the paper based on **U-Net Architechture** and to do **image segmentation**, which involves working with very small dataset but yet getting SOTA accuracies by changing the given image by performing various functions on it, such as cropping, flipping, reverting etc. 
The dataset we used for this assignment was **CARVANA dataset**, which is a dataset containing cars at various different angles and colors.

The dataset and its mask are :

<img width="96" alt="Screenshot 2022-07-25 at 7 02 53 PM" src="https://user-images.githubusercontent.com/94215375/180789709-da377205-d68d-4499-8cd9-9fd329b2a235.png">

Happy Coding ♥️
