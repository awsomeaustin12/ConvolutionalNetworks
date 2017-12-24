import numpy as np
import pickle
from app.model.preprocessor import Preprocessor as img_prep

#class for loading our saved model and then classifying new images
class LiteOCR:

    def __init__(self, fn="aplha_weights.pkl", pool_size=2):
        #load the weights from the pickle file and the meta deta
        [weights,meta] = pickle.load(open(fn,'rb'), encoding='latin1')
        #list to store the labels
        self.vocab = meta["vocab"]

        #save how many rows and columns are in an image
        self.img_rows = meta["img_side"] ; self.img_cols = meta["img_side"]

        #load the convolutional neural network
        self.CNN = LiteCNN()
        #with the saved weightss
        self.CNN.load_weights(weights)
        #define the size of the pooling layers
        self.CNN.pool_size = int(pool_size)


    #classify the new image
    def predict(self,image):
        print(image.shape)
        #vectorize the image into the right shape for our network
        X = np.reshape(image,(1,1,self.img_rows,self.img_cols))
        X = X.astype("float32")



        #make the prediction
        predicted_i = self.CNN.predict(X)
        #return the predicted label
        return self.vocab[predicted_i]





class LiteCNN:

    def __init__(self):
        #a place to store the layers
        self.layers = []
        #size of pooling area for max pooling
        self.pool_size = None

    def load_weights(self, weights):
        assert not self.layers, "Weights can only be loaded once"
        #add the saved matrix value to the convultional network
        for k in range(len(weights.keys())):
            self.layers.append(weights['layer_{}'.format(k)])

    def predict(self, X):
        #shis is where the network magic happens at a high level
        h = self.cnn_layer(X, layer_i=0,border_mode="full") ; X=h
        h = self.relu_layer(X); X=h;
        h = self.cnn_layer(X, layer_i=2, border_mode="valid"); X=h
        h = self.relu_layer(X); X=h;
        h = self.maxpooling_layer(X); X=h
        h = self.dropout_layer(X, 0.25); X=h
        h = self.flatten_layer(X, layer_i=7); X = h;
        h = self.dense_layer(X,fully,layer_i=10); X=h;
        h = self.softmax_layer2D(X); x=h;
        max_i = self.classify(X)
        return max_i[0]



    def cnn_layer(self, X, layer_i=0,border_mode="full"):
        #first store the feature map and bias values in 2 variables
        features = self.layers[layer_i]["param_0"]
        bias = self.layers[layer_i]["param_1"]

        #size of filter
        patch_dim = features[0].shape[-1]
        #number of features
        nb_features = features = features.shape[0]
        #size of image
        img_dim = X.shape[0]
        #number of channels
        image_channels = X.shape[1]
        #number of images
        nb_images = X.shape[0]




        #border mode full just means you get an output that is the full size
        #that means the filter has to go outside the bounds of the input and the areas outside of
        #the input is normally padded with zeros

        if border_mode == "full":
            conv_dim = img_dim + patch_dim -1
        elif border_mode == "valid":
            conv_dim = img_dim - patch_dim + 1


        #initiate the feature matrix
        convolved_features = np.zeros((nb_images,nb_features,conv_dim,conv_dim))
        #then interate through every image
        for image_i in range(nb_images):
            #for each feature
            for feature_I in range(nb_features):
                #intiialize convolved image as empty
                convolved_img = np.zeros((conv_dim,conv_dim))
                #now go through each channel
                for channel in range(image_channels):
                    #lets extract a feature from the feature map
                    feature = features[feature_I,channel,:,:]
                    #then define a channel specific part of our image
                    image = X[image_i,channel,:,:]
                    #perform convoution on image usig a given feature
                    convolved_img += self.convolved2d(image, feature,border_mode)

                #add abias to the convolved image
                convolved_img = convolved_img + bias[feature_I]
                #add it to the list of our convolved feature
                convolved_features[image_i,feature_I,:,:] = convolved_img
        return convolved_features


    def dense_layer(self, X, layer_i=0):
        #so well intiailaize or weiht and bias for this layer
        W = self.layers[layer_i]["param_0"]
        b = self.layers[layer_i]["param_1"]
        #and multiply them by our input (dot products all the way through)
        output = np.dot(X,W)+b
        return output

    @staticmethod

    #what does the convolution operaation look like given an image and a feature map
    def convolve2d(image, feature, border_mode="full"):
        #define the tensor dimensions of the image and the feature
        image_dim = np.array(image.shape)
        feature_dim = np.array(feature.shape)

        #as well as a target dimension we want it to get down to
        target_dim = image_dim +feature_dim -1

        #then perform a fast fourier transorm on both the input and the filter
        #performing a convolution can be a written as a for loop but
        #for a lot of convolutions the approach is to slow
        #it will be much much faster using a fast fourier transform algorithm
        fft_result = np.fft.fft2(image,target_dim) * np.fft.fft2(feature,target_dim)

        #set the result to our target
        target = np.fft.ifft2(fft_result).real

        if border_mode == "valid":
            #decide a target dimension to convolve arounf
             valid_dim = image_dim - feature_dim + 1
            if np.any(valid_dim <1):
                valid_dim = feature_dim - image_dim +1
                start_i = (target_dim-valid_dim) // 2
                end_i = start_i + valid_dim
                target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]
        return target

    #make feature map more dense by pooling
    #selct the max values from the image matrix and use that
    def maxpooling_layer(self, convolved_features):
        #given our learned features and images
        nb_features = convolved_features.shape[0]
        nb_images = convolved_features.shape[1]
        conv_dim = convolved_features.shape[1]
        res_dim = int(conv_dim / self.pool_size) #assumed square shape


        #intiialize our mode dense feature list as empty
        pooled_features = np.zeros((nb_features,nb_images,res_dim,res_dim))

        #for each image
        for image_i in range(nb_images):
            #and each feature map
            for feature_i in range(nb_features):
                #begin by the row
                for pool_row in range(res_dim):
                    #define the start and end points
                    row_start = pool_row * self.pool_size
                    row_end = row_start + self.pool_size

                    #for each column since its a 2d iteration
                    for pool_col in range(res_dim):
                        #define the start and end points
                        col_start = pool_row * self.pool_size
                        col_end = col_start + self.pool_size

                        #define the patch give our defined starting ending pioints
                        patch = convolved_features[feature_i,image_i,row_start:row_end, col_start:col_end]
                        #then take the max from thaat patch and store it
                        #this is our new learned feature/filter
                        pooled_features[feature_i,image_i,pool_row,pool_col] = np.max(patch)

        return pooled_features



    def relu_layer(x):
        #turn all negaative values in a matrix into zeros
        z = np.zeros_like(x)
        return np.where(x>z,x,z)

    def dropout_layer(X,p):
        retain_prob = 1. -p
        X *= retain_prob
        return X


    #tensor tranformation, less dimenensions
    def flatten_layer(X):
        flatX = np.zeros((X.shape[0],np.prod(X.shape[1:])))
        for i in range(X.shape[0]):
            flatX[i,:] = X[i].flatten(order='C')
        return  flatX

    def softmax_layer2D(w):
        #this function will calculate the probablilities of each
        #target class over all possible target classes
        maxes = np.amax(w,axis=1)
        maxes = maxes.reshape(maxes.shape[0],1)
        e = np.exp(w-maxes)
        dist = e / np.sum(e, axis =1, keepdims=True)
        return dist

    def classify(X):
        return X.argmax(axis=-1)

    def train(self, data, labels, iteration_counts, alpha, X):

        alpha = 0.01;
        batch_size = X.shape[0]

        for x in range(iteration_counts):
            print('Iteration #{}'.format(x))
            errors = np.zeros((batch_size,self.output_size))
            for y in range(batch_size):
                errors[y,:] = (labels[y,1] - self.predict(self,data[y,1])) * self.transfer_derivative(X)
                errors[y,:] = np.square(errors)
                mse= 0.5 * np.sum(errors,axis=1,dtype="float32")
                W = self.update_weights(self, X, error=mse,data[y,:])

    def test(self,data, labels, X):
        good = 0
        amount_tested = np.shape(data)[a]
        accuracy = 0
        for i in range(amount_tested):
            if np.argmax(self.predict(self, X) == np.argmax(labels[x, :]):
                good += 1 ; accuracy = good/i;
                print('The current accuracy is' + accuracy);

        accuracy = good/amount_tested
        return accuracy


    def transfer_derivative(X):
        return (X)* (1-X)

    def update_weights(self,X,error,input):
        W = self.layers[layer_i]["param_0"]
        b = self.layers[layer_i]["param_1"]

    '''error = (expected - output) * transfer_derivative(output) '''

        return W - (alpha*error*input + b)


    def trai(self data, labels, iteration_count, alpha, X):

        alpha = 0.01;
        batch_size = X.shape[0];


        for x in range(iteration_count):
            print('Iteration #{}'.format(x))
            errors = np.zeros((batch_size,self.output_size))
            for y in range(batch_size):
                errors[y,:] = labels[y]-self.predict(self,data) * self.transfer_derivative(X)
                self.update_weights




'''   # For the sake of simplicity use Mean Squared Error
        for x in range(iteration_count):
            print('Iteration #{}'.format(x))
            errors = np.zeros((batch_size, self.output_size))
            for y in range(batch_size):
                errors[y, :] = (self.__cnn(data[x * batch_size + y]) - labels[x * batch_size + y]) ** 2
            self.__backpropagation(np.mean(errors, axis=1))
'''

 '''  def test(self, data, labels, X):
    
        """
            Description: Test the ConvNet
            Parameters:
                data -> The data to be used for testing
                labels -> The labels to be used for testing
        """

        good = 0
        amount_tested = np.shape(data)[0]
        for x in range(np.shape(data)[0]):
            if np.argmax(self.predict(self,X) == np.argmax(labels[x, :]):
                good += 1

        accuracy = good / amount_tested
        return accuracy
'''