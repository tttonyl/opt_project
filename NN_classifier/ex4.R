

## Initialization
rm(list=ls())
sources <- c("checkNNGradients.R","computeNumericalGradient.R",
             "debugInitializeWeights.R","displayData.R","lbfgsb3_.R",
             "nnCostFunction.R","predict.R","randInitializeWeights.R",
             "sigmoid.R","sigmoidGradient.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}


input_layer_size  <- 784  # 28x28 Input Images of Digits
hidden_layer_size <- 25   # 25 hidden units
num_labels <- 10          # 10 labels, from 1 to 10
                          # mapped "0" to label 10


# Load Training Data
cat(sprintf('Loading and Visualizing Data ...\n'))

load('train_dat.Rda')
list2env(train_dat,.GlobalEnv)
rm(train_dat)

m <- dim(X)[1]

# Randomly select 100 data points to display
sel <- sample(m)
sel <- sel[1:100]

displayData(X[sel,])

cat(sprintf('Program paused. Press enter to continue.\n'))
line <- readLines(con = stdin(),1)


cat(sprintf('\nLoading Saved Neural Network Parameters ...\n'))



cat(sprintf('\nInitializing Neural Network Parameters ...\n'))

initial_Theta1 <- randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 <- randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params <- c(initial_Theta1,initial_Theta2)


## -------------------- Part 8: Training NN --------------------
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
cat(sprintf('\nTraining Neural Network... \n'))

#  You should also try different values of lambda
lambda <- 1

# Create "short hand" for the cost function to be minimized
costFunction <- nnCostFunction(input_layer_size, hidden_layer_size, 
                                   num_labels, X, y, lambda) #over nn_params

gradFunction <- nnGradFunction(input_layer_size, hidden_layer_size, 
                               num_labels, X, y, lambda) #over nn_params

# Now, costFunction and gradFunction are functions that take in only one argument (the
# neural network parameters)

# lbfgsb3 works like fmincg (fast)
library(lbfgsb3)

# After you have completed the assignment, change the maxit to a larger
# value to see how more training helps.
# opt <- lbfgsb3(initial_nn_params, fn= costFunction, gr=gradFunction,control = list(trace=1,maxit=50))
opt <- lbfgsb3_(initial_nn_params, fn= costFunction, gr=gradFunction,
               control = list(trace=1,maxit=50))

nn_params <- opt$prm
cost <- opt$f

# Obtain Theta1 and Theta2 back from nn_params
Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                 hidden_layer_size, (input_layer_size + 1))

Theta2 <- matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
                 num_labels, (hidden_layer_size + 1))

cat(sprintf('Program paused. Press enter to continue.\n'))
line <- readLines(con = stdin(),1)


## ------------------- Part 10: Implement Predict -------------------
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred <- predict(Theta1, Theta2, X)

cat(sprintf('\nTraining Set Accuracy: %f\n', mean(pred==y) * 100))

cat('Program paused. Press enter to continue.\n')
line <- readLines(con = stdin(),1)

####### add some line of code to do test !!!!######################

load('test_dat.Rda')
list2env(test_dat,.GlobalEnv)
pred <- predict(Theta1, Theta2, X)
cat(sprintf('\nTesting Set Accuracy: %f\n', mean(pred==y) * 100))
#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

