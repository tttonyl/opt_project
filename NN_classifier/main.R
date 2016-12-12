
library(lbfgsb3)
## Initialization
rm(list=ls())
sources <- c("lbfgsb3_.R","nnCostFunction.R","predict.R","randInitializeWeights.R",
             "sigmoid.R","sigmoidGradient.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}


input_layer_size  <- 784  # 28x28 Input Images of Digits
hidden_layer_size <- 30   # 25 hidden units
num_labels <- 10          # 10 labels, from 1 to 10
                          # mapped "0" 3to label 10




load('train_dat.Rda')
list2env(train_dat,.GlobalEnv)
rm(train_dat)

m <- dim(X)[1]


initial_Theta1 <- randInitializeWeights(input_layer_size, hidden_layer_size, 123)
initial_Theta2 <- randInitializeWeights(hidden_layer_size, num_labels, 123 )

# Unroll parameters
initial_nn_params <- c(initial_Theta1,initial_Theta2)

# lambda = 1
# Training Set Accuracy: 0.793800
# 0.769500

# lambda = 0.05
# Training Set Accuracy: 0.748700
# 0.731000

lambda <- 0.09

# closure over nn_params
costFunction <- nnCostFunction(input_layer_size, hidden_layer_size, 
                                   num_labels, X, y, lambda)

gradFunction <- nnGradFunction(input_layer_size, hidden_layer_size, 
                               num_labels, X, y, lambda)

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

# predict

pred <- predict(Theta1, Theta2, X)

cat(sprintf('\nTraining Set Accuracy: %f\n', mean(pred==y)))


load('test_dat.Rda')
list2env(test_dat,.GlobalEnv)
pred <- predict(Theta1, Theta2, X)
cat(sprintf('\nTesting Set Accuracy: %f\n', mean(pred==y)))


