nnCostFunction  <-
  function(input_layer_size, hidden_layer_size, num_labels,X, y, lambda) {
 
    
    function(nn_params) {
      Theta1 <-
        matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
               hidden_layer_size, (input_layer_size + 1))
      
      Theta2 <-
        matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
               num_labels, (hidden_layer_size + 1))
      m <- dim(X)[1]
      
      J <- 0
      
      # I <- diag(num_labels)
      # Y <- matrix(0, m, num_labels)
      # for (i in 1:m)
      #   Y[i,] <- I[y[i],]
      Y <- y
      
      a1 <- cbind(rep(1,m),X)
      z2 <- a1 %*% t(Theta1)
      a2 <- cbind(rep(1,dim(z2)[1]), sigmoid(z2))
      z3 <- a2 %*% t(Theta2)
      a3 <- sigmoid(z3)
      h <- a3
      
      p <- sum(Theta1[,-1] ^ 2) + sum(Theta2[,-1] ^ 2)
      
      J <-
        sum((-Y) * log(h) - (1 - Y) * log(1 - h)) / m + lambda * p / (2 * m)
      
      J
    }
  }

nnGradFunction  <-
  function(input_layer_size, hidden_layer_size, num_labels,
           X, y, lambda) {

    function(nn_params) {
      Theta1 <-
        matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
               hidden_layer_size, (input_layer_size + 1))
      # 1 * 31
      Theta2 <-
        matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
               num_labels, (hidden_layer_size + 1))
      # print(dim(Theta2))
      
      m <- dim(X)[1]
      
      
      Theta1_grad <- matrix(0,dim(Theta1)[1],dim(Theta1)[2])
      Theta2_grad <- matrix(0,dim(Theta2)[1],dim(Theta2)[2])
    
      # I <- diag(num_labels)
      # Y <- matrix(0, m, num_labels)
      # for (i in 1:m)
      #   Y[i,] <- I[y[i],]
      
      
      Y <- y
      # feedforward
      # 5000 785
      a1 <- cbind(rep(1,m),X)
      # 5000 785 785 30
      z2 <- a1 %*% t(Theta1)
      # 5000 31
      a2 <- cbind(rep(1,dim(z2)[1]), sigmoid(z2))
      # 5000 31 31 1
      z3 <- a2 %*% t(Theta2)
      # 5000 1
      a3 <- sigmoid(z3)
      h <- a3
      
      # backward propagation
      # 5000 1 
      sigma3 <- h - Y
      # 5000  31 * 5000 31 
      sigma2 <-
        (sigma3 %*% Theta2) * sigmoidGradient(cbind(rep(1,dim(z2)[1]),z2))
      # 5000 30
      sigma2 <- sigma2[,-1]
      
      # accumulate gradients
      # 30 5000 5000 785 = 30 785 
      delta_1 <- (t(sigma2) %*% a1)
      # 1 5000 5000 31 = 1 31
      delta_2 <- (t(sigma3) %*% a2)
      
      # calculate regularized gradient
      # theta1 30 785
      p1 <- (lambda / m) * cbind(rep(0,dim(Theta1)[1]), Theta1[,-1])
      # theta2 1 * 31
      
      p2 <- (lambda / m) * cbind(rep(0,dim(Theta2)[1]), t(Theta2[,-1]))
      # 30 785 + 30 785
      Theta1_grad <- delta_1 / m + p1
      # 1 31 + 
      Theta2_grad <- delta_2 / m + p2
      
      # Unroll gradients
      grad <-  c(c(Theta1_grad), c(Theta2_grad))
      grad
      # -------------------------------------------------------------
    }
  }