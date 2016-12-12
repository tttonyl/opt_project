randInitializeWeights <- function(L_in, L_out, seed) {
  W <- matrix(0,L_out, 1 + L_in)

  epsilon_init <- 0.086
  set.seed(seed)
  rnd <- runif(L_out * (1 + L_in))
  rnd <- matrix(rnd,L_out,1 + L_in)
  W <- rnd * 2 * epsilon_init - epsilon_init
  W
  
}
