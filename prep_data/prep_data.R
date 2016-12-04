source("load_mnist.R")
load_mnist()
list2env(train,.GlobalEnv)
rm(train)
rm(test)

# sample 10000 train data and 2000 testing data
train_y = c()
train_x = c()
test_y = c()
test_x = c()
for ( i in 0:9){
  ind = which(y %in% i)
  ind_train = ind[1:1000]
  ind_test = ind[1001:1200]
  
  train_y = c( train_y, y[ind_train] )
  test_y = c( test_y, y[ind_test] )
  
  train_x = rbind( train_x, X[ind_train,] )
  test_x = rbind( test_x, X[ind_test,] )
}
# mapping 0 to 10
train_y[1:1000] = 10
test_y[1:200] = 10

train_dat = list()
train_dat$X = train_x
train_dat$y = train_y
test_dat = list ()
test_dat$X = test_x
test_dat$y = test_y
save( train_dat, file = "train_dat.Rda" )
save( test_dat, file = "test_dat.Rda" )

rm( list = ls() )



