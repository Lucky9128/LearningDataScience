#Data Preprocessing
dataset = read.csv('Preprocessing/Data.csv')
dataset = dataset[,2:3]
#spliting data into training and test set
#install.packages('caTools')
library(caTools)
set.seed(1)
split = sample.split(dataset$Purchased,SplitRatio=0.8)
training_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split==FALSE)

#features scaling 
# training_set[,2:3] = scale(training_set[,2:3] )
# test_set[,2:3]  = scale(test_set[,2:3] )


  
