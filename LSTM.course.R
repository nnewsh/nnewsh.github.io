install.packages("keras")
library(keras)
install_keras()


install.packages("tensorflow")
install.packages("keras")
install.packages("tibble")
install.packages("tidyverse")
install.packages("readr")

library(tensorflow)
library(keras)
library(tibble)
library(tidyverse)
library(readr)

#setwd


##Data Processing ----
#Read in our spatial panel of 20 years (rows) and x areas (columns)
Data <- read.csv("UK.csv")
#Inspect our data:
head(Data, 10)

#Define the number of areas and variables - essential for subsequent data wrangling and LSTM model:
num_areas <- nrow(Data)
num_variables <- 1

#Convert spatial panel data to a long format matrix
Data.L <- Data %>% 
  pivot_longer(cols = starts_with('X2'), names_to = "Year", values_to = "Population") %>% 
  mutate(Year = as.numeric(sub("X", "", Year)))

data <- data.matrix(Data.L[, c(4)])

# Add row names for identification
row_names <- paste0("Area", rep(1:num_areas, each = 21), ".", rep(2000:2020, times = num_areas))
rownames(data) <- row_names

head(data, 25)

#Clear dataframes not in use
rm(Data, Data.L, row_names)

index <- rep(rep(c(1, 2), c(15, 6)), length.out = nrow(data))
head(index, 21)

train <- data[index ==1, , drop = FALSE]
head(train, 15)

mean <- apply(train, 2, mean)
std <- apply(train, 2, sd)

normalized_data <- scale(data, center = mean, scale = std)

# Remove dataframes not in use
rm(train, data, index)

##Sliding Window Process ----
#We first establish the sliding window sequences of length 5:
window_size <- 5

#Create empty lists to store sequences and targets for each area
#Results in
sequences <- vector("list", num_areas)
targets <- vector("list", num_areas)

#Given that each sequence consists of 5 consecutive population values and is accompanied by a target, the 6th value, *each area contains 16 sequences

for (i in 1:num_areas) {
  #Calculate the starting row index for the current area
  start_row <- (i - 1) * 21 + 1
  
  #Extract the population data for the current area
  area_data <- normalized_data[start_row:(start_row + 21 - 1), "Population"]
  
  #Create empty matrices to store sequences and targets for the current area
  area_sequences <- matrix(0, nrow = length(area_data) - window_size, ncol = window_size)
  area_targets <- matrix(0, nrow = length(area_data) - window_size, ncol = 1)
  
  #Generate sequences and targets for the current area
  for (j in 1:(length(area_data) - window_size)) {
    # Extract the sliding window sequence from the area data
    area_sequences[j, ] <- as.vector(t(area_data[j:(j + window_size - 1)]))
    
    #Extract the target value for the next time step
    area_targets[j, ] <- area_data[j + window_size]
  }
  
  #Store the sequences and targets in the respective lists
  sequences[[i]] <- area_sequences
  targets[[i]] <- area_targets
  
}

# Combine sequences and targets for all areas
all_sequences <- do.call(rbind, sequences)
all_targets <- do.call(rbind, targets)

print(normalized_data[16:21])

print(all_sequences[16, ])

print(all_targets[16, ])

##Train/Test Split ----
index <- rep(rep(c(1, 2), c(15, 1)), length.out = nrow(all_sequences))

train_data <- all_sequences[index == 1, , drop = FALSE]
train_targets <- all_targets[index == 1, , drop = FALSE]

test_data <- all_sequences[index == 2, , drop = FALSE]
test_targets <- all_targets[index == 2, , drop = FALSE]

#reshape train and test data to 3d array:
train_data <- array_reshape(train_data, c(dim(train_data)[1], window_size, 1))
test_data <- array_reshape(test_data, c(dim(test_data)[1], window_size, 1))

##Training an LSTM ----
set.seed(123) 
epochs <- 5
batch_size <- 15

#Build the LSTM model
model <- keras_model_sequential()
model %>% 
  layer_lstm(units = 32, input_shape = c(window_size, 1), activation = 'relu') %>%
  layer_dense(units = 1)

#Compile the model
model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001))

#Train the model
history <- model %>% fit( 
  train_data, train_targets,
  epochs = epochs,
  batch_size = batch_size,
  validation_data = list(test_data, test_targets)
)

##Forecasting Population ----
predicted_targets <- model %>% predict(test_data)

#Denormalize the predictions
predicted_targets <- predicted_targets *  std[1] + mean[1]

head(predicted_targets)


actual.values <- read.csv("C:/Users/niall/Downloads/UK.csv")
actual.values <- actual.values[c(1,23)]

Accuracy <- cbind(actual.values,predicted_targets)

Accuracy$APE <- abs((Accuracy$X2020 - Accuracy$predicted_targets) / Accuracy$X2020) * 100

mean(Accuracy$APE)
median(Accuracy$APE)


