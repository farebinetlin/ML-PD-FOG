

library(randomForest)
# Assume you have loaded the data and stored it in a data.frame
# Assume the last column is the target variable, and the other columns are features

# Set the number of iterations and the number of folds for cross-validation
num_iterations <- 100
num_folds <- 10

# Store the importance scores for each feature
feature_importance_list <- list()

for (i in 1:num_iterations) {
  # Randomly split the data into a 3:1 training and testing set
  set.seed(i)  # Set the random seed for reproducibility
  sample_indices <- sample(1:nrow(your_data), nrow(your_data) * 0.75)
  train_data <- your_data[sample_indices, ]
  test_data <- your_data[-sample_indices, ]
  
  # Split the training and testing sets
  X_train <- train_data[, 1:(ncol(train_data) - 1)]
  y_train <- train_data[, ncol(train_data)]
  
  # Store the importance scores for each feature
  feature_importance <- matrix(0, nrow = num_folds, ncol = ncol(X_train))
  
  # Perform cross-validation
  for (fold in 1:num_folds) {
    # Split the training and validation sets for each cross-validation fold
    set.seed(fold)  # Set the random seed for each cross-validation fold
    
    # Train a tree model (e.g., random forest)
    rf_model <- randomForest(X_train, y_train, ntree = 100)  # Adjust parameters as needed
    
    # Record the importance scores for each feature
    feature_importance[fold, ] <- rf_model$importance[, 1]
  }
  
  # Store the feature importance scores in a list
  feature_importance_list[[i]] <- feature_importance
}

# Calculate the average score for each feature
average_feature_importance <- colMeans(simplify2array(feature_importance_list))

# Get the feature names
feature_names <- colnames(X_train)

# Create a data frame with feature names and average scores
feature_importance_df <- data.frame(Feature = feature_names, Average_Importance = average_feature_importance)

# Count the number of times each element in a row exceeds 0.5
count_over_0.5 <- apply(feature_importance_df > 0.5, 1, sum)

# Print the results
print(count_over_0.5)

feature_count_over_0.5_df <- data.frame(Feature = feature_names, count_over_0.5 = count_over_0.5)

# Export as a CSV file
write.csv(feature_count_over_0.5_df, file = "feature_feature_count_over_0.5_df-TREE.csv", row.names = FALSE)

