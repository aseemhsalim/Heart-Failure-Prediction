################################################
## Data Analytics & Predictive Modeling Project
##  HEART FAILURE PREDICTION
##
##
################################################
getwd() # Check current working directory

# Update working directory
#setwd("/Users/aseemhsalim/Documents/MS IE/Classes/Fall23/Project/DAPM/Submissions/")
#getwd() # verify working directory

# Extracting csv data from heart.csv to df0
df0 = read.csv("heart.csv",sep = ",",header = TRUE)

library(corrplot)
library(ggplot2)
library(ggbeeswarm)
library(dplyr)
library(leaps) # Load library leaps to use regsubsets
library(caret)
library(rpart)
library(rpart.plot)
library(vip)
library(randomForest)


set.seed(1)

str(df0) # There are 299 observations with 13 variables.

summary(df0)

# Show column names
colnames(df0)

# There are 6 categorical variables in given dataset.
cat_cols = c("Event", "Gender", "Smoking", "Diabetes", "BP", "Anaemia")

sum(is.na(df0) == TRUE) # Get total number of missing values
# There are no missing values in given dataset.


boxplot(df0[,2]~ df0[,3], xlab = names(df0[3]), ylab = names(df0[2]), col = 3)

#single variable
op = par()

par(mfrow = c(2,3))

# Barplots of Categorical variables including response Event
barplot(table(df0$Event), col=c("lightblue","darkred"),
        main="Heart failure (0) vs. Ok (1)",
        xlab = "Heart failure", ylab="Count")
table(df0$Event)

barplot(table(df0$Gender), col=c("lightblue","darkred"),
        main="Female(0) vs Male(1)",
        xlab = "Gender", ylab="Count")
table(df0$Gender)

barplot(table(df0$Smoking), col=c("lightblue","darkred"),
        main="No smoking(0) vs Smoking(1)",
        xlab = "Smoking", ylab="Count")
table(df0$Smoking)


barplot(table(df0$Diabetes), col=c("lightblue","darkred"),
        main="No Diabetes(0) vs Diabetes(1)",
        xlab = "Diabetes", ylab="Count")
table(df0$Diabetes)

barplot(table(df0$BP), col=c("lightblue","darkred"),
        main="No BP(0) vs BP(1)",
        xlab = "BP", ylab="Count")
table(df0$BP)

barplot(table(df0$Anaemia), col=c("lightblue","darkred"),
        main="No Anaemia(0) vs Anaemia(1)",
        xlab = "Anaemia", ylab="Count")
table(df0$Anaemia)



#categorical response vs. categorical predictor
par(mfrow = c(1,1))

barplot(table(df0$Event, df0$Gender),
        beside=T,
        col=c("white","blue"),
        names.arg=c("Female","Male"),
        main="Barplot of Heart Failure by Gender")     

legend("topleft", legend=c("No Failure", "Heart Failed"),fill=c("white", "blue"))

barplot(table(df0$Event, df0$Smoking),
        beside=T,
        col=c("white","blue"),
        names.arg=c("No Smoking","Smoking"),
        main="Barplot of Heart Failure by Smoking")
legend("topright", legend=c("No Failure", "Heart Failed"),fill=c("white", "blue"))

barplot(table(df0$Event, df0$Diabetes),
        beside=T,
        col=c("white","blue"),
        names.arg=c("No Diabetes","Diabetes"),
        main="Barplot of Heart Failure by Diabetes")
legend("topright", legend=c("No Failure", "Heart Failed"),fill=c("white", "blue"))

barplot(table(df0$Event, df0$BP),
        beside=T,
        col=c("white","blue"),
        names.arg=c("No BP","BP"),
        main="Barplot of Heart Failure by BP")
legend("topright", legend=c("No Failure", "Heart Failed"),fill=c("white", "blue"))

barplot(table(df0$Event, df0$Anaemia),
        beside=T,
        col=c("white","blue"),
        names.arg=c("No Anaemia","Anaemia"),
        main="Barplot of Heart Failure by Anaemia")
legend("topright", legend=c("No Failure", "Heart Failed"),fill=c("white", "blue"))



# Categorical response vs Continuous predictors
par(mfrow = c(1,1))

plot(df0$TIME[df0$Event==0],df0$Event[df0$Event==0],
     pch="N",
     main="Follow up time v/s Heart Failure",
     col="green",
     ylim=c(0,1),
     xlab="Time",ylab="Heart Failure")

points(df0$TIME[df0$Event==1],df0$Event[df0$Event==1],pch="Y",col="red")

#abline(v=40,lty=2)

plot(df0$Age[df0$Event==0],df0$Event[df0$Event==0],
     pch="N",
     main="Age v/s Heart Failure",
     col="green",
     ylim=c(0,1),
     xlab="Age",ylab="Heart Failure")

points(df0$Age[df0$Event==1],df0$Event[df0$Event == 1],pch="Y",col="red")


plot(df0$Ejection.Fraction[df0$Event==0],df0$Event[df0$Event==0],
     pch="N",
     main="Ejection Fraction v/s Heart Failure",
     col="green",
     ylim=c(0,1),
     xlab="Ejection fraction",ylab="Heart Failure")

points(df0$Ejection.Fraction[df0$Event==1],df0$Event[df0$Event == 1],pch="Y",col="red")


plot(df0$Sodium[df0$Event==0],df0$Event[df0$Event==0],
     pch="N",
     main="Sodium v/s Heart Failure",
     col="green",
     ylim=c(0,1),
     xlab="Sodium",ylab="Heart Failure")

points(df0$Sodium[df0$Event==1],df0$Event[df0$Event == 1],pch="Y",col="red")


plot(df0$Creatinine[df0$Event==0],df0$Event[df0$Event==0],
     pch="N",
     main="Serum Creatinine v/s Heart Failure",
     col="green",
     ylim=c(0,1),
     xlab="Creatinine",ylab="Heart Failure")

points(df0$Creatinine[df0$Event==1],df0$Event[df0$Event == 1],pch="Y",col="red")


plot(df0$Pletelets[df0$Event==0],df0$Event[df0$Event==0],
     pch="N",
     main="Platelets v/s Heart Failure",
     col="green",
     ylim=c(0,1),
     xlab="Platelets",ylab="Heart Failure")

points(df0$Pletelets[df0$Event==1],df0$Event[df0$Event == 1],pch="Y",col="red")


plot(df0$CPK[df0$Event==0],df0$Event[df0$Event==0],
     pch="N",
     main="Creatinine phosphokinase v/s Heart Failure",
     col="green",
     ylim=c(0,1),
     xlab="CPK",ylab="Heart Failure")

points(df0$CPK[df0$Event==1],df0$Event[df0$Event == 1],pch="Y",col="red")



# Look for outliers
op = par()

par(mfrow = c(2,3))

for (i in 8:13) {
 boxplot(df0[,i], ylab = names(df0[i]), col = i-6)
}
mtext("Outlier detection for continuous variables", side = 3, line = - 2, outer = TRUE)

par(op)





df1 = df0
df1$Event = factor(df1$Event)
df1$Event <- recode(df1$Event, "0" = "No", "1" = "Yes")

#ggplot(iris, aes(Species, Sepal.Length)) + geom_beeswarm()

par(mfrow = c(2,3))

ggplot(df1, aes(Event, TIME,col = Event)) + 
  geom_beeswarm() + 
  scale_colour_manual(values = c("green","red")) +
  labs(x = "Heart Failure", y = "Follow-up Time", 
       title = "Follow-up Time v/s Heart Failure")


ggplot(df1, aes(Event, Age,col = Event)) + 
  geom_beeswarm() + 
  scale_colour_manual(values = c("green","red")) +
  labs(x = "Heart Failure", y = "Age", 
       title = "Age v/s Heart Failure")

ggplot(df1, aes(Event, Ejection.Fraction,col = Event)) + 
  geom_beeswarm() + 
  scale_colour_manual(values = c("green","red")) +
  labs(x = "Heart Failure", y = "Ejection Fraction", 
       title = "Ejection Fraction v/s Heart Failure")

ggplot(df1, aes(Event, Sodium,col = Event)) + 
  geom_beeswarm() + 
  scale_colour_manual(values = c("green","red")) +
  labs(x = "Heart Failure", y = "Sodium", 
       title = "Sodium v/s Heart Failure")

ggplot(df1, aes(Event, Creatinine,col = Event)) + 
  geom_beeswarm() + 
  scale_colour_manual(values = c("green","red")) +
  labs(x = "Heart Failure", y = "Creatinine", 
       title = "Creatinine v/s Heart Failure")

ggplot(df1, aes(Event, Pletelets,col = Event)) + 
  geom_beeswarm() + 
  scale_colour_manual(values = c("green","red")) +
  labs(x = "Heart Failure", y = "Platelets", 
       title = "Platelets v/s Heart Failure")

ggplot(df1, aes(Event, CPK,col = Event)) + 
  geom_beeswarm() + 
  scale_colour_manual(values = c("green","red")) +
  labs(x = "Heart Failure", y = "Creatinine phosphokinase", 
       title = "Creatinine phosphokinase v/s Heart Failure")



par(mfrow = c(1,1))
# corrplot(cor(df1[,7:13]),method = "pie", type = "upper")
# corrplot(cor(df1[,7:13]),method = "shade", type = "upper")
# corrplot(cor(df1[,7:13]),method = "square", type = "upper")
# corrplot(cor(df1[,7:13]),method = "ellipse", type = "upper")
# corrplot(cor(df1[,7:13]),method = "number", type = "upper", pch = 2)
#corrplot(cor(df1[,7:13]),method = "pie", type = "upper", is.corr = FALSE)
corrplot(cor(df1[,7:13]), method="color", 
         type="upper", order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)




# Split data into train and test sets (75% train and 25% test).
set.seed(123)

F_Split <- function(df, P_train){              # Split dataset Fn
  
  # Get percentage in decimals
  P_train = P_train/100
  
  nobs = round((nrow(df)) * P_train)
  #print(nobs)
  
  # Create train data indexes randomly
  s = sample(1:nrow(df),nobs)
  
  df_train1 = df[s,]
  df_test1  = df[-s,]
  
  return(list(df_test = df_test1, df_train = df_train1))
}

df2 = df1
df_split = F_Split(df2,75) # Split dataset to 75% train and 25% test set
df2_train = df_split$df_train
df2_test = df_split$df_test

dim(df2_train)
dim(df2_test)



# RMSE and R2 fucntions
rmse <- function(y_actual, y_pred) {            # RMSE function
  return (sqrt(mean((y_pred - y_actual)^2)))
}

r2 <- function(y_actual, y_pred) {              # R-squared function
  mean_y = mean(y_actual)
  return (1 - (sum((y_actual - y_pred)^2)/ sum((y_actual - mean_y)^2)))
}


# Outlier removal performance check
# Function to remove rows with outliers based on Z-score using training set statistics
remove_rows_with_outliers_zscore <- function(df, mean_train, sd_train, threshold = 3) {
  numeric_columns <- sapply(df, is.numeric)
  
  # Identify rows with outliers in any numeric column
  rows_to_remove <- apply(df[, numeric_columns], 1, function(row) {
    z_scores <- abs((row - mean_train) / sd_train)
    any(z_scores > threshold)
  })
  
  # Remove rows with outliers
  df[!rows_to_remove, ]
}

# Identify numeric and factor columns in the training set
numeric_columns_train <- sapply(df2_train, is.numeric)
factor_columns_train <- sapply(df2_train, is.factor)

# Check if there are numeric columns before calculating mean and sd
if (any(numeric_columns_train)) {
  # Calculate mean and standard deviation from the training set (df2_train)
  mean_train <- colMeans(df2_train[, numeric_columns_train], na.rm = TRUE)
  sd_train <- apply(df2_train[, numeric_columns_train], 2, sd, na.rm = TRUE)
  
  # Apply Z-score outlier removal to the training data (df2_train)
  df2_train_no_outliers <- remove_rows_with_outliers_zscore(df2_train, mean_train, sd_train)
  
  # Ensure factor levels are consistent
  for (col in names(df2_train_no_outliers)[factor_columns_train]) {
    df2_train_no_outliers[[col]] <- factor(df2_train_no_outliers[[col]], levels = levels(df2_train[[col]]))
  }
} else {
  # Handle the case where there are no numeric columns
  stop("No numeric columns found in df2_train.")
}

# Identify numeric and factor columns in the test set
numeric_columns_test <- sapply(df2_test, is.numeric)
factor_columns_test <- sapply(df2_test, is.factor)

# Check if there are numeric columns before calculating mean and sd
if (any(numeric_columns_test)) {
  # Apply Z-score outlier removal to the test data (df2_test) using training set statistics
  df2_test_no_outliers <- remove_rows_with_outliers_zscore(df2_test, mean_train, sd_train)
  
  # Ensure factor levels are consistent
  for (col in names(df2_test_no_outliers)[factor_columns_test]) {
    df2_test_no_outliers[[col]] <- factor(df2_test_no_outliers[[col]], levels = levels(df2_test[[col]]))
  }
} else {
  # Handle the case where there are no numeric columns
  stop("No numeric columns found in df2_test.")
}

# Visualize the original and cleaned data for both training and test sets
par(mfrow = c(2, 2))
boxplot(df2_train, main = "Training Data (Original)")
boxplot(df2_train_no_outliers, main = "Training Data (No Outliers)")

boxplot(df2_test, main = "Test Data (Original)")
boxplot(df2_test_no_outliers, main = "Test Data (No Outliers)")
par(mfrow = c(1, 1))


# Train a logistic regression model on the original training data (df2_train)
model_original <- glm(Event ~ ., data = df2_train, family = binomial)

# Evaluate the model on the original test data (df2_test)
original_test_predictions <- predict(model_original, newdata = df2_test, type = "response")
original_test_predictions_binary <- ifelse(original_test_predictions > 0.5, "Yes", "No")
original_test_performance <- confusionMatrix(as.factor(original_test_predictions_binary),
                                             as.factor(df2_test$Event))


# Retrain a logistic regression model on the cleaned training data (df2_train_no_outliers)
model_cleaned <- glm(Event ~ ., data = df2_train_no_outliers, family = binomial)

# Evaluate the cleaned model on the cleaned test data (df2_test_no_outliers)
cleaned_test_predictions <- predict(model_cleaned, newdata = df2_test_no_outliers, type = "response")
cleaned_test_predictions_binary <- ifelse(cleaned_test_predictions > 0.5, "Yes", "No")
cleaned_test_performance <- confusionMatrix(as.factor(cleaned_test_predictions_binary),
                                            as.factor(df2_test_no_outliers$Event))

# Compare performance metrics
print("Original Test Performance:")
print(original_test_performance)

print("Cleaned Test Performance:")
print(cleaned_test_performance)


# Install and load necessary libraries
library(pROC)

# Assuming cleaned_test_predictions and df2_test_no_outliers$Event are available

# Original model ROC
roc_original <- roc(df2_test$Event, original_test_predictions)
auc_original <- auc(roc_original)

# Cleaned model ROC
roc_cleaned <- roc(df2_test_no_outliers$Event, cleaned_test_predictions)
auc_cleaned <- auc(roc_cleaned)

# Plot ROC curves
plot(roc_original, col = "blue", main = "ROC Curve Comparison")
lines(roc_cleaned, col = "red")
legend("bottomright", legend = c(paste("Original Model (AUC =", round(auc_original, 3), ")"),
                                 paste("Cleaned Model (AUC =", round(auc_cleaned, 3), ")")),
       col = c("blue", "red"), lty = 1)

# Add diagonal reference line
abline(a = 0, b = 1, col = "gray", lty = 2)

# Outlier removal shows better AUC. But reduced accuracy.
# Plus those outlier values are expected when considering real data.
# Hence, continuing without outlier removal.



# Subset selection - Best subset

# Fit Best subset model(nvmax=12) to return 12 models
fit_best = regsubsets(Event ~ .,
                      data = df2_train,
                      method = "exhaustive",
                      nvmax = 12)


# Plot BIC, R2 and adjusted R2 for each model 
# Mark minimum or maximum values on each plot

op = par()                      # Save original setting
#par(mfrow = c(3,1))

L3 = fit_best

# Find max and min value of BIC
maxBIC = which.max(summary(L3)$bic)
minBIC = which.min(summary(L3)$bic)

# Find max and min value of R-Squared
maxR2 = which.max(summary(L3)$rsq)
minR2 = which.min(summary(L3)$rsq)

# Find max and min value of Adjusted R-Squared
maxAdjR2 = which.max(summary(L3)$adjr2)
minAdjR2 = which.min(summary(L3)$adjr2)

# Plot BIC
plot(summary(L3)$bic, xlab = "No. of Variables", ylab = "BIC")
points(maxBIC,summary(L3)$bic[maxBIC], col = "red", pch = 20, cex = 2)
points(minBIC,summary(L3)$bic[minBIC], col = "black", pch = 20, cex = 2)
lines(summary(L3)$bic, col = "blue")

# Plot R2
plot(summary(L3)$rsq, xlab = "No. of Variables", ylab = "R-Squared")
points(maxR2,summary(L3)$rsq[maxR2], col = "red", pch = 20, cex = 2)
points(minR2,summary(L3)$rsq[minR2], col = "black", pch = 20, cex = 2)
lines(summary(L3)$rsq, col = "blue")

# Plot Adjusted R2
plot(summary(L3)$adjr2, xlab = "No. of Variables", ylab = "Adj R-Squared")
points(maxAdjR2,summary(L3)$adjr2[maxAdjR2], col = "red", pch = 20, cex = 2)
points(minAdjR2,summary(L3)$adjr2[minAdjR2], col = "black", pch = 20, cex = 2)
lines(summary(L3)$adjr2, col = "blue")


#par(op)                         # reset the pars to defaults

# Best model as per Adj-R2 is 6
coef(L3,6) # Show estimates or coefficients of model with 6 subsets

# Best model as per BIC is 4
coef(L3,4) # Show estimates or coefficients of model with 4 subsets



# Extract the best subset of features based on minimum BIC
best_subset_summary <- summary(fit_best)
best_subset_features <- rownames(best_subset_summary$outmat)[which.min(best_subset_summary$bic)]

best_subset_features


# Extracting model 4 coefficient
estimates = coef(fit_best,4)


best_subset = c("Event", "TIME","Age", "Ejection.Fraction", "Creatinine")

df3_train = df2_train[,best_subset]
df3_test = df2_test[,best_subset]

best_subset_adjR2 = c("Event", "TIME","Age", "Ejection.Fraction", "Creatinine",
                      "Sodium", "CPK")
df4_train = df2_train[,best_subset_adjR2]
df4_test = df2_test[,best_subset_adjR2]


# Function to train and evaluate a logistic regression model
train_and_evaluate <- function(train_data, test_data, dataset_name) {
  # Train a logistic regression model
  model <- glm(Event ~ ., data = train_data, family = binomial)
  
  # Make predictions on the test data
  predictions <- predict(model, newdata = test_data, type = "response")
  
  # Convert probabilities to binary predictions
  predictions_binary <- ifelse(predictions > 0.5, "Yes", "No")
  
  # Ensure that factor levels match between predictions and actual target variable
  predictions_binary <- factor(predictions_binary, levels = levels(test_data$Event))
  
  # Evaluate model performance
  performance <- confusionMatrix(predictions_binary, test_data$Event)
  
  cat(paste("Performance for", dataset_name, ":\n"))
  print(performance)
  
  return(list(Accuracy = performance$overall["Accuracy"], Sensitivity = performance$byClass["Sensitivity"]))
}

# Compare model performance for df2, df3, and df4
performance_df2 <- train_and_evaluate(df2_train, df2_test, "Original Data")
performance_df3 <- train_and_evaluate(df3_train, df3_test, "BIC best subset")
performance_df4 <- train_and_evaluate(df4_train, df4_test, "AdjR2 best subset")

# Combine performance values into a data frame
performance_data <- data.frame(
  Dataset = c("Original Data", "BIC best subset", "AdjR2 best subset"),
  Accuracy = sapply(list(performance_df2, performance_df3, performance_df4), function(x) x$Accuracy),
  Sensitivity = sapply(list(performance_df2, performance_df3, performance_df4), function(x) x$Sensitivity)
)

# Create a bar plot with performance values
barplot(performance_data$Accuracy,
        names.arg = performance_data$Dataset,
        main = "Model Accuracy Comparison",
        ylab = "Accuracy",
        col = c("blue", "red", "green"),
        ylim = c(0, 1))

# Add text labels for performance values
text(seq_along(performance_data$Accuracy), performance_data$Accuracy, labels = round(performance_data$Accuracy, 2), pos = 3, col = "black")

# Add legend
legend("bottomright", legend = c("Original Data", "BIC best subset", "AdjR2 best subset"),
       fill = c("blue", "red", "green"))


# Best subset using BIC optimal gave best Accuracy






# Model Selection
library(boot)

set.seed(123)  # Set seed for reproducibility

# Define your models
logistic_model <- glm(Event ~ ., data = df3_train, family = binomial)
tree_model <- train(Event ~ ., data = df3_train, method = "rpart")
rf_model <- randomForest(Event ~ ., data = df3_train)


# Function to calculate model performance
calculate_performance <- function(model, data) {
  # Predict using appropriate type for caret models
  # print(model)
  
  
  if (inherits(model, "randomForest")) {
    binary_predictions <- predict(model, newdata = data, type = "response")
  } else if (model$method == "rpart") {
    binary_predictions <- predict(model, newdata = data)
  } else if (inherits(model, "glm")) {
    predictions <- predict(model, newdata = data, type = "response")
  } else {
    cat("Unsupported model method: ", model$method, "\n")
    stop("Please update the calculate_performance function to support this method.")
  }
  
  
  # print(predictions)
  if (inherits(model, "glm")) {
    binary_predictions <- ifelse(predictions > 0.5, "Yes", "No")
  }
  
  #print(length(binary_predictions))
  #print(length(data$Event))
  
  #print(factor(binary_predictions))
  #print(factor(data$Event))
  
  # Check if lengths match
  if (length(binary_predictions) != length(data$Event)) {
    stop("Lengths of binary_predictions and data$Event do not match.")
  }
  
  confusionMatrix(as.factor(binary_predictions),
                  as.factor(data$Event))$overall["Accuracy"]
}


# Bootstrap function for logistic regression
boot_logistic <- function(data, indices) {
  sample_data <- data[indices, ]
  model <- glm(Event ~ ., data = sample_data, family = binomial)
  return(calculate_performance(model, data))
}

# Bootstrap function for decision tree
boot_tree <- function(data, indices) {
  sample_data <- data[indices, ]
  model <- train(Event ~ ., data = sample_data, method = "rpart")
  return(calculate_performance(model, data))
}

# Bootstrap function for random forest
boot_rf <- function(data, indices) {
  sample_data <- data[indices, ]
  model <- randomForest(Event ~ ., data = sample_data)
  return(calculate_performance(model, data))
}




library(ggplot2)

# Define the numbers of iterations
iteration_values <- c(100, 1000, 5000)

# Initialize an empty data frame to store results
results_df <- data.frame(Model = character(),
                         Iterations = integer(),
                         Accuracy = numeric())

# Loop through different numbers of iterations
for (num_iterations in iteration_values) {
  cat("Number of Iterations:", num_iterations, "\n")
  
  # Perform bootstrapping for logistic regression
  results_logistic <- boot(df3_train, boot_logistic, R = num_iterations)
  
  # Perform bootstrapping for decision tree
  results_tree <- boot(df3_train, boot_tree, R = num_iterations)
  
  # Perform bootstrapping for random forest
  results_rf <- boot(df3_train, boot_rf, R = num_iterations)
  
  # Store results in the data frame
  results_df <- rbind(results_df,
                      data.frame(Model = c("Logistic", "Decision Tree", "Random Forest"),
                                 Iterations = rep(num_iterations, 3),
                                 Accuracy = c(mean(results_logistic$t),
                                              mean(results_tree$t),
                                              mean(results_rf$t))))
}

# Plot the results
ggplot(results_df, aes(x = Iterations, y = Accuracy, color = Model)) +
  geom_line() +
  labs(title = "Accuracy vs. Number of Iterations",
       x = "Number of Iterations",
       y = "Accuracy",
       color = "Model") +
  theme_minimal()

results_df



# # Model selection based on Bootstrapping
# # Number of bootstrap samples
# num_bootstraps <- 100
# data = df3_train
# 
# 
# # Bootstrap with logistic regression
# # Create an empty list to store results
# bootstrap_results <- list()
# # Set seed for reproducibility
# set.seed(123)
# 
# for (i in 1:num_bootstraps) {
#   # Create a bootstrap sample
#   bootstrap_indices <- sample(nrow(data), replace = TRUE)
#   bootstrap_data <- data[bootstrap_indices, ]
#   
#   # Split the bootstrap sample into training and testing sets
#   df_split = F_Split(df2,80) # Split dataset to 80% train and 20% test set
#   train_data = df_split$df_train
#   test_data = df_split$df_test
#   
#   # Create a logistic regression model
#   model <- glm(Event ~ ., data = train_data, family = binomial)
#   
#   # Make predictions on the test set
#   predictions <- predict(model, newdata = test_data, type = 'response')
#   
#   # Convert probabilities to binary predictions
#   threshold <- 0.5
#   predicted_classes <- ifelse(predictions > threshold, "Yes", "No")
#   
#   # Evaluate the model and store results
#   conf_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(test_data$Event))
#   bootstrap_results[[i]] <- conf_matrix$overall
# }
# 
# # Combine results from all bootstraps
# combined_results <- do.call(rbind, bootstrap_results)
# 
# # Calculate average performance metrics
# average_metrics <- colMeans(combined_results)
# 
# # Print the average performance metrics
# print(average_metrics)
# 
# 
# 
# 
# 
# 
# 
# # Bootstrap with Decision Tree
# # Create an empty list to store results
# bootstrap_results <- list()
# # Set seed for reproducibility
# set.seed(123)
# 
# for (i in 1:num_bootstraps) {
#   # Create a bootstrap sample
#   bootstrap_indices <- sample(nrow(data), replace = TRUE)
#   bootstrap_data <- data[bootstrap_indices, ]
#   
#   # Split the bootstrap sample into training and testing sets
#   df_split = F_Split(df2,80) # Split dataset to 75% train and 25% test set
#   train_data = df_split$df_train
#   test_data = df_split$df_test
#   
#   # Create a Decision tree model
#   model <- rpart(Event ~ ., data = train_data, method = "class")
#   
#   # Make predictions on the test set
#   predictions <- predict(model, newdata = test_data, type = "class")
#   
#   # Convert probabilities to binary predictions
#   #threshold <- 0.5
#   #predicted_classes <- ifelse(predictions > threshold, "Yes", "No")
#   
#   # Evaluate the model and store results
#   conf_matrix <- confusionMatrix(predictions, test_data$Event)
#   bootstrap_results[[i]] <- conf_matrix$overall
# }
# 
# # Combine results from all bootstraps
# combined_results <- do.call(rbind, bootstrap_results)
# 
# # Calculate average performance metrics
# average_metrics <- colMeans(combined_results)
# 
# # Print the average performance metrics
# print(average_metrics)
# 
# 
# 
# 
# # Bootstrap with Random forest
# # Create an empty list to store results
# bootstrap_results <- list()
# # Set seed for reproducibility
# set.seed(123)
# 
# for (i in 1:num_bootstraps) {
#   # Create a bootstrap sample
#   bootstrap_indices <- sample(nrow(data), replace = TRUE)
#   bootstrap_data <- data[bootstrap_indices, ]
#   
#   # Split the bootstrap sample into training and testing sets
#   df_split = F_Split(df2,80) # Split dataset to 75% train and 25% test set
#   train_data = df_split$df_train
#   test_data = df_split$df_test
#   
#   # Create a random forest model
#   model <- randomForest(Event ~ ., data = train_data, method = "class")
#   
#   # Make predictions on the test set
#   predictions <- predict(model, newdata = test_data, type = "class")
#   
#   # Evaluate the model and store results
#   conf_matrix <- confusionMatrix(predictions, test_data$Event)
#   bootstrap_results[[i]] <- conf_matrix$overall
# }
# 
# # Combine results from all bootstraps
# combined_results <- do.call(rbind, bootstrap_results)
# 
# # Calculate average performance metrics
# average_metrics <- colMeans(combined_results)
# 
# # Print the average performance metrics
# print(average_metrics)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # Log Regression
# D1_train = df3_train
# D1_test = df3_test
# 
# # Fit log reg model on train set
# glm.fits <- glm(
#   Event ~ .,
#   data = D1_train,
#   family = binomial
# )
# #summary(glm.fits)
# 
# # Apply the fitted model to the test dataset
# glm.probs <- predict(glm.fits, D1_test, type = "response")
# 
# # Plot the predicted probabilities versus actual probabilities 
# ggplot(D1_test, aes(x = glm.probs, y = Event)) + geom_point() +
#   geom_smooth(color = "red") +
#   labs(title = "Predicted Probability v/s Actual Probability",
#        x = "Predicted Probability",
#        y = "Actual Probability")
# 
# 
# # Discuss the performance of the fitted model. 
# threshold <- 0.5
# glm.pred <- ifelse(glm.probs > threshold, "Yes", "No")
# table(glm.pred, D1_test$Event)
# 
# mean(glm.pred == D1_test$Event)          # Prediction accuracy rate
# sprintf("Logistic Regression model misclassification test error: %f",
#         mean(glm.pred != D1_test$Event)) # Prediction error rate
# 
# conf_matrix <- confusionMatrix(as.factor(glm.pred), D1_test$Event)
# print(conf_matrix)
# 
# 
# 
# 
# 
# # Construct a decision tree model using rpart.
# decTree = rpart(Event ~ ., data = df2_train, method = "class")
# summary(decTree)
# 
# # Visualize and interpret decision tree model.
# # One aspect of interpretation is understanding important variables from dataset.
# rpart.plot(decTree) # Visualize decision tree model
# 
# 
# # Variable importance is usually determined by features used for splitting at 
# # nodes.
# vip(decTree)
# 
# 
# 
# # Evaluate the Decision Tree modeling performance by a confusion matrix
# pred_train = predict(decTree, df2_train, type = "class")
# 
# cm_train = table(pred_train, df2_train$Event)
# cm_train
# 
# pred_test = predict(decTree, df2_test, type = "class")
# 
# # Create a confusion matrix
# cm_test = table(pred_test, df2_test$Event)
# cm_test
# 
# 
# 
# # Train prediction CM
# confusionMatrix(df2_train$Event, pred_train)
# 
# # Test prediction CM
# confusionMatrix(df2_test$Event, pred_test)
# 
# 
# # Predict the outputs and calculate the percentage of correctly classified 
# # observations for both training and test
# accuracy_train <- mean(df2_train$Event == pred_train) * 100
# accuracy_train
# 
# accuracy_test <- mean(df2_test$Event == pred_test) * 100
# accuracy_test



# RANDOM FOREST

# Create a random forest model
rf_model <- randomForest(Event ~ ., data = df3_train)

# Make predictions on the test data
y_pred <- predict(rf_model, newdata = df3_test, type = "class")
ytrain_pred <- predict(rf_model, newdata = df3_train)

# Calculate the accuracy of the model
accuracy <- sum(y_pred == df3_test$Event) / nrow(df3_test)
cat("Test RF Model Accuracy:", accuracy, "\n")

# Test prediction CM
confusionMatrix(df3_test$Event, y_pred)

vip(rf_model)

plot(rf_model)


# Hyperparameter tuning
set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(df3_train$Event, p = 0.8, list = FALSE)
training_data <- df3_train[trainIndex, ]
testing_data <- df3_train[-trainIndex, ]

# Define the hyperparameter grid
mtry_values <- c(1, 2, 3, 4)  # Considering all features
ntree_values <- c(50, 100, 200, 300, 500)
nodesize_values <- c(5, 10, 15, 20)

# Initialize variables to store tuning results
accuracy_values <- array(NA, dim = c(length(mtry_values), length(ntree_values), length(nodesize_values)))

# Initialize variables for best hyperparameters
best_accuracy <- 0
best_mtry <- NULL
best_ntree <- NULL
best_nodesize <- NULL

# Perform manual grid search
for (i in seq_along(mtry_values)) {
  for (j in seq_along(ntree_values)) {
    for (k in seq_along(nodesize_values)) {
      mtry <- mtry_values[i]
      ntree <- ntree_values[j]
      nodesize <- nodesize_values[k]
      
      # Train the model
      custom_rf_model <- randomForest(
        Event ~ .,
        data = training_data,
        mtry = mtry,
        ntree = ntree,
        nodesize = nodesize
      )
      
      # Make predictions on the testing set
      predictions <- predict(custom_rf_model, newdata = testing_data)
      
      # Calculate accuracy
      accuracy <- sum(predictions == testing_data$Event) / length(testing_data$Event)
      
      # Update best hyperparameters if the current model is better
      if (accuracy > best_accuracy) {
        best_accuracy <- accuracy
        best_mtry <- mtry
        best_ntree <- ntree
        best_nodesize <- nodesize
      }
      
      # Store accuracy in the results array
      accuracy_values[i, j, k] <- accuracy
    }
  }
}

# Print the best hyperparameters
cat("Best mtry:", best_mtry, "\n")
cat("Best ntree:", best_ntree, "\n")
cat("Best nodesize:", best_nodesize, "\n")

# Create a plot for nodesize
library(ggplot2)
data_long_nodesize <- reshape2::melt(accuracy_values[, , , drop = FALSE], varnames = c("mtry", "ntree", "nodesize"))
ggplot(data_long_nodesize, aes(x = factor(nodesize), y = value, color = factor(mtry))) +
  geom_point(position = position_jitter(width = 0.1, height = 0.1), size = 3) +
  geom_line(aes(group = factor(mtry)), alpha = 0.7) +
  labs(title = "Nodesize Impact on Accuracy",
       x = "Nodesize",
       y = "Accuracy",
       color = "mtry") +
  theme_minimal()




# # Initialize variables for best hyperparameters
# best_accuracy <- 0
# best_mtry <- NULL
# best_ntree <- NULL
# best_nodesize <- NULL
# 
# # Initialize a variable to store accuracy values for plotting
# accuracy_vs_ntree <- data.frame(ntree = numeric(0), accuracy = numeric(0))
# 
# # Get the total number of features
# num_features <- ncol(training_data) - 1  # Exclude the target variable
# 
# # Perform manual grid search
# for (j in seq_along(ntree_values)) {
#   ntree <- ntree_values[j]
#   
#   # Choose a specific value or percentage for mtry
#   mtry <- min(4, floor(sqrt(num_features)))  # You can adjust this value as needed
#   
#   # Train the model
#   custom_rf_model <- randomForest(
#     Event ~ .,
#     data = training_data,
#     mtry = mtry,
#     ntree = ntree,
#     nodesize = best_nodesize  # Use the best nodesize found so far
#   )
#   
#   # Make predictions on the testing set
#   predictions <- predict(custom_rf_model, newdata = testing_data)
#   
#   # Calculate accuracy
#   accuracy <- sum(predictions == testing_data$Event) / length(testing_data$Event)
#   
#   # Update best hyperparameters if the current model is better
#   if (accuracy > best_accuracy) {
#     best_accuracy <- accuracy
#     best_ntree <- ntree
#     best_mtry <- mtry
#   }
#   
#   # Store accuracy values for plotting
#   accuracy_vs_ntree <- rbind(accuracy_vs_ntree, data.frame(ntree = ntree, accuracy = accuracy))
# }
# 
# # Print the best hyperparameters
# cat("Best mtry:", best_mtry, "\n")
# cat("Best ntree:", best_ntree, "\n")
# cat("Best nodesize:", best_nodesize, "\n")
# 
# # Create a plot for accuracy vs number of trees
# library(ggplot2)
# ggplot(accuracy_vs_ntree, aes(x = ntree, y = accuracy)) +
#   geom_line() +
#   geom_point() +
#   labs(title = "Accuracy vs Number of Trees",
#        x = "Number of Trees",
#        y = "Accuracy") +
#   theme_minimal()




# Train the best model on the entire training dataset
best_rf_model <- randomForest(
  Event ~ .,
  data = df3_train,
  mtry = best_mtry,
  ntree = best_ntree,
  nodesize = best_nodesize
)

# Make predictions on the test dataset
predictions <- predict(best_rf_model, newdata = df3_test)

conf_matrix <- confusionMatrix(predictions, df3_test$Event)
print(conf_matrix)

# Plot prediction performance
library(pROC)

roc_curve <- roc(df3_test$Event, as.numeric(predictions))
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
