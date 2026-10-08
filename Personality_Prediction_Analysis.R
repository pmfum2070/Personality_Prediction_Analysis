####################### LOGISTICS REGRESSION ###########################

setwd("C:/Users/HP/Desktop/PERSONALITY PREDICTION")
set.seed(100)
library(tidymodels)

#DATA IMPORTATION
df1<- read.csv("personality_datasert.csv")
df2<- read.csv("2018-personality-data.csv")
View(df1)
view(df2)

#EDA
data_frame2<- select(df2, 2:6)
view(data_frame2)
data_frame1 <- df1[1:1834, ]
view(data_frame1)

#CONVERTING DECIMALS TO INTEGERS
data_frame2$openness <- floor(data_frame2$openness)
data_frame2$agreeableness <- floor(data_frame2$agreeableness)
data_frame2$emotional_stability <- floor(data_frame2$emotional_stability)
data_frame2$conscientiousness <- floor(data_frame2$conscientiousness)
data_frame2$extraversion <- floor(data_frame2$extraversion)
view(data_frame2)

#MERGING BY COLUMNS
merged_df <- cbind(data_frame1,data_frame2 )
view(merged_df)

#CONVERTING PREDICTOR VARIABLE CHARACTERS TO FACTORS
merged_df$Personality <- as.factor(merged_df$Personality)

#DATA PARTITIONING
pers_split <- initial_split(merged_df, prop = 0.70,
                            strata = Personality)

#POPULATING TRAINING AND TESTING DATA SET
pers_training<- pers_split%>%
  training()

pers_testing<- pers_split%>%
  testing()

glimpse(pers_training)
glimpse(pers_testing)

#MODEL SPECIFICATION
logistic_model<- logistic_reg()%>%
  set_engine("glm")%>%
  set_mode("classification")

#MODEL FITTING
logistics_fit<- logistic_model%>%
  fit(Personality ~.,
      data= pers_training)
logistics_fit

#PROBABILITY PREDICTION FOR OUTCOME VARIABLE
prob_pred<- logistics_fit%>%
  predict(new_data= pers_testing,,
          type= "prob")
prob_pred

#CLASS PREDICTION FOR OUTCOME VARIABLE 
class_pred<- logistics_fit%>%
  predict(new_data= pers_testing,,
          type= "class")
class_pred

#COMBINING RESULTS
pers_results<- pers_testing%>%
  select(Personality)%>%
  bind_cols(class_pred, prob_pred)
pers_results

#CONFUSION MATRIX
cm<- conf_mat(pers_results,
              truth= Personality,
              estimate= .pred_class)
cm

#MODEL EVALUATION (ACCURACY,SENSITIVITY,SPECIFICITY)
accuracy(pers_results,
         truth= Personality,
         estimate= .pred_class)

sens(pers_results,
     truth= Personality,
     estimate= .pred_class)

spec(pers_results,
     truth= Personality,
     estimate= .pred_class)

#VISUALIZING MODEL EVALUATIONS
autoplot(cm) + ggtitle("Confusion Matrix for Logistic Regression")

#ROC CURVE
pers_results%>%
  roc_curve(truth = Personality,.pred_Extrovert)%>%
  autoplot()

#CALCULATING AREA UNDER CURVE
pers_results%>%
  roc_auc(truth= Personality, .pred_Extrovert)


####################### DECISION TREE ###########################
library(rpart)
library(rpart.plot)

#DATA PARTITIONING
pers_split <- initial_split(merged_df, prop = 0.70,
                            strata = Personality)

#POPULATING TRAINING AND TESTING DATA SET
pers_train<- pers_split%>%
  training()

pers_test<- pers_split%>%
  testing()

glimpse(pers_train)
glimpse(pers_test)

#BUILDING DECISION TREE MODEL
pers_tree<- rpart(Personality~ Time_spent_Alone + Stage_fear + Social_event_attendance + 
                    Going_outside + Drained_after_socializing + Friends_circle_size + Post_frequency +
                    openness + agreeableness + emotional_stability + conscientiousness  + extraversion,
                  data = pers_training )
pers_tree

#PLOTTING DECISION TREE
rpart.plot(pers_tree, type= 5, extra= 104)

#MODEL SPECIFICATION
dt_model<-decision_tree()%>%
  set_engine("rpart")%>%
  set_mode("classification")
dt_model

#MODEL FITTING
dt_fit<- dt_model%>%
  fit(Personality~.,
      data= pers_train)
dt_fit

#CLASS PREDICTION FOR OUTCOME VARIABLE
dt_class_pred<- dt_fit%>%
  predict(new_data= pers_test,
          type="class")
dt_class_pred

#PROBABILITY PREDICTION FOR OUTCOME VARIABLE
dt_prob_pred<- dt_fit%>%
  predict(new_data= pers_test,
          type="prob")
dt_prob_pred

#COMBINING RESULTS
dt_results<- pers_test%>%
  select(Personality)%>%
  bind_cols(dt_class_pred,dt_prob_pred)
dt_results

#MODEL EVALUATION (CONFUSION MATRIX,ACCURACY,SENSITIVITY,SPECIFICITY)
conf_mat(dt_results,
         truth= Personality,
         estimate=.pred_class)

accuracy(dt_results,
         truth= Personality,
         estimate=.pred_class)

sens(dt_results,
     truth= Personality,
     estimate=.pred_class)

spec(dt_results,
     truth= Personality,
     estimate=.pred_class)

#ROC CURVE
dt_results%>%
  roc_curve(truth=Personality, .pred_Extrovert)%>%
  autoplot()

#CALCULATING AREA UNDER CURVE
dt_results%>%
  roc_auc(truth=Personality, .pred_Extrovert)

prune_control<- rpart.control(maxdepth = 7, minsplit = 2,
                              cp =0.01)
prune_control

#PLOT COST COMPLEXITY PARAMETER
dt_model_3<- rpart(Personality~.,data = pers_train, method = "class" )
plotcp(dt_model_3)
printcp(dt_model_3)

#INCORPORATING HYPERPARAMETERS CONTROL
tree_prune<-rpart(Personality~ Time_spent_Alone + Stage_fear + Social_event_attendance + 
                    Going_outside + Drained_after_socializing + Friends_circle_size + Post_frequency +
                    openness + agreeableness + emotional_stability + conscientiousness  + extraversion,
                  data= pers_train, control= prune_control)

rpart.plot(tree_prune, type=0, extra=0)

####################### K-NEAREST NEIGHBOUR ###########################

#CREATING FACTORS FOR CATEGORICAL VARIABLES
unique(merged_df$`Stage_fear`)
unique(merged_df$`Drained_after_socializing`)

merged_df_fac<- within(merged_df,{
  Stage_fear<-factor(Stage_fear,c("No","Yes"),
                     labels = c("0","1"))
  Drained_after_socializing<-factor(Drained_after_socializing,c("No","Yes"),
                                    labels = c("0","1"))
})

merged_df_fac
view(merged_df_fac)

#CONVERTING FACTORS TO NUMERIC
merged_df_fac$Stage_fear<- as.numeric(merged_df_fac$Stage_fear)
merged_df_fac$Drained_after_socializing<- as.numeric(merged_df_fac$Drained_after_socializing)
glimpse(merged_df_fac)

#DATA PARTITIONING
pers_knn_split<- initial_split(merged_df_fac,
                               prop = 0.75,
                               strata = Personality)
pers_knn_split

#POPULATING TRAINING AND TESTING DATA
pers_knn_train<- pers_knn_split%>%
  training()

pers_knn_test<- pers_knn_split%>%
  testing()

#CREATING LABELS FOR TRAINING AND TESTING DATASETS
train_labels <- pers_knn_train$Personality
test_labels <- pers_knn_test$Personality
train_labels
test_labels


#REMOVING TARGET VARIABLE FROM TRAINING AND TESTING FEATURES
train_features <- pers_knn_train %>% select(-Personality)
test_features <- pers_knn_test %>% select(-Personality)

#LOAD REQUIRED LIBRARIES
library(class)
library(caret)
library(e1071)
library(pROC)

#SCALING/NORMALIZING THE DATA
preproc <- preProcess(train_features, method = c("center", "scale"))
train_scaled <- predict(preproc, train_features)
test_scaled <- predict(preproc, test_features)

#FINDING OPTIMAL K VALUE
k_values <- seq(1, 21, by = 2)
accuracy_scores <- numeric(length(k_values))

#CROSS-VALIDATION TO FIND BEST K
for(i in seq_along(k_values)) {
  k <- k_values[i]
  knn_pred <- knn(train = train_scaled, 
                  test = train_scaled, 
                  cl = train_labels, 
                  k = k)
  accuracy_scores[i] <- mean(knn_pred == train_labels)
}

optimal_k <- k_values[which.max(accuracy_scores)]
cat("Optimal k value:", optimal_k, "\n")

#PLOT K VS ACCURACY 
par(mar = c(5, 4, 4, 2) + 0.1)
plot(k_values, accuracy_scores, type = "b", 
     xlab = "k Value", ylab = "Accuracy",
     main = "k-NN: Accuracy vs k Value")
abline(v = optimal_k, col = "red", lty = 2)

#TRAINING THE KNN MODEL WITH OPTIMAL K
knn_predictions <- knn(train = train_scaled,
                       test = test_scaled,
                       cl = train_labels,
                       k = optimal_k)

#MODEL EVALUATION
# Confusion Matrix
conf_matrix <- confusionMatrix(knn_predictions, as.factor(test_labels))
print(conf_matrix)

# Extract key metrics
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Recall']
f1_score <- conf_matrix$byClass['F1']

cat("\n=== MODEL PERFORMANCE METRICS ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n")

knn_prob <- knn(train = train_scaled,
                test = test_scaled,
                cl = train_labels,
                k = optimal_k,
                prob = TRUE)

# Extract probabilities
prob_values <- attr(knn_prob, "prob")
# Convert to probability of positive class
# KNN returns probability of predicted class, so we need to adjust
predicted_probs <- ifelse(knn_prob == levels(as.factor(train_labels))[1], 
                          1 - prob_values, prob_values)


# Binary classification
binary_actual <- as.numeric(as.factor(test_labels)) - 1

#CALCULATING ROC CURVE
roc_obj <- roc(binary_actual, predicted_probs)
auc_value <- auc(roc_obj)

#PLOT ROC CURVE 
par(mar = c(5, 4, 4, 2) + 0.1)
plot(roc_obj, 
     main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"),
     col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")

cat("\n=== ROC ANALYSIS ===\n")
cat("AUC:", round(auc_value, 4), "\n")
