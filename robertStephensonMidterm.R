# Robert Stephenson Midterm IS470

# Here we are initializing the libraries we may need.
library(tidyverse)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(Metrics)
library(randomForest)
library(naivebayes)
library(rpart)
library(DescTools)
library(klaR)
library(modelr)

# We will now load in the training data
labelDataRaw <- read.csv("websites_labelled.csv")

# Finding how many missing values there are and removing them.
sum(is.na(labelDataRaw))
labelData <- na.omit(labelDataRaw)

#converting good/bad to 0/1
labelData$label <- ifelse(labelData$label == "bad", 1, 0)

#removing unneeded variables.
labelData <- subset(labelData, select = -c(unique_id, ip_add, website_domain))

# We are now going to create a  simple logistic regression model to 
# get some measurements on an un-tweaked algorithm using the 
# independent test set method.

# Making the training/testing split
set.seed(2112)
indx <- sample(nrow(labelData), nrow(labelData) * 0.80)
roughTrain <- labelData[indx, ]
roughTest <- labelData[-indx, ]

# Making the logistic regression model
roughLrModel <- glm(label ~ ., data = roughTrain, family = binomial)
summary(roughLrModel)

# Making predictions with our rough model
roughLrPredictions <- predict(roughLrModel, newdata = roughTest, 
                                type = "response")
roughLrPredictions <- ifelse(roughLrPredictions >= 0.5, 1, 0)
roughLrPredictions

# Tabulating observed vs. actual observations
roughLrTab <- table(roughTest$label, roughLrPredictions)
roughLrTab

# calculating accuracy
sum(diag(roughLrTab))/sum(roughLrTab)

# Calculating Precision
7178/(7178 + 14)

# This model seems to have a high accuracy, recall, and precision.
# We will still try some input engineering methods and other
# algorithms to see if we can get a more accurate model.

# Here we are trying to spread some of the columns to see if our
# model improves.
spreadTrain <- roughTrain %>% 
  mutate(value = 1) %>%
  spread(server_loc, value, fill = 0)

spreadTest <- roughTest %>% 
  mutate(value = 1) %>%
  spread(server_loc, value, fill = 0)

spreadLrModel <- glm(label ~ ., data = spreadTrain, 
                     family = binomial)
summary(spreadLrModel)

spreadLrPredictions <- predict(spreadLrModel, newdata = spreadTest, 
                              type = "response")
spreadLrPredictions <- ifelse(spreadLrPredictions >= 0.5, 1, 0)

spreadLrTab <- table(spreadTest$label, spreadLrPredictions)
spreadLrTab

# We can see that spreading the columns made no difference in
# the results of our model. We will now see if up-sampling will help.

train.up <- upSample(x = roughTrain %>% 
                       dplyr::select(-label), 
                     y = as.factor(roughTrain$label))

upLrModel <- glm(train.up$Class ~ ., data = train.up, 
                     family = binomial)
summary(upLrModel)

upLrPredictions <- predict(upLrModel, newdata = roughTest, 
                               type = "response")
upLrPredictions <- ifelse(upLrPredictions >= 0.5, 1, 0)

upLrTab <- table(roughTest$label, upLrPredictions)
upLrTab
sum(diag(upLrTab))/sum(upLrTab)
6973/(6973 + 3)

# While up-sampling slightly decreased our accuracy we see a much 
# greater precision which is exactly what we want for this problem.

# Making plot to compare error metrics
compared <- read.csv("compareData.csv")

ggplot(compared, aes(x = Error.Metric, y = Percent.Value)) + 
  geom_point(size = 3) + 
  geom_segment(aes(x = Error.Metric, 
                   xend = Error.Metric, 
                   y = 97, 
                   yend = Percent.Value)) + 
  labs(title = "Up-Sampling Precision Improvement", 
       subtitle = "Up-sampled vs. Rough Model") + 
  theme(axis.text.x = element_text(angle = 65, vjust = 0.6))

# Making plot to compare false positives
plotFP <- read.csv("plotFP.csv")

ggplot(plotFP, aes(x = Model.Type, y = False.Positives)) + 
  geom_point(size=3) + 
  geom_segment(aes(x = Model.Type, 
                   xend = Model.Type, 
                   y = 0, 
                   yend = False.Positives)) + 
  labs(title="False Positives", 
       subtitle="Up-sampled vs. Rough Model") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

# I will now train what is our best model (The up-sampled model)
# many times to build confidence in the model.

control.final <- trainControl(method = "LGOCV",
                              p = 0.8,
                              number = 250)

final.model <- train(Class ~ ., 
                     data = train.up, 
                     method = "glm",
                     trControl = control.final)
summary(final.model)

final.pred <- predict(final.model, newdata = roughTest)
final.pred

final.tab <- table(roughTest$label, final.pred)
final.tab

# We can see that we get the same final results from our up-sampled
# model even after testing on 250 random independent test sets.
sum(diag(final.tab))/sum(final.tab)
6973/(6973 + 3)

#We will now make predictions for the unlabeled websites.
unlabData <- read.csv("websites_unlabelled.csv")

sum(is.na(unlabData))
unlabData <- na.omit(unlabData)

new.pred <- predict(final.model, newdata = unlabData)
new.pred

newLabelData <- add_predictions(unlabData, final.model, 
                                var = "Predictions")

# We will now save the new labelled data and the final model

write.csv(newLabelData, file = "websites_new_append.csv")

saveRDS(final.model, file = "finalGlmStephenson.rda")
