# load packages
library(readr)
library(gridExtra)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(lubridate)
library(rpart)
library(caret)
library(GGally)
library(gridExtra)
library(purrr)
library(e1071)
library(randomForest)

# read data
animals <- read.csv("data/train.csv")
head(animals)
num_obs <- nrow(animals)
levels(animals$OutcomeSubtype)

# Data Cleanup:
# Our code for data cleanup is adapted from the code in Quick and Dirty Random Forest notebook by Megan Risdal.

dim(animals)
str(animals)

# Replace blank names with "Nameless"
name_character <- as.character(animals$Name)
animals$Name <- ifelse(nchar(name_character) == 0, 'Nameless', as.character(animals$Name))

# Make a name v. no name variable
animals$HasName[animals$Name == 'Nameless'] <- 0
animals$HasName[animals$Name != 'Nameless'] <- 1
animals$HasName <- factor(animals$HasName)

# Extract time variables from date (uses the "lubridate" package)
animals$Hour    <- hour(animals$DateTime)
animals$Weekday <- wday(animals$DateTime)
animals$Month   <- month(animals$DateTime)
animals$Year    <- year(animals$DateTime)

# Time of day may also be useful
animals$TimeofDay <- ifelse(animals$Hour > 5 & animals$Hour < 11, 'morning',
                            ifelse(animals$Hour > 10 & animals$Hour < 16, 'midday',
                                   ifelse(animals$Hour > 15 & animals$Hour < 20, 'lateday', 'night')))
animals$TimeofDay <- factor(animals$TimeofDay)

# Use "grepl" to look for "Intact"
animals$Intact <- ifelse(grepl('Intact', animals$SexuponOutcome), 1,
                         ifelse(grepl('Unknown', animals$SexuponOutcome), 'Unknown', 0))

# Use "grepl" to look for sex
animals$Sex <- ifelse(grepl('Male', animals$SexuponOutcome), 'Male',
                      ifelse(grepl('Unknown', animals$Sex), 'Unknown', 'Female'))
animals$Intact <- factor(animals$Intact)
animals$Sex <- factor(animals$Sex)

# Convert to chracter
animals$AgeuponOutcome <- as.character(animals$AgeuponOutcome)
# Get the time value:
animals$TimeValue <- sapply(animals$AgeuponOutcome,
                            function(x) strsplit(x, split = ' ')[[1]][1])
# Now get the unit of time:
animals$UnitofTime <- sapply(animals$AgeuponOutcome,
                             function(x) strsplit(x, split = ' ')[[1]][2])
# Fortunately any "s" marks the plural, so we can just pull them all out
animals$UnitofTime <- gsub('s', '', animals$UnitofTime)
animals$TimeValue  <- as.numeric(animals$TimeValue)
animals$UnitofTime <- as.factor(animals$UnitofTime)

# Make a multiplier vector
multiplier <- ifelse(animals$UnitofTime == 'day', 1,
                     ifelse(animals$UnitofTime == 'week', 7,
                            ifelse(animals$UnitofTime == 'month', 30, # Close enough
                                   ifelse(animals$UnitofTime == 'year', 365, NA))))

# Apply our multiplier
animals$AgeinDays <- animals$TimeValue * multiplier
summary(animals$AgeinDays)

levels(factor(animals$Breed))[1:10]

# Use "grepl" to look for "Mix"
animals$IsMix <- ifelse(grepl('Mix', animals$Breed), 1, 0)
# Remove "Mix" and split on "/" and remove " Mix" to simplify Breed
animals$SimpleBreed <- sapply(animals$Breed,
                              function(x) gsub(' Mix', '', strsplit(as.character(x), split = '/')[[1]][1]))
animals$IsMix <- factor(factor(animals$IsMix))
animals$SimpleBreed <- factor(factor(animals$SimpleBreed))

# check the simplified breeds
levels(animals$SimpleBreed)[1:10]
str(animals$SimpleBreed)

# Use strsplit to grab the first color
animals$SimpleColor <- sapply(animals$Color,
                              function(x) strsplit(as.character(x), split = '/| ')[[1]][1])
animals$SimpleColor <- factor(animals$SimpleColor)
# check the simplified colors
levels(animals$SimpleColor)
str(factor(animals$SimpleColor))

for (i in seq_len(ncol(animals))) {
  print(names(animals[i]))
  print(sum(is.na(animals[[i]])))
}

for (i in seq_len(nrow(animals))) {
  if (is.na(animals$AgeinDays[[i]])) {
    animals$AgeinDays[[i]] <- median(animals$AgeinDays, na.rm = TRUE)
  }
}

animals$Lifestage[animals$AgeinDays < 365] <- 'baby'
animals$Lifestage[animals$AgeinDays >= 365] <- 'adult'
animals$Lifestage <- factor(animals$Lifestage)

head(animals)
str(animals)

# pair plot
animals %>%
  select(c("AgeinDays", "HasName", "TimeofDay", "IsMix", "AnimalType", "Sex", "Intact","OutcomeType")) %>%
  ggpairs()

# The code for data exploration below is also adapted from the code in Quick and Dirty Random Forest notebook by Megan Risdal.
outcomes <- animals[1:num_obs, ] %>%
  group_by(AnimalType, OutcomeType) %>%
  summarise(num_animals = n())

ggplot(outcomes, aes(x = AnimalType, y = num_animals, fill = OutcomeType)) +
  geom_bar(stat = 'identity', position = 'fill', colour = 'black') +
  coord_flip() +
  labs(y = 'Proportion of Animals',
       x = 'Animal Type',
       title = 'Outcomes by Animal Type: Cats & Dogs')

daytimes <- animals[1:num_obs, ] %>%
  group_by(AnimalType, TimeofDay, OutcomeType) %>%
  summarise(num_animals = n())

ggplot(daytimes, aes(x = TimeofDay, y = num_animals, fill = OutcomeType)) +
  geom_bar(stat = 'identity', position = 'fill', colour = 'black') +
  facet_wrap(~AnimalType) +
  coord_flip() +
  labs(y = 'Proportion of Animals',
       x = 'Time of Day ',
       title = 'Outcomes by Time of Day: Cats & Dogs')

intact <- animals[1:num_obs, ] %>%
  group_by(AnimalType, Intact, OutcomeType) %>%
  summarise(num_animals = n())

# Plot
ggplot(intact, aes(x = Intact, y = num_animals, fill = OutcomeType)) +
  geom_bar(stat = 'identity', position = 'fill', colour = 'black') +
  facet_wrap(~AnimalType) +
  coord_flip() +
  labs(y = 'Proportion of Animals',
       x = 'Animal Intactness',
       title = 'Outcomes by Intactness: Cats & Dogs')

# Initial train/test split ("estimation"/test) and cross-validation folds

set.seed(8348)
train_inds <- createDataPartition(
  y = animals$OutcomeType, # response variable as a vector
  p = 0.8 # approximate proportion of data used for training and validation
)
animals_train <- animals %>% slice(train_inds[[1]])
animals_test <- animals %>% slice(-train_inds[[1]])

crossval_val_fold_inds <- createFolds(
  y = animals_train$OutcomeType, # response variable as a vector
  k = 10 # number of folds for cross-validation
)
get_complementary_inds <- function(x) {
  return(seq_len(nrow(animals_train))[-x])
}

crossval_train_fold_inds <- map(crossval_val_fold_inds, get_complementary_inds)

# fit all variables with sample of 5000 observations
sample <- animals[sample(nrow(animals), 5000), ]

rf_fit <- train(
  form = OutcomeType ~ TimeofDay + Intact + Sex + AgeinDays + AnimalType + Lifestage + IsMix + SimpleBreed + SimpleColor + HasName,
  data = sample,
  method = "rf",
  trControl = trainControl(method = "oob"),
  tuneLength = 1
)

varImpPlot(rf_fit$finalModel, type = 2)

# data modification
animals_train$TimeofDay_reduced <- ifelse(grepl('morning', animals_train$TimeofDay), 'morning',
                                          ifelse(grepl('midday', animals_train$TimeofDay), 'midday', 'other'))

animals_test$TimeofDay_reduced <- ifelse(grepl('morning', animals_test$TimeofDay), 'morning',
                                         ifelse(grepl('midday', animals_test$TimeofDay), 'midday', 'other'))

animals_train$Color_reduced <- ifelse(grepl('Black', animals_train$Color), 'Black',
                                      ifelse(grepl('Brown', animals_train$Color), 'Brown',
                                             ifelse(grepl('White', animals_train$Color), 'White', 'other')))

animals_test$Color_reduced <- ifelse(grepl('Black', animals_test$Color), 'Black',
                                     ifelse(grepl('Brown', animals_test$Color), 'Brown',
                                            ifelse(grepl('White', animals_test$Color), 'White', 'other')))

# random forest model
rf_fit_var <- train(
  form = OutcomeType ~ TimeofDay_reduced + Intact + Sex + AgeinDays + AnimalType + Lifestage + IsMix + HasName + Color_reduced,
  data = animals_train,
  method = "rf",
  trControl = trainControl(method = "oob",
                           number = 10,
                           index = crossval_train_fold_inds,
                           indexOut = crossval_val_fold_inds,
                           returnResamp = "all",
                           savePredictions = TRUE),
  tuneLength = 10
)

# plot of the results of cross-validation for mtry
ggplot(data = rf_fit_var$results, mapping = aes(x = mtry, y = Accuracy)) +
  geom_line() +
  geom_vline(xintercept = rf_fit_var$bestTune$mtry)

# test-set performance
y_hats <- predict(rf_fit_var, newdata = animals_test)
rf_mce <- mean(y_hats != animals_test$OutcomeType)
rf_mce
```

## Out-of-class method: Support Vector Machine
# The code is adapted from ISLR – 9.6: Lab: Support Vector Machines by James, Witten, Hastie, and Tibshirani.
set.seed(1234)
# svm with gamma 1 and cost 1
svm_fit <- svm(
  formula = OutcomeType ~ TimeofDay_reduced + Intact + Sex + AgeinDays + AnimalType + Lifestage + IsMix + HasName + Color_reduced,
  data = animals_train %>% select(OutcomeType, TimeofDay_reduced, Intact, Sex, AgeinDays, AnimalType, Lifestage, IsMix, HasName, Color_reduced),
  kernel = 'radial',
  gamma = 1,
  cost = 1,
  scale = FALSE
)

summary(svm_fit)
y_preds <- predict(svm_fit, newdata = animals_test %>%
                     select(OutcomeType, TimeofDay_reduced, Intact, Sex, AgeinDays, AnimalType, Lifestage, IsMix, HasName, Color_reduced))

# confusion matrix
conf_matrix <- table(animals_test$OutcomeType, y_preds)
conf_matrix

# test-set perfomance
svm_mce <- 1 - sum(diag(conf_matrix))/length(y_preds)
svm_mce
