# Load the data
# https://cran.r-project.org/web/packages/MASS/MASS.pdf
data("Boston", package = "MASS")

#install.packages("caret")
library(caret)

# Inspect the data
sample(Boston, 3)

summary(Boston)



#preProcValues <- preProcess(Boston, method = c("center", "scale", "nzv"))
preProcValues <- preProcess(Boston, method = c("range"))
data.centered.scaled <- predict(preProcValues, Boston)


# Split the data into training and test set
set.seed(2)
training.samples <- caret::createDataPartition(data.centered.scaled$medv, p = 0.8, list = FALSE)
train.data  <- data.centered.scaled[training.samples, ]
test.data <- data.centered.scaled[-training.samples, ]



fitControl <- trainControl(method = "repeatedcv", # validación cruzada con repetición
                           number = 5,            # número de paquetes de muestra
                           repeats = 3)           # para optimizar métricas 

tuneGrid = expand.grid(size=seq(from = 1, to = 10, by = 1),
                        decay = seq(from = 0.1, to = 0.5, by = 0.1))


rf_fit = caret::train(medv ~ ., data = train.data, method = "nnet",
                       trControl = fitControl,
                       tuneGrid=tuneGrid)


# Plot model error RMSE vs different values of k
plot(rf_fit)

# Best tuning parameter k that minimize the RMSE
rf_fit$bestTune

# Make predictions on the test data
predictions <- predict(rf_fit,test.data)
head(predictions)

# Compute the prediction error RMSE
RMSE(predictions, test.data$medv)

# plots real vs predicted values
x = 1:dim(test.data)[1]
plot(x, test.data$medv, col = "red", type = "l", lwd=2,
     main = "Boston housing test data prediction")
lines(x, predictions, col = "blue", lwd=2)
legend("topright",  legend = c("original-medv", "predicted-medv"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid() 
