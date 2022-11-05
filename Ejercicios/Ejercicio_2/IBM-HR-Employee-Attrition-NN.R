library(caret)

#install.packages('brio')
library(devtools)
library(roxygen2)


source_url("https://raw.githubusercontent.com/ragnemul/Neural-Network-samples-with-R/main/Ejercicios/Ejercicio_2/DrawConfusionMatrix.R")

data.df = read.csv("https://raw.githubusercontent.com/ragnemul/K-NN/main/Ejercicio_4.2/IBM-HR-Employee-Attrition.csv")
# Eliminamos datos no necesarios
# Eliminamos datos invariantes, variables independientes (no afecan al target), colineales múltiples, 

# eliminamos los valores nulos
data.df = na.omit(data.df)

# Eliminaos los valores innecesarios
data.df <- subset(data.df, select = -c(EmployeeCount,StandardHours,Over18,EmployeeNumber) )


# Converting  categorical variables to dummies
dmy <- dummyVars(Attrition ~ ., data = data.df, fullRank = T) 
data.dummies <- data.frame(predict(dmy, newdata = data.df)) 

# Añadimos columna Attrition con los valores Yes y No
data.dummies <- cbind(data.dummies, Attrition = c(data.df$Attrition))

# Fijamos semilla para inicializar datos aleatorios, así podremos obtener 
# repetitividad en los experimentos
set.seed(123)

# Particionamiento de los datos en conjuntos de entrenamiento y test
train_split_idx <- caret::createDataPartition(data.dummies$Attrition, p = 0.8, list = FALSE)
train <- data.dummies[train_split_idx, ]
test <- data.dummies[-train_split_idx, ]


fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 3, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)


rf_fit = caret::train(Attrition ~ ., data = train, method = "nnet",
                      trControl = fitControl,
                      preProcess = c("center","scale"))





######################################
# curvas ROC PARA COMPARAR LOS MODELOS

# para las curvas ROC
xtest <- subset(test, select = -c(Attrition))
ytest <- as.data.frame(test$Attrition)

pred.prob <- predict (rf_fit, newdata = xtest, type="prob")

roc_obj <- roc (ytest$`test$Attrition`, pred.prob[,1])
plot(roc_obj, print.auc = TRUE, print.auc.y = 0.6, col = "red")

# curvas ROC PARA COMPARAR LOS MODELOS
######################################



######################################
# Matriz de confusion

preds <- predict(rf_fit, newdata=xtest, type="raw")
caret::confusionMatrix(as.factor(preds), as.factor(test$Attrition),positive="Yes")

# Matriz de confusion
######################################

