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


fitControl <- trainControl(method = "repeatedcv", # validación cruzada con repetición
                           number = 5,            # número de paquetes de muestra
                           repeats = 3,           # repeticiones
                           classProbs = TRUE,     # clasificación
                           summaryFunction = twoClassSummary) # para optimizar métricas 

rf_fit = caret::train(Attrition ~ ., data = train, method = "mlpML",
                      trControl = fitControl,
                      preProcess = c("center","scale"))



tuneGrid2 = expand.grid(size=seq(from = 1, to = 10, by = 1),
            decay = seq(from = 0.1, to = 0.5, by = 0.1))

rf_fit2 = caret::train(Attrition ~ ., data = train, method = "nnet",
                      trControl = fitControl,
                      preProcess = c("center","scale"),
                      tuneGrid=tuneGrid2)


library(nnet)
train[,45] = ifelse(train[,45] == "No",0,1)
rf_fit3 <- nnet(Attrition~., data = train, size = 20, maxit = 5000, decay = .01)



######################################
# curvas ROC PARA COMPARAR LOS MODELOS
library(pROC)
# para las curvas ROC
xtest <- subset(test, select = -c(Attrition))
ytest <- as.data.frame(test$Attrition)

pred.prob <- predict (rf_fit, newdata = xtest, type="prob")
roc_obj <- roc (ytest$`test$Attrition`, pred.prob[,1])
plot(roc_obj, print.auc = TRUE, print.auc.y = 0.6, col = "red")

pred.prob <- predict (rf_fit2, newdata = xtest, type="prob")
roc_obj <- roc (ytest$`test$Attrition`, pred.prob[,1])
plot(roc_obj, print.auc = TRUE, print.auc.y = 0.5, col = "green", add=T)

pred.prob <- predict(rf_fit3, newdata = xtest, type="raw")
roc_obj <- roc (ytest$`test$Attrition`, pred.prob[,1])
plot(roc_obj, print.auc = TRUE, print.auc.y = 0.4, col = "blue", add=T)

# curvas ROC PARA COMPARAR LOS MODELOS
######################################





######################################
# Matriz de confusion

preds <- predict(rf_fit, newdata=xtest, type="raw")
confusion_matrix <- caret::confusionMatrix(as.factor(preds), as.factor(test$Attrition),positive="Yes")
# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = confusion_matrix, caption = "Matriz de confusión test 1")

preds <- predict(rf_fit2, newdata=xtest, type="raw")
confusion_matrix <- caret::confusionMatrix(as.factor(preds), as.factor(test$Attrition),positive="Yes")
# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = confusion_matrix, caption = "Matriz de confusión test 2")


ytest_num <- unlist(ifelse(ytest == "Yes",1,0))
preds <- unlist(round(predict(rf_fit3, newdata = xtest, type="raw")))
confusion_matrix <- caret::confusionMatrix(as.factor(preds), as.factor(ytest_num),positive="1")
draw_2D_confusion_matrix(cm = confusion_matrix, caption = "Matriz de confusión test 3")

# Matriz de confusion
######################################

source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
plot.nnet(rf_fit3)

