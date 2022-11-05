library(neuralnet)
library(caret)

data = read.csv("https://raw.githubusercontent.com/ragnemul/Neural-Network-samples-with-R/main/Ejercicios/Ejercicio_1/RidingMowers.csv")

# eliminamos los valores nulos
data = na.omit(data)

# Especificando la misma semilla para la inicialización de los números aleatorios obtendremos los mismos resultados
set.seed(6548)

# --------------
# NORNMALIZACION

# Ahora normalizamos los datos. Esto se puede hacer de diferentes maneras

# Primeramente definimos la normalización a las columnas 1 y 2 del conjunto de datos 
norm.values <- preProcess(data[, 1:2], method=c("range"))
# y ahora aplicamos la normalización mediante la llamda a predict 
data.norm <- predict(norm.values, data[, 1:2])

# NORNMALIZACION
# --------------

# Añadimos la columna Owership a los datos normalizados
Ownership = data$Ownership
data.norm <- cbind(data.norm, Ownership)


# ---------------------------------
# MOSTRAMOS LA GRÁFICA DE LOS DATOS

plot(Lot_Size ~ Income, # campos a mostrar
     data=data.norm,    # fuente de los datos
     pch=ifelse(data$Ownership =="Owner", 1, 3),    # caracter a pintar en función de si es propietario
     col=ifelse(data$Ownership=="Owner","dark green","dark red"))    # color del caracter a pintar en función de si es propietario

# Podemos mostrar el número de registro dentro del conjunto de datos
text(data.norm$Income, data.norm$Lot_Size, rownames(data.norm), pos=4)
# mostramos la leyenda de la gráfica
legend("topright", # ubicación
       c("owner", "non-owner", "new"), # campos a mostrar
       pch = c(1, 3, 4),   # caracter a mostrar en función de owner, non-owner o new
       col=c("dark green","dark red","dark blue")) # color a mostrar en función de owner, non-owner o new

# MOSTRAMOS LA GRÁFICA DE LOS DATOS
# ---------------------------------


# Entrenamos la red neuronal para predicción de propietarios con una única neurona
nn <- neuralnet(Ownership =="Owner" ~ Income + Lot_Size, data = data.norm, linear.output = F, hidden=0)


predict <- compute(nn, data.frame(data.norm[1], data.norm[2]))
predicted.class=apply(predict$net.result,1,round)
confusionMatrix(as.factor(ifelse(predicted.class=="1", "Owner", "Nonowner")),as.factor(data$Ownership))


# Mostramos la recta que separa los dos conjuntos de datos
# -w0/w2 es el punto de corte con el eje de ordenadas
# -w1/w2 es la pendiente de la recta
w0 = nn$weights[[1]][[1]][1]
w1 = nn$weights[[1]][[1]][2]
w2 = nn$weights[[1]][[1]][3]
abline(a = -w0/w2, b = -w1/w2, col='dark green', lwd=3, lty=2)


# ---------------------------------
# MOSTRAMOS DATO A PREDECIR

X <- data.frame("Income"=60,"Lot_Size"=20)
# tenemos que normalizar el dato
X.norm <- predict(norm.values, X)
# mostramos la posición del nuevo dato
text(X.norm, "X", col="dark blue")

# Aplicamos la red neuronal al dato a predecir
X.predicted <- compute(nn, data.frame(X.norm[1],X.norm[2]))
# redondeamos la salida
apply(X.predicted$net.result,1,round)

# MOSTRAMOS DATO A PREDECIR
# ---------------------------------


# Mostramos la red neuronal
plot(nn, rep="best")

