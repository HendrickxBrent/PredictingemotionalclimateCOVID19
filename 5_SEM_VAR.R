
library(corrplot)
library(lavaan)
library(semPlot)
library(semTools)
library(vars)
library(dplyr)

data <- n2023_forR_6

data_standardized <- scale(data)
data_standardized <- data
data_standardized[] <- lapply(data, scale)
data <- data_standardized[]
data$fear <- n2023_forR_6$fear
data$anger <- n2023_forR_6$anger

################################################
# Calculate the correlations between the variables
datacor <- data %>% select(-serie, -datecode, -joy,-love, -sadness, -surprise)
correlations <- cor(datacor)
corrplot(correlations, method = "color")


correlations <- cor(data)
# Plot the correlations
corrplot(correlations, method = "color")


################################## rearrange
df <- datacor
columns_to_keep <- names(df)[-1]
# Add the first column to the end of the list of columns to keep
columns_to_keep <- c(columns_to_keep, names(df)[1])
# Use the select function to rearrange the columns
df <- df %>% select(columns_to_keep)
datacor <-df
datacor <- datacor %>% select(-test)
correlations <- cor(datacor)
corrplot(correlations, method = "color")


### CFA




#### corona

model <- '
  f1a =~ covid + corona + virus + death + pandemic + symptom + hospit + case + patient + test3
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]


model <- '
  f1 =~ covid + corona + virus+ pandemic + symptom + hospit + case + patient + test3
  corona ~~ virus
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]



model <- '
  f1 =~ covid + corona + virus+ pandemic + symptom + hospit + case + patient
  corona ~~ virus
  covid ~~  corona
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]

model <- '
  Covid =~ pandemic + symptom + case + patient + covid + corona + virus  + hospit
  corona ~~ virus
  covid ~~  corona
  covid ~~   virus
  symptom ~~ patient
  pandemic ~~ symptom
  pandemic ~~     case
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 3)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]
fitcovid <- cfa(model, data)

### cfi: 0.986
### tli: 0.972
### rmsea: 0.08
### srmr: 0.018











###  Education
model <- '
  f1 =~ universi + study + school + teach + degre + educ + student + children
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]

###   Education
model <- '
  Education =~ school + children + student + teach + universi + study  + educ 
  school ~~ children
  student ~~ children
  school ~~   teach
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]
fitedu <- cfa(model, data)







#### restrictions
model <- '
  res =~ stay + protect + social + distanc + irrespons  + isol  + selfish 
  social ~~   distanc
  stay ~~   protect
  protect ~~   selfish
  stay ~~      isol
  distanc ~~      isol
  protect ~~   distanc
  irrespons ~~   selfish
  social ~~      isol
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]
fitrestrict <- cfa(model, data)





#### politics
model <- '
  f1 =~ govern + polit + report + johnson + minist + govt + propaganda + trust + defend + cabinet + decis + issu + incompet
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]



#### politics
model <- '
  pol =~ govern + polit + report + minist + govt + trust + defend + cabinet + decis + issu
  govern ~~   decis
  defend ~~ cabinet
  decis ~~    issu
  polit ~~  minist
  govern ~~    govt
  report ~~  defend
  polit ~~ cabinet
  polit ~~  defend
  minist ~~ cabinet
  govern ~~ report
  minist ~~  defend
  govern ~~  defend
'
fit <- cfa(model, data)
summary(fit, fit.measures = TRUE)
semPaths(fit,"model","stand",style="LISREL",rotation=1, 
         edge.color="black",sizeLat=4,layout = 'tree',
         edge.label.cex=0.6,mar=c(4,1,4,1),sizeMan = 4)
mi<-inspect(fit,"mi")
mi.sorted<- mi[order(-mi$mi),]  # sort from high to low
mi.sorted[1:10,]
fitpol <- cfa(model, data)




## FACTOR POLITICS
idx <- lavInspect(fitpol, "case.idx")
fscores <- lavPredict(fitpol)
## loop over factors
for (fs in colnames(fscores)) {
  data[idx, fs] <- fscores[ , fs]
}
data$Fpolitics <- data$f1

## FACTOR  RESTRICTIONS
idx <- lavInspect(fitrestrict, "case.idx")
fscores <- lavPredict(fitrestrict)
## loop over factors
for (fs in colnames(fscores)) {
  data[idx, fs] <- fscores[ , fs]
}
data$Frestrictions <- data$f1


## FACTOR EDUCATION
idx <- lavInspect(fitedu, "case.idx")
fscores <- lavPredict(fitedu)
## loop over factors
for (fs in colnames(fscores)) {
  data[idx, fs] <- fscores[ , fs]
}


## FACTOR COVID
idx <- lavInspect(fitcovid, "case.idx")
fscores <- lavPredict(fitcovid)
## loop over factors
for (fs in colnames(fscores)) {
  data[idx, fs] <- fscores[ , fs]
}



##############################################################################################################
#####                                    VAR
##############################################################################################################

## random variable with mean 0 and sd of 1
data$zero <- rnorm(nrow(data), mean = 0, sd = 1)


# Select the columns to use in the VAR model
data <- transform(data, coviddat = covid + corona + virus)

data2 <- data[, c("anger", "fear", "Covid", "Education", "res", "pol")]
tsdata <- ts(data2, frequency = 18, start = c(2020, 3))
fit <- VAR(tsdata, p = 1)

summary(fit)
devAskNewPage(ask = FALSE)
plot(fit)



### split data
first <- head(data, n = nrow(data) / 2)
last <- tail(data, n = nrow(data) / 2)

data2 <- data[, c("anger", "fear", "Covid", "Education", "Frestrictions", "Fpolitics")]
tsdata <- ts(data2, frequency = 18, start = c(2020, 3))
fit <- VAR(tsdata, p = 2)
summary(fit)
plot(fit)



data2 <- data[, c("anger", "Education", "res", "pol")]
tsdata <- ts(data2, frequency = 18, start = c(2020, 3))
fit <- VAR(tsdata, p = 2)
summary(fit)
plot(fit)
fevd(fit, n.ahead = 10)   # variance explained by variable


## separate
data2 <- data[, c("anger","Fpolitics")]
tsdata <- ts(data2, frequency = 18, start = c(2020, 3))
fit <- VAR(tsdata, p = 20)
summary(fit)
plot(fit)
fevd(fit, n.ahead = 10)   # variance explained by variable


data2 <- data[, c("anger", "Frestrictions")]
tsdata <- ts(data2, frequency = 18, start = c(2020, 3))
fit <- VAR(tsdata, p = 2)
summary(fit)
plot(fit)
fevd(fit, n.ahead = 10)   # variance explained by variable



data2 <- data[, c("fear", "Covid")]
tsdata <- ts(data2, frequency = 20, start = c(2020, 3))
fit <- VAR(tsdata, p = 2)
summary(fit)
plot(fit)

fevd(fit, n.ahead=20)   # variance explained by variable #the FEVD for two steps ahead, does not directly relate to the lags of the VAR model


##### testing times
data2 <- data[, c("fear", "Covid")]
first <- head(data2, n = nrow(data2) / 6)
tsdata <- ts(first, frequency = 18, start = c(2020, 3))
fit <- VAR(tsdata, p = 2)
summary(fit)
plot(fit)

data2 <- data[, c("fear", "zero")]
first <- head(data2, n = nrow(data2) / 6)
tsdata <- ts(first, frequency = 18, start = c(2020, 3))
fit <- VAR(tsdata, p = 2)
summary(fit)
plot(fit)



plot(fit, type = "irf")


fevd_results <- fevd(fit)

######################################### variance explained irf
irf.res <- irf(fit, impulse = "res", response = "anger", n.ahead = 40, boot = TRUE)
plot(irf.res, ylab = "anger", main = "Shock from restrictions")

irf.pol <- irf(fit, impulse = "pol", response = "anger", n.ahead = 40, boot = TRUE)
plot(irf.pol, ylab = "anger", main = "Shock from politics")

irf.edu <- irf(fit, impulse = "Education", response = "anger", n.ahead = 40, boot = TRUE)
plot(irf.edu, ylab = "anger", main = "Shock from education")



irf.covid <- irf(fit, impulse = "Covid", response = "fear", n.ahead = 40, boot = TRUE)
plot(irf.covid, ylab = "fear", main = "Shock from covid-19")






#################### residuals
residuals <- residuals(fit)
res <- as.data.frame(residuals)
resang <- res$anger
qqplot(res$anger, res$Education)
qqplot(res$anger, res$Frestrictions)
qqplot(res$anger, res$Fpolitics)

qqplot(res$fear, res$COVID)



Box.test(res$Education, lag=1, type="Ljung-Box")
Box.test(res$Frestrictions, lag=1, type="Ljung-Box")
Box.test(res$Fpolitics, lag=1, type="Ljung-Box")
Box.test(res$anger, lag=1, type="Ljung-Box")

qqplot(res$fear,res$Covid)

residuals <- residuals(fit)
res <- as.data.frame(residuals)
Box.test(res$fear, lag=1, type="Ljung-Box")
Box.test(res$Covid, lag=1, type="Ljung-Box")




### tolerance

# Load the 'rsq' package
library(rsq)
library(car)

# Compute the tolerance for each independent variable
condition_number <- conditionNumber(tsdata)
vifs <- car::vif(fit)

model_all <- lm(anger ~ ., data=data2)
summary(model_all)
vif(model_all)








