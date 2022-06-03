library(caret)
library(e1071)
set.seed(12345)

#Create simulation data
.topxdata <- matrix(rnorm(20000, mean=0, sd=1), nrow = 2000, ncol = 10)
.botxdata <- matrix(rnorm(200, mean=1, sd=1), nrow = 20, ncol = 10)
xdata <- rbind(.topxdata, .botxdata)
colnames(xdata) <- 1:10

ydata <- c(rep("Top", 2000), rep("Bottom", 20) )
ydata <- as.factor(ydata)

# Shuffle
.idx <- sample(1:2020, 2020)
xdata <- xdata[.idx,]
ydata <- ydata[.idx]

.w <- ifelse(ydata == "Top", 
            yes = (1/(table(ydata)/length(ydata)))["Top"], 
            no = (1/(table(ydata)/length(ydata)))["Bottom"])

# Setup for cross validation
ctrl <- trainControl(method="cv", number = 5,   # 10fold cross validation
                     summaryFunction=twoClassSummary,   # Use AUC to pick the best model
                     classProbs=TRUE,
                     verboseIter = TRUE
                     )


#Train and Tune the SVM
svm.tune <- train(x = xdata,
                  y = ydata,
                  method = "svmRadial",   # Radial kernel
                  tuneGrid = expand.grid(
                    C = seq(0.2, 2, 0.2),
                    sigma = 10^seq(-2, 2)
                    ),
                  metric="ROC",
                  weights = .w,
                  trControl=ctrl)

phat <- predict(svm.tune, xdata, type = "prob")[,2]
pROC::auc(pROC::roc(ydata, phat, levels = c("Bottom", "Top"), direction = "<"))


.test_topxdata <- matrix(rnorm(200, mean=0, sd=1), nrow = 20, ncol = 10)
.test_botxdata <- matrix(rnorm(200, mean=1, sd=1), nrow = 20, ncol = 10)
test_xdata <- rbind(.test_topxdata, .test_botxdata)
test_ydata <- c(rep("Top", 20), rep("Bottom", 20) )
test_ydata <- as.factor(test_ydata)
test_phat <- predict(svm.tune, test_xdata, type = "prob")[,2]



pROC::auc(pROC::roc(test_ydata, test_phat, levels = c("Bottom", "Top"), direction = "<"))

