library(forecast)
library(glmnet)
setwd("/Users/vsokolov/Dropbox/proj/dl-graph/code/model")
d = read.csv("3.csv")
head(d)

mu = mean(d$log_ret_y)

sd(d$log_ret_y)
plot(exp(d$log_ret_y), type='l')

yts = ts(data = d$log_ret_y,start = 1,frequency = 1)
m = auto.arima(y = yts)
sqrt(mean((m$fitted - yts)^2))
sqrt(mean((mu - d$log_ret_y)^2))


lmod = lm(log_ret_y~., data=d)
summary(lmod)
sqrt(mean((lmod$fitted - yts)^2))

plot(lmod)

cvfit = cv.glmnet(x = d[,1:5] %>% as.matrix(), y = d[,6])
plot(cvfit)

pred = predict(cvfit, newx = d[,1:5] %>% as.matrix(), s = "lambda.min")

sqrt(mean((m$fitted - yts)^2))
sqrt(mean((pred - d$log_ret_y)^2))
sqrt(mean((mu - d$log_ret_y)^2))
