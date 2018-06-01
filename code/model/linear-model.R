library(forecast)
library(glmnet)
library(tidyverse)
setwd("/Users/vsokolov/Dropbox/prj/dl-graph/code/model")
d = read.csv("3.csv")
head(d)


mu = mean(d$log_ret_y)
yts = ts(data = d$log_ret_y,start = 1,frequency = 1)
tsm = auto.arima(y = yts, xreg =  d[,1:5] %>% as.matrix())
summary(tsm)



plot(d$log_ret_y, type='l')
abline(h=mu, col=2, lwd=2)
lines(tsm$fitted %>% as.numeric(), col=3)
lines(glmpred %>% as.numeric(), col=4)
lines(lmod$fitted.values %>% as.numeric(), col=4)
