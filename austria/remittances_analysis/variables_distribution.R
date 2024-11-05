gc()
rm(list=ls())

library(readxl)
library(plm)
library(lmtest)
library(glue)
library(plotly)
library(ggplot2)
library(ggpubr)
library(ggstats)
library(jtools)

## define power 2 function
fun <- function(x) {
  x ^ 2
}

df <- read_xlsx("c:\\data\\my_datasets\\remittances_austria_panel.xlsx")
df<- pdata.frame(df, index = c("country", "year"))
dim(df)
df_log <- subset(df, mln_euros > 0 & pop > 0 & hcpi > 0)

#check distribution of the variables
for (i in colnames(df)[3:13]){
  p <- ggdensity(df[[i]], 
            main = glue("Density plot of {i}"),
            xlab = i)
   p2 <- ggqqplot(df[[i]],
                  main = glue("Comparison with normal distribution of {i}"))
  print(p)
  print(p2)
 }

## check the distribution of the logged variables
 for (i in colnames(df)[3:13]){
   p <- ggdensity(log(df_log[[i]]), 
                  main = glue("Density plot of log {i}"),
                  xlab = i)
   p2 <- ggqqplot(log(df_log[[i]]),
                  main = glue("Comparison with normal distribution of log {i}"))
   print(p)
   print(p2)
 }