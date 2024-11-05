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
# for (i in colnames(df)[3:13]){
#   p <- ggdensity(df[[i]], 
#             main = glue("Density plot of {i}"),
#             xlab = i)
#   p2 <- ggqqplot(df[[i]],
#                  main = glue("Comparison with normal distribution of {i}"))
#   print(p)
#   print(p2)
# }

## check the distribution of the logged variables
# for (i in colnames(df)[3:13]){
#   p <- ggdensity(log(df_log[[i]]), 
#                  main = glue("Density plot of log {i}"),
#                  xlab = i)
#   p2 <- ggqqplot(log(df_log[[i]]),
#                  main = glue("Comparison with normal distribution of log {i}"))
#   print(p)
#   print(p2)
# }

## simple regression model
reg_mod_lin <- lm(mln_euros ~ pop  + income + pct_cost + hcpi + gdp + dep_ratio + neighbour_dummy, data = df)
coeftest(reg_mod_lin, vcov. = vcovHC, type = "HC1")

plot(x = as.double(df$mln_euros), 
     y = as.double(reg_mod_lin$residuals), 
     xlab = "Real remittances data",
     ylab = "residuals",
     main = glue("Remittances and squared errors of the estimates, linear model"),
     pch = 20, 
     col = "steelblue")
abline(0,0)

## simple regression model  with logged variables
reg_mod_log <- lm(log(mln_euros) ~ log(pop)  + log(income) + pct_cost + log(hcpi_cap) + log(gdp) + dep_ratio + neighbour_dummy, data = df_log)
coeftest(reg_mod_log, vcov. = vcovHC, type = "HC1")

##test errors distribution
plot(x = as.double(df_log$mln_euros), 
     y = as.double(reg_mod_log$residuals), 
     xlab = "Real remittances data",
     ylab = "residuals",
     main = glue("Remittances and residuals of the estimates, log model"),
     pch = 20, 
     col = "steelblue")
abline(0,0)
summary(reg_mod_log)
ggcoef_model(reg_mod_log)

##try the model with natural disaster dummy
reg_mod_natdis <- lm(log(mln_euros) ~ log(pop)  + log(income) + pct_cost + 
                    log(hcpi_cap) + log(gdp) + dep_ratio 
                  + neighbour_dummy + nat_dist_dummy, data = df_log)
coeftest(reg_mod_natdis, vcov. = vcovHC, type = "HC1")

##try the model with the number of affected people
reg_mod_affect <- lm(log(mln_euros) ~ log(pop)  + log(income) + pct_cost + 
                       log(hcpi_cap) + log(gdp) + dep_ratio 
                     + neighbour_dummy + total.affected, data = df_log)
coeftest(reg_mod_affect, vcov. = vcovHC, type = "HC1")

##try the model with the both affected people and dummy
reg_mod_affect_dummy <- lm(log(mln_euros) ~ log(pop)  + log(income) + pct_cost + 
                       log(hcpi_cap) + log(gdp) + dep_ratio 
                     + neighbour_dummy + total.affected + nat_dist_dummy, 
                     data = df_log)
coeftest(reg_mod_affect_dummy, vcov. = vcovHC, type = "HC1")

##panel
simple_mod <- lm(log(mln_euros) ~ log(pop) + log(income)+ total.affected, 
                           data = df_log)
summary(simple_mod)
