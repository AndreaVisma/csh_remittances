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
df$year <- factor(df$year) 
df$country <- factor(df$country) 
df<- pdata.frame(df, index = c("country", "year"))
dim(df)
df_log <- subset(df, mln_euros > 0 & pop > 0 & hcpi > 0)

## simple regression model
reg_mod_lin <- lm(mln_euros ~ pop  + pct_cost + gdp + dep_ratio + neighbour_dummy
                  + factor(group) + factor(year), data = df)
coeftest(reg_mod_lin, vcov. = vcovHC, type = "HC1")

plot(x = as.double(df$mln_euros), 
     y = as.double(reg_mod_lin$residuals), 
     xlab = "Real remittances data",
     ylab = "residuals",
     main = glue("Remittances and squared errors of the estimates, linear model"),
     pch = 20, 
     col = "steelblue")
abline(0,0)
ggcoef_model(reg_mod_lin)
summary(reg_mod_lin)
##see fitted v. real observations
plot(x = as.double(df$mln_euros), 
     y = as.double(reg_mod_lin$fitted.values), 
     xlab = "Real remittances data",
     ylab = "Fitted remittance data",
     main = glue("observed and fitted remittances"),
     pch = 20, 
     col = "steelblue",
     panel.first=grid())
abline(0,1)

## simple regression model  with logged variables
reg_mod_log <- lm(log(mln_euros) ~ log(pop)  + factor(group) + pct_cost + log(hcpi_cap) + log(gdp) + dep_ratio + neighbour_dummy, data = df_log)
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

##try to add year values
reg_mod_log_year <- lm(log(mln_euros) ~ log(pop) + pct_cost  + log(gdp) + dep_ratio + factor(group)
                       + year, data = df_log)
coeftest(reg_mod_log_year, vcov. = vcovHC, type = "HC1")

##test errors distribution
plot(x = as.double(df_log$mln_euros), 
     y = as.double(reg_mod_log_year$residuals), 
     xlab = "Real remittances data",
     ylab = "residuals",
     main = glue("Remittances and residuals of the estimates, log model"),
     pch = 20, 
     col = "steelblue")
abline(0,0)
summary(reg_mod_log_year)
ggcoef_model(reg_mod_log_year)

##see fitted v. real observations
plot(x = as.double(log(df_log$mln_euros)), 
     y = as.double(reg_mod_log_year$fitted.values), 
     xlab = "Real remittances data",
     ylab = "Fitted remittance data",
     main = glue("observed and fitted remittances"),
     pch = 20, 
     col = "steelblue",
     panel.first=grid())
abline(0,1)
