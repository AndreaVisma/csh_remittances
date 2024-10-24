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
for (i in colnames(df)[3:8]){
  p <- ggdensity(df[[i]], 
            main = glue("Density plot of {i}"),
            xlab = i)
  p2 <- ggqqplot(df[[i]],
                 main = glue("Comparison with normal distribution of {i}"))
  print(p)
  print(p2)
}

## log the variables
for (i in colnames(df)[3:8]){
  p <- ggdensity(log(df_log[[i]]), 
                 main = glue("Density plot of log {i}"),
                 xlab = i)
  p2 <- ggqqplot(log(df_log[[i]]),
                 main = glue("Comparison with normal distribution of log {i}"))
  print(p)
  print(p2)
}


## simple regression model
reg_mod <- lm(mln_euros ~ pop  + income + pct_cost + hcpi_cap + gdp + dep_ratio + neighbour_dummy, data = df)
coeftest(reg_mod, vcov. = vcovHC, type = "HC1")

## estimated values from regression
df$est_rem <- reg_mod$coefficients[1] + as.double(df$pop * reg_mod$coefficients['pop'] + df$income * reg_mod$coefficients['income'] + 
                   df$pct_cost * reg_mod$coefficients['pct_cost'] + df$hcpi_cap * reg_mod$coefficients['hcpi_cap'] + 
                   df$gdp * reg_mod$coefficients["gdp"] + df$dep_ratio * reg_mod$coefficients["dep_ratio"] + 
                   df$neighbour_dummy * reg_mod$coefficients["neighbour_dummy"])
df$err <- df$mln_euros - df$est_rem

df$sq_err <- lapply(df$err, fun)

plot(x = as.double(df$mln_euros), 
     y = as.double(df$sq_err), 
     xlab = "Real remittances data",
     ylab = "residuals",
     main = glue("Remittances and squared errors of the estimates"),
     pch = 20, 
     col = "steelblue")
abline(0,0)

## simple regression model  with logged variables
reg_mod <- lm(log(mln_euros) ~ log(pop)  + log(income) + pct_cost + log(hcpi_cap) + log(gdp) + dep_ratio + neighbour_dummy, data = df_log)
coeftest(reg_mod, vcov. = vcovHC, type = "HC1")

##test errors distribution
plot(x = as.double(df_log$mln_euros), 
     y = as.double(reg_mod$residuals), 
     xlab = "Real remittances data",
     ylab = "residuals",
     main = glue("Remittances and residuals of the estimates"),
     pch = 20, 
     col = "steelblue")
abline(0,0)
summary(reg_mod)
ggcoef_model(reg_mod)

##try the model with country fixed effects
fixef_mod <- plm(log(mln_euros) ~ log(pop)  + log(income) + pct_cost + log(hcpi_cap) + log(gdp) + dep_ratio + neighbour_dummy, 
                 index = c("country", "year"),
                 data = df_log)
coeftest(fixef_mod, vcov. = vcovHC, type = "HC1")
