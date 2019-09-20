setwd("~/PycharmProjects/fortune_500")
library(dplyr)
library(ggplot2)
library(ggpubr)

company_df = read.csv('indeed_eda/data/company_weights_stacked.csv', stringsAsFactors = FALSE)
company_df$Topic = replace(company_df$Topic, 
                           c("Topic_1","Topic_2","Topic_3","Topic_4","Topic_5","Topic_6","Topic_7","Topic_8","Topic_9","Topic_10",
                             "Topic_11","Topic_12","Topic_13","Topic_14","Topic_15","Topic_16","Topic_17","Topic_18","Topic_19",
                             "Topic_20"), 
                           c("Engagement", "Mission/Purpose", "Daily_Work", "Acquisitions/Merger", "Organization/Structure",
                             "Manager", "Technology","Hiring/Recruitment","Physical Demand","Career_Development","Location",
                             "Overall_Company", "Co-workers", "Benefits/Pay", "Culture", "Hostile_Environment",
                             "Diversity/Inclusion", "CEO/Executives", "Work-Life_Balance", "Customer_Service"))
ggballoonplot(company_df,x="Topic",y="Company", size = "Topic_Weight",
              fill = 'Mean_Rating',size.range=c(0.1,8)) + gradient_fill(c("red","white","blue"))