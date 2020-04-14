# 第（2）题

# 设置工作目录并读取数据
setwd("F:/第2章/02-习题程序")
data <- read.transactions("./data/data.txt", format = "basket", sep = ",")
inspect(data)

# 设置支持度为0.2，置信度为0.5
library(arules)
Rules <- apriori(data, parameter = list(support = 0.2, confidence = 0.5))
inspect(Rules)
