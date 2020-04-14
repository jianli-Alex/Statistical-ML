# 设置工作目录并读取数据
setwd("F:/第2章/01-任务程序")
GoodsOrder <- read.csv("./data/GoodsOrder.csv", stringsAsFactors = FALSE)
library(arules)  # 导入所需库包

# 数据形式转换
dataList <- list()
for (i in unique(GoodsOrder$ID)) {
    dataList[[i]] <- GoodsOrder[which(GoodsOrder$ID == i), 2]
}
TransRep <- as(dataList, "transactions")

RulesRep <- apriori(TransRep, parameter = list(support = 0.02, confidence = 0.25))

inspect(sort(RulesRep, by = "lift")[1:25])  # 按提升度从高到低查看前25条规则
