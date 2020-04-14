# 设置工作目录并读取数据
setwd("F:/第2章/01-任务程序")
GoodsOrder <- read.csv("./data/GoodsOrder.csv", stringsAsFactors = FALSE)
# 统计热销商品
hotGoods <- data.frame(table(GoodsOrder[, 2]))
names(hotGoods) <- c("Goods", "Num")
hotGoods["Percent"] <- hotGoods$Num / sum(hotGoods$Num)
hotGoods <- hotGoods[order(hotGoods$Percent, decreasing = TRUE),]
write.csv(hotGoods, "./tmp/hotGoods.csv", row.names = FALSE)



# 售出商品类型结构分析
GoodTypes <- read.csv("./data/GoodsTypes.csv", stringsAsFactors = FALSE)
Goods <- merge(GoodsOrder, GoodTypes, 'Goods', all.x = TRUE, all.y = TRUE)
hotTypes <- data.frame(table(Goods$Types))
names(hotTypes) <- c("Types", "Num")
hotTypes["Percent"] <- hotTypes[, 2] / sum(hotTypes[, 2])
hotTypes <- hotTypes[order(hotTypes$Percent, decreasing = TRUE),]
write.csv(hotTypes, "./tmp/hotTypes.csv", row.names = FALSE)



# 售出商品类型内部结构分析
Drink <- Goods[which(Goods[,3] == "非酒精饮料"),]
hotDrink <- data.frame(table(Drink$Goods))
names(hotDrink) <- c("Goods", "Num")
hotDrink["Percent"] <- hotDrink$Num / sum(hotDrink$Num)
hotDrink <- hotDrink[order(hotDrink$Percent, decreasing = TRUE),]
write.csv(hotDrink, './tmp/hotDrink.csv', row.names = FALSE)
