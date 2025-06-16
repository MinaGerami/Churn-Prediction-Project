install.packages("rpart")
install.packages("rpart.plot")

library(rpart)
library(rpart.plot)

setwd("C:/Users/Irana/Desktop/Marketing/Github/Churn-Prediction-Project/R/Decision Tree")
data <- read.csv("Data.csv", stringsAsFactors = TRUE)

tree_model <- rpart(churn ~ ., data = data, method = "class", parms = list(split = "information"))

png("decision_tree.png", width = 3000, height = 2000, res = 300)

rpart.plot(tree_model,
           type = 2,              
           extra = 104,           
           fallen.leaves = TRUE,  
           box.palette = "RdYlGn")

dev.off()



#print(tree_model)

png("decision_tree2.png", width = 3000, height = 2000, res = 300)
plot(tree_model, uniform = TRUE, margin = 0.1)
text(tree_model, use.n = TRUE, all = TRUE, cex = 0.8)

dev.off()


png("simple_tree4.png", width = 3000, height = 2000, res = 300)

plot(tree_model, uniform = TRUE, margin = 0.2)  # margin بیشتر برای فضای کناری

text(tree_model, use.n = TRUE, all = TRUE, cex = 0.6)  # کوچکتر کردن متن

dev.off()


#-------------------------------

library(rpart)
library(grid)
library(gridExtra)

data <- read.csv("Data.csv", stringsAsFactors = TRUE)

tree_model <- rpart(churn ~ ., data = data, method = "class", parms = list(split = "information"))

paths <- path.rpart(tree_model, nodes = rownames(tree_model$frame[tree_model$frame$var == "<leaf>", ]))

rules_text <- c()
for (i in seq_along(paths)) {
  rule <- paste(paths[[i]], collapse = " AND ")
  prediction <- as.character(tree_model$frame[rownames(tree_model$frame)[tree_model$frame$var == "<leaf>"], "yval"][[i]])
  rules_text[i] <- paste0("Rule ", i, ": IF ", rule, " → THEN churn = ", prediction)
}

rules_grob <- textGrob(paste(rules_text, collapse = "\n\n"), x = 0, y = 1, just = c("left", "top"), gp = gpar(fontsize = 14), default.units = "npc")

png("DecisionTreeRules.png", width = 3000, height = 4000, res = 300)
grid.newpage()
grid.draw(rules_grob)
dev.off()
