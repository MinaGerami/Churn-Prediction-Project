install.packages("rpart")
install.packages("rpart.plot")

library(rpart)
library(rpart.plot)

setwd("C:/Users/Esmaeil/Downloads/Churn-Prediction-Project-main (1)/Churn-Prediction-Project-main/R/Decision Tree")
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


data <- read.csv("Data.csv", stringsAsFactors = TRUE)

# آموزش درخت تصمیم
tree_model <- rpart(churn ~ ., data = data, method = "class", parms = list(split = "information"))

# استخراج مسیرهای برگ‌ها
leaf_nodes <- rownames(tree_model$frame[tree_model$frame$var == "<leaf>", ])
paths <- path.rpart(tree_model, nodes = leaf_nodes)

# تولید متن قوانین
rules_text <- c()
for (i in seq_along(paths)) {
  rule <- paste(paths[[i]], collapse = " AND ")
  prediction <- as.character(tree_model$frame[leaf_nodes[i], "yval"])
  rules_text[i] <- paste0("Rule ", i, ": IF ", rule, " → THEN churn = ", prediction)
}

# ذخیره در فایل TXT
writeLines(rules_text, con = "DecisionTreeRules.txt")

print("✅ قوانین درخت در فایل 'DecisionTreeRules.txt' ذخیره شدند.")

