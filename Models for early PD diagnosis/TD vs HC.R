# 清除环境变量
rm(list = ls())

# 安装并加载所需包
if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret, pROC, randomForest, e1071, ggplot2, dplyr, purrr, doParallel, knitr, kernlab, tidyverse)

# 启用并行计算
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)
clusterEvalQ(cl, {
  library(pROC)
  library(caret)
  library(randomForest)
  library(e1071)
  library(purrr) 
})

# 读取数据
data <- read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/HC_TD/HC_TD_Wholebrain/MODELS/FinalFeature.csv") %>% 
  select(-(1:5)) %>% 
  mutate(Diagnosis = factor(.[,1], levels = c(0, 1), labels = c("HC", "TD"))) %>% 
  rename(Diagnosis = 1)

# 设置随机种子
set.seed(863)

# 定义交叉验证参数
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE
)

# 训练三种分类器
train_models <- function(data, ctrl) {
  list(
    logistic = train(Diagnosis ~ ., data = data, 
                     method = "glm", family = "binomial",
                     trControl = ctrl, metric = "ROC"),
    
    svm = train(Diagnosis ~ ., data = data,
                method = "svmRadial",
                trControl = ctrl, tuneLength = 10,
                metric = "ROC"),
    
    rf = train(Diagnosis ~ ., data = data,
               method = "rf",
               trControl = ctrl, tuneLength = 10,
               importance = TRUE, metric = "ROC")
  )
}

# 训练模型
models <- train_models(data, ctrl)

# 性能指标计算
calculate_metrics <- function(model) {
  if (is.null(model$pred)) return(NULL)
  
  fold_stats <- model$pred %>%
    group_by(Resample) %>%
    summarise(
      Accuracy = mean(obs == pred, na.rm = TRUE),
      Sensitivity = sum(obs == "TD" & pred == "TD", na.rm = TRUE) / 
        sum(obs == "TD", na.rm = TRUE),
      Specificity = sum(obs == "HC" & pred == "HC", na.rm = TRUE) / 
        sum(obs == "HC", na.rm = TRUE),
      AUC = as.numeric(pROC::auc(pROC::roc(obs, TD, quiet = TRUE))),
      .groups = "drop"
    )
  
  # 计算均值和标准差
  data.frame(
    Accuracy_mean = round(mean(fold_stats$Accuracy, na.rm = TRUE), 3),
    Accuracy_sd = round(sd(fold_stats$Accuracy, na.rm = TRUE), 3),
    Sensitivity_mean = round(mean(fold_stats$Sensitivity, na.rm = TRUE), 3),
    Sensitivity_sd = round(sd(fold_stats$Sensitivity, na.rm = TRUE), 3),
    Specificity_mean = round(mean(fold_stats$Specificity, na.rm = TRUE), 3),
    Specificity_sd = round(sd(fold_stats$Specificity, na.rm = TRUE), 3),
    AUC_mean = round(mean(fold_stats$AUC, na.rm = TRUE), 3),
    AUC_sd = round(sd(fold_stats$AUC, na.rm = TRUE), 3)
  )
}

metrics_list <- purrr::map(models, safely(calculate_metrics))

# 提取有效结果
metrics_list <- map(metrics_list, ~ .x$result) %>% 
  compact()

# 置换检验函数
permutation_test <- function(model, data, n_perm = 1000) {
  obs_prob <- predict(model, data, type = "prob")[, "TD"]
  obs_auc <- pROC::roc(data$Diagnosis, obs_prob)$auc
  
  perm_aucs <- foreach(i = 1:n_perm, .combine = 'c',
                       .packages = c("pROC", "caret")) %dopar% {
                         perm_data <- data
                         perm_data$Diagnosis <- sample(perm_data$Diagnosis)
                         perm_prob <- predict(model, perm_data, type = "prob")[, "TD"]
                         pROC::roc(perm_data$Diagnosis, perm_prob)$auc
                       }
  
  (sum(perm_aucs >= obs_auc) + 1) / (n_perm + 1)
}

# 执行置换检验
perm_results <- map(models, ~{
  tryCatch(
    permutation_test(.x, data, 1000),
    error = function(e) {
      message("置换检验失败: ", conditionMessage(e))
      NA
    }
  )
})

# 创建结果表
results_table <- map2_dfr(names(metrics_list), metrics_list, function(name, metrics) {
  data.frame(
    Model = case_when(
      name == "logistic" ~ "Logistic Regression",
      name == "svm" ~ "SVM (RBF Kernel)", 
      name == "rf" ~ "Random Forest"
    ),
    Accuracy = sprintf("%.3f ± %.3f", metrics$Accuracy_mean, metrics$Accuracy_sd),
    Sensitivity = sprintf("%.3f ± %.3f", metrics$Sensitivity_mean, metrics$Sensitivity_sd),
    Specificity = sprintf("%.3f ± %.3f", metrics$Specificity_mean, metrics$Specificity_sd),
    AUC = sprintf("%.3f ± %.3f", metrics$AUC_mean, metrics$AUC_sd),
    `Permutation p` = ifelse(is.na(perm_results[[name]]), "Failed",
                             ifelse(perm_results[[name]] < 0.001, "<0.001",
                                    sprintf("%.3f", perm_results[[name]]))),
    stringsAsFactors = FALSE
  )
})

# 打印结果
cat("\n=== 三分类器性能比较 (10×5 CV) ===\n")
print(knitr::kable(results_table, align = c('l', rep('c', 5))))

# 停止并行
stopCluster(cl)
