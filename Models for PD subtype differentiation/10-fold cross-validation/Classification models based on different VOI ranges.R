# 清除环境变量
rm(list = ls())

# 安装并加载所需包
if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret, pROC, randomForest, ggplot2, ggpubr, dplyr, tidyr, purrr, doParallel, kernlab)

# 启用并行计算并确保所有节点加载必要包
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)
clusterEvalQ(cl, {
  library(pROC)
  library(caret)
  library(randomForest)
})

# 读取三组数据
data_list <- list(
  Wholebrain = read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/TD_PIGD/TD_PIGD_Wholebrain/MODELS/FinalFeature.csv") %>% 
    select(-(1:4)) %>% 
    mutate(Diagnosis = factor(.[,1], labels = c("Class0", "Class1"))) %>% 
    rename(Diagnosis = 1),
  
  ONS = read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/TD_PIGD/TD_PIGD_OTHERS/FinalFeature_ons.csv") %>% 
    select(-(1:4)) %>% 
    mutate(Diagnosis = factor(.[,1], labels = c("Class0", "Class1"))) %>% 
    rename(Diagnosis = 1),
  
  NS = read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/TD_PIGD/TD_PIGD_SN/FinalFeature.csv") %>% 
    select(-(1:4)) %>% 
    mutate(Diagnosis = factor(.[,1], labels = c("Class0", "Class1"))) %>% 
    rename(Diagnosis = 1)
)

# 设置随机种子
set.seed(863)

# 定义10×5交叉验证控制参数
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE
)

# 训练随机森林模型
train_rf_model <- function(data) {
  train(Diagnosis ~ ., data = data,
        method = "rf",
        trControl = ctrl,
        tuneGrid = expand.grid(mtry = seq(2, sqrt(ncol(data)-1), length.out = 3)),
        importance = TRUE,
        metric = "ROC")
}

# 训练所有模型
models <- map(data_list, train_rf_model)

# 性能指标计算函数
calculate_metrics <- function(model) {
  preds <- model$pred %>% 
    group_by(Resample) %>% 
    summarise(
      Accuracy = mean(obs == pred),
      Sensitivity = sum(obs == "Class1" & pred == "Class1") / sum(obs == "Class1"),
      Specificity = sum(obs == "Class0" & pred == "Class0") / sum(obs == "Class0"),
      AUC = as.numeric(auc(roc(obs, Class1, levels = c("Class0", "Class1"), quiet = TRUE))),
      .groups = "drop"
    )
  
  list(
    Mean = preds %>% 
      select(-Resample) %>% 
      summarise_all(mean, na.rm = TRUE),
    SD = preds %>% 
      select(-Resample) %>% 
      summarise_all(sd, na.rm = TRUE)
  )
}

# 定义置换检验函数
permutation_test <- function(model, data, n_perm = 1000) {
  obs_prob <- predict(model, data, type = "prob")[, "Class1"]
  obs_auc <- roc(data$Diagnosis, obs_prob, levels = c("Class0", "Class1"), quiet = TRUE)$auc
  perm_aucs <- foreach(i = 1:n_perm, .combine = 'c', 
                       .packages = c("pROC", "caret", "randomForest")) %dopar% {
                         perm_data <- data
                         perm_data$Diagnosis <- sample(perm_data$Diagnosis)
                         perm_prob <- predict(model, perm_data, type = "prob")[, "Class1"]
                         roc(perm_data$Diagnosis, perm_prob, levels = c("Class0", "Class1"), quiet = TRUE)$auc
                       }
  p_value <- (sum(perm_aucs >= obs_auc) + 1) / (n_perm + 1)
  return(p_value)
}

# 计算各模型性能指标
model_metrics <- map(models, calculate_metrics)

# 执行置换检验
perm_results <- list(
  Wholebrain = tryCatch(permutation_test(models$Wholebrain, data_list$Wholebrain, 1000),
                        error = function(e) {message("Wholebrain permutation failed: ", e$message); NA}),
  ONS = tryCatch(permutation_test(models$ONS, data_list$ONS, 1000),
                 error = function(e) {message("ONS permutation failed: ", e$message); NA}),
  NS = tryCatch(permutation_test(models$NS, data_list$NS, 1000),
                error = function(e) {message("NS permutation failed: ", e$message); NA})
)

# 创建结果汇总表
results_table <- map2_dfr(names(model_metrics), perm_results, function(model_name, perm_result) {
  metrics <- model_metrics[[model_name]]
  data.frame(
    Model = model_name,
    Accuracy = sprintf("%.3f ± %.3f", metrics$Mean$Accuracy, metrics$SD$Accuracy),
    Sensitivity = sprintf("%.3f ± %.3f", metrics$Mean$Sensitivity, metrics$SD$Sensitivity),
    Specificity = sprintf("%.3f ± %.3f", metrics$Mean$Specificity, metrics$SD$Specificity),
    AUC = sprintf("%.3f ± %.3f", metrics$Mean$AUC, metrics$SD$AUC),
    Permutation_p = ifelse(is.na(perm_result), "Failed", sprintf("%.4f", perm_result)),
    stringsAsFactors = FALSE
  )
})

# 打印美观的结果
cat("\n=== 随机森林模型性能评估结果 (10×5 CV) ===\n")
print(knitr::kable(results_table, align = 'c'))

# 停止并行计算
stopCluster(cl)
