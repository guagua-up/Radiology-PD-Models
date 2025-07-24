library(tidymodels)
library(naniar)
library(future)
library(pROC)
library(ggplot2)
library(patchwork)
library(kernlab)

### Data preparation
Wholebrain <- read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/TD_PIGD/TD_PIGD_Wholebrain/MODELS/wb_FinalFeature.csv")
Wholebrain <- Wholebrain[, -c(1:4)] 
Wholebrain[,1] <- factor(Wholebrain[,1], levels = c(1, 0), labels = c("Class1", "Class0"))
names(Wholebrain)[1] <- "Diagnosis"
str(Wholebrain)
Wholebrain |> count(Diagnosis)
miss_var_summary(Wholebrain)

set.seed(401)
resamples <- vfold_cv(data = Wholebrain,
                      v = 10,
                      repeats = 10,
                      strata = Diagnosis)
metrics <- metric_set(accuracy, sens, spec, f_meas)

# ==================== Logistic Regression(LR) ====================
# parsnip
lr_spec <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

# Recipe
recipe_lr <- recipe(Diagnosis ~ ., data = Wholebrain) |>
  step_scale(all_numeric_predictors()) |>  
  step_center(all_numeric_predictors())

# workflow
lr_wf <- workflow() |>
  add_recipe(recipe_lr) |>
  add_model(lr_spec)

# cross—validation
lr_final <- lr_wf|>
  fit_resamples(resamples,metrics=metrics)

# ==================== Support Vector Machine (SVM) ====================
# parsnip
svm_spec <- svm_poly(degree = 1) |>
  set_mode("classification") |>
  set_engine("kernlab", scaled = FALSE) |>
  set_args(cost = tune())

# Recipe
recipe_svm <- recipe(Diagnosis ~ ., data = Wholebrain) |>
  step_scale(all_numeric_predictors()) |>  
  step_center(all_numeric_predictors())

# workflow
svm_wf <- workflow() |>
  add_recipe(recipe_svm) |>
  add_model(svm_spec)

# Tune Parameters
param_grid <- grid_regular(cost(), levels = 10)

plan(multisession, workers = parallel::detectCores() - 1) 
tune_svm <- tune_grid(
  svm_wf,
  resamples = resamples,
  grid = param_grid,
  metrics = metrics,
  control = control_grid(verbose = TRUE,
                         allow_par = TRUE,
                         parallel_over = "everything",
                         event_level = "first")
)

# Optimal Parameters
best_cost <- tune_svm |> select_best(metric = 'accuracy')

# Fit Final Model
svm_fit <- finalize_workflow(svm_wf, best_cost)|>
  fit(Wholebrain)

# cross—validation
svm_final <- svm_wf %>%
  finalize_workflow(best_cost) %>%
  fit_resamples(resamples, metrics = metrics)

# ==================== Random Forest (RF) ====================
# parsnip
rf_spec <- rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) |>
  set_engine('ranger') |>
  set_mode('classification')

# recipe
recipe_rf <- recipe(Diagnosis ~ ., data = Wholebrain) |>
  step_scale(all_numeric_predictors()) |>  
  step_center(all_numeric_predictors())

# workflow
rf_wf <- workflow() |>
  add_recipe(recipe_rf) |>
  add_model(rf_spec)

rf_param <- rf_wf |>
  extract_parameter_set_dials() |>
  update(mtry = mtry_prop(c(0.1, 1)))

# Tune Parameters
tune_rf <- tune_grid(rf_wf,
                     resamples = resamples,
                     param_info = rf_param,
                     grid = grid_space_filling(rf_param, size = 10),
                     metrics = metrics,
                     control = control_grid(verbose = TRUE,
                                            allow_par = TRUE,
                                            parallel_over = "everything",
                                            event_level = "first"))

# Optimal Parameters
optim_rf_param <- tune_rf |> select_best(metric = "accuracy")

# Fit Final Model
rf_fit <-finalize_workflow(rf_wf, optim_rf_param) %>%
  fit(Wholebrain)

# Cross-Validation
rf_final <- rf_wf %>%
  finalize_workflow(optim_rf_param) %>%
  fit_resamples(resamples, metrics = metrics) 

# Extract Final Results
cat("\n=== LR Model (Final CV Results) ===\n")
final_metrics_lr <- collect_metrics(lr_final)
print(final_metrics_lr)
cat("\n=== SVM Model (Final CV Results) ===\n")
final_metrics_svm <- collect_metrics(svm_final)
print(final_metrics_svm)
cat("=== RF Model (Final CV Results) ===\n")
final_metrics_rf <- collect_metrics(rf_final)
print(final_metrics_rf)
