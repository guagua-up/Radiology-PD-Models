library(tidymodels)
library(naniar)
library(future)
library(pROC)
library(ggplot2)
library(patchwork)

# ==================== WB Model ====================
# Data preparation
Wholebrain <- read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/TD_PIGD/TD_PIGD_Wholebrain/MODELS/wb_FinalFeature.csv")
Wholebrain <- Wholebrain[, -c(1:4)] 
Wholebrain[,1] <- factor(Wholebrain[,1], levels = c(1, 0), labels = c("Class1", "Class0"))
names(Wholebrain)[1] <- "Diagnosis"
str(Wholebrain)
Wholebrain |> count(Diagnosis)
miss_var_summary(Wholebrain)
set.seed(401)
# parsnip
rf_spec <- rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) |>
  set_engine('ranger') |>
  set_mode('classification')

# recipe
recipe <- recipe(Diagnosis ~ ., data = Wholebrain) |>
  step_scale(all_numeric_predictors()) |>  
  step_center(all_numeric_predictors())

# workflow
rf <- workflow() |>
  add_recipe(recipe) |>
  add_model(rf_spec)

rf_param <- rf |>
  extract_parameter_set_dials() |>
  update(mtry = mtry_prop(c(0.1, 1)))

# Resamples
resamples <- vfold_cv(data = Wholebrain,
                      v = 10,
                      repeats = 10,
                      strata = Diagnosis)
metrics <- metric_set(accuracy, sens, spec, f_meas)

# Tune Parameters
plan(multisession, workers = parallel::detectCores() - 1) 
tune_rf_wb <- tune_grid(rf,
                     resamples = resamples,
                     param_info = rf_param,
                     grid = grid_space_filling(rf_param, size = 10),
                     metrics = metrics,
                     control = control_grid(verbose = TRUE,
                                            allow_par = TRUE,
                                            parallel_over = "everything",
                                            event_level = "first"))
# Optimal Parameters
best_params_wb <- select_best(tune_rf_wb, metric = "accuracy")

# Fit Final Model
final_wb <- finalize_workflow(rf, best_params_wb) %>%
  fit(Wholebrain)

# Cross-validation
final_rs_wb <- rf %>%
  finalize_workflow(best_params_wb) %>%
  fit_resamples(resamples, metrics = metrics)  

# ==================== NS Model ====================
# Data preparation
set.seed(401)
NS <- read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/TD_PIGD/TD_PIGD_SN/ns_FinalFeature.csv")
NS <- NS[, -c(1:4)]
NS[,1] <- factor(NS[,1], levels = c(1, 0), labels = c("Class1", "Class0"))
names(NS)[1] <- "Diagnosis"
str(NS)
NS |> count(Diagnosis)
miss_var_summary(NS)

# parsnip
rf_ns <- rand_forest(mtry = tune(),
                     trees = tune(),
                     min_n = tune()) |>
  set_engine('ranger') |>
  set_mode('classification')

# recipe
recipe_ns <- recipe(Diagnosis ~ ., data = NS) |>
  step_scale(all_numeric_predictors()) |>  
  step_center(all_numeric_predictors())

# workflow
rf_ns <- workflow() |>
  add_recipe(recipe_ns) |>
  add_model(rf_ns)

rf_param_ns <- rf_ns |>
  extract_parameter_set_dials() |>
  update(mtry = mtry_prop(c(0.1, 1)))

# Resamples
resamples <- vfold_cv(data = NS,
                      v = 10,
                      repeats = 10,
                      strata = Diagnosis)

# Tune Parameters
plan(multisession, workers = parallel::detectCores() - 1) 
tune_rf_ns <- tune_grid(rf_ns,
                        resamples = resamples,
                        param_info = rf_param_ns,
                        grid = grid_space_filling(rf_param_ns, size = 10),
                        metrics = metrics,
                        control = control_grid(verbose = TRUE,
                                               allow_par = TRUE,
                                               parallel_over = "everything",
                                               event_level = "first"))

# Optimal Parameters
best_params_ns <- select_best(tune_rf_ns, metric = "accuracy")

# Fit Final Model
final_ns <- finalize_workflow(rf_ns, best_params_ns) %>%
  fit(NS)

# Cross-validation
final_rs_ns <- rf_ns %>%
  finalize_workflow(best_params_ns) %>%
  fit_resamples(resamples, metrics = metrics)

# ==================== ONS Model ====================
# Data preparation
set.seed(401)
ONS <- read.csv("D:/PD/Imagings/SUVR/CentrumSemiovale_1/TD_PIGD/TD_PIGD_OTHERS/ons_FinalFeature.csv")
ONS <- ONS[, -c(1:4)]
ONS[,1] <- factor(ONS[,1], levels = c(1,0), labels = c("Class1", "Class0"))
names(ONS)[1] <- "Diagnosis"
str(ONS)
ONS |> count(Diagnosis)
miss_var_summary(ONS)

# parsnip
rf_ons <- rand_forest(mtry = tune(),
                      trees = tune(),
                      min_n = tune()) |>
  set_engine('ranger') |>
  set_mode('classification')

# recipe
recipe_ons <- recipe(Diagnosis ~ ., data = ONS) |>
  step_scale(all_numeric_predictors()) |>  
  step_center(all_numeric_predictors())

# workflow
rf_ons <- workflow() |>
  add_recipe(recipe_ons) |>
  add_model(rf_ons)

rf_param_ons <- rf_ons |>
  extract_parameter_set_dials() |>
  update(mtry = mtry_prop(c(0.1, 1)))

# Resamples
resamples <- vfold_cv(data = ONS,
                      v = 10,
                      repeats = 10,
                      strata = Diagnosis)

# Tune Parameters
tune_rf_ons <- tune_grid(rf_ons,
                         resamples = resamples,
                         param_info = rf_param_ons,
                         grid = grid_space_filling(rf_param_ons, size = 10),
                         metrics = metrics,
                         control = control_grid(verbose = TRUE,
                                                allow_par = TRUE,
                                                parallel_over = "everything",
                                                event_level = "first"))

# Optimal Parameters
best_params_ons <- select_best(tune_rf_ons, metric = "accuracy")

# Fit Final Model
final_ons <- finalize_workflow(rf_ons, best_params_ons) %>%
  fit(ONS)

# Cross-validation
final_rs_ons <- rf_ons %>%
  finalize_workflow(best_params_ons) %>%
  fit_resamples(resamples,metrics = metrics)

# Extract Final Results
cat("\n=== NS Model (Final CV Results) ===\n")
final_metrics_ns <- collect_metrics(final_rs_ns)
print(final_metrics_ns)
cat("\n=== ONS Model (Final CV Results) ===\n")
final_metrics_ons <- collect_metrics(final_rs_ons)
print(final_metrics_ons)
cat("=== Wholebrain Model (Final CV Results) ===\n")
final_metrics <- collect_metrics(final_rs_wb)
print(final_metrics)
