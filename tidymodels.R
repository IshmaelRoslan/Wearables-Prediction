library(tidyverse)

train <-
  read_csv(
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
    na = c("", "NA", "#DIV/0!")
  )

test <-
  read_csv(
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
    na = c("", "NA", "#DIV/0!")
  )

training <-
  train %>%
  mutate(
    datetime = lubridate::dmy_hms(train$cvtd_timestamp),
    across(where(is.logical), as.numeric),
    new_window = str_replace(new_window, "no", "FALSE"),
    new_window = str_replace(new_window, "yes", "TRUE"),
    new_window = as.logical(new_window)
  ) %>%
  select(-c(raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))

testing <-
  test %>%
  mutate(
    datetime = lubridate::dmy_hms(test$cvtd_timestamp),
    across(where(is.logical), as.numeric),
    new_window = str_replace(new_window, "no", "FALSE"),
    new_window = str_replace(new_window, "yes", "TRUE"),
    new_window = as.logical(new_window)
  ) %>%
  select(-c(raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))

library(tidymodels)
tidymodels_prefer()

wear_folds <-
  vfold_cv(training, v = 5, strata = classe)

wear_rec <-
  recipe(classe ~ ., data = training) %>%
  update_role("...1", new_role = "id") %>%
  step_filter_missing(all_predictors(), threshold = 0.1) %>%
  step_nzv(all_numeric()) %>%
  step_corr(all_numeric()) %>%
  step_rm(c("...1", "user_name", "new_window", "num_window", "datetime")) %>%
  step_normalize(all_predictors()) %>%
  prep()

rf_spec <-
  rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

tree_spec <-
  boost_tree() %>%
  set_engine("xgboost") %>%
  set_mode("classification")

knn_spec <-
  nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

wear_models <-
  workflow_set(
    preproc = list(wear_rec),
    models = list(rf =  rf_spec,
                  xgb = tree_spec,
                  knn = knn_spec),
    cross = TRUE
  )

doParallel::registerDoParallel()
set.seed(1690)
wear_rs <-
  wear_models %>%
  workflow_map("fit_resamples",
               resamples = wear_folds,
               metrics = metric_set(accuracy, kap))

wear_rs %>%
  unnest(result) %>%
  unnest(.metrics) %>%
  group_by(wflow_id, .metric) %>%
  summarise(estimate = mean(.estimate))

#Choose Random Forests
tune_spec <-
  rand_forest(mtry = tune(),
              trees = 1000,
              min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

tune_wf <- workflow() %>%
  add_recipe(wear_rec) %>%
  add_model(tune_spec)

doParallel::registerDoParallel()
set.seed(1690)

tune_res <-
  tune_grid(tune_wf,
            resamples = wear_folds,
            grid  = 10)

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend =  F) +
  facet_wrap( ~ parameter, scales = "free")

rf_grid <-
  grid_regular(mtry(range = c(6, 15)),
               min_n(range = c(2, 5)),
               levels = 40)

set.seed(1690)

regular_res <-
  tune_grid(tune_wf,
            resamples = wear_folds,
            grid  = rf_grid)

regular_res %>%
  collect_metrics() %>%
  select(mtry:.metric,mean) %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  facet_wrap(~.metric, scales = "free")

regular_res %>%
  collect_metrics() %>%
  select(mtry:.metric,mean) %>%
  ggplot(aes(mtry, min_n, fill = mean)) +
  geom_tile() +
  facet_wrap(~.metric, scales = "free")

best_acc <- select_best(regular_res, "roc_auc")
best_acc

final_rf <- finalize_model(tune_spec,
                           best_acc)

library(vip)

final_fit <-
  final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(classe ~ .,
      data = juice(wear_rec)) %>%
  vip(geom = "point")

  