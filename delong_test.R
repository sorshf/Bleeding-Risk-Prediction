require(pROC)

setwd("./Time Series Prediction of Bleeding/keras_tuner_results/")

all_models <- c("Baseline_Dense", "FUP_RNN", "LastFUP_Dense", "Ensemble", 'CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI')

p_value_matrix <- matrix(nrow = length(all_models), ncol = length(all_models), dimnames = list(all_models, all_models))

for (model_1_name in all_models) {
  for (model_2_name in all_models) {
    model_1_results <- read.csv(paste("./", model_1_name, "/", model_1_name, "_detailed_test_results.csv", sep = ""))
    model_2_results <- read.csv(paste("./", model_2_name, "/", model_2_name, "_detailed_test_results.csv", sep = ""))
    
    roc_model_1 <- roc(response=model_1_results$y_actual, predictor=model_1_results$y_pred)
    roc_model_2 <- roc(response=model_2_results$y_actual, predictor=model_2_results$y_pred)
    
    test_res <- roc.test(roc_model_1, roc_model_2, method="delong")
    
    p_value_matrix[model_1_name, model_2_name] <- test_res$p.value
    
  }
}

write.csv(p_value_matrix, file="delong_test_not_corrected.csv")




