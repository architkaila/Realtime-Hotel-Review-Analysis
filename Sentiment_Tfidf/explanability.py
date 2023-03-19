import shap

def explain_prediction(model, X_infer_vec, pred_class, vec, instance = 0):
    """
    Explain the prediction of the model using SHAP and force plot

    Args:
        model (sklearn model): trained model
        X_infer_vec (pd.DataFrame): featurized data
        pred_class (int): predicted class
        vec (sklearn vectorizer): vectorizer used to featurize the data
        instance (int): instance to explain

    Returns:
        plot (matplotlib plot): force plot
        shap_values (np.array): shap valuesclea
    """
    
    ## Initialize the explainer
    explainer = shap.TreeExplainer(model)

    ## Get the shap values
    shap_values = explainer.shap_values(X_infer_vec.toarray()[instance].reshape(1, -1), check_additivity=False)
    
    ## Initialize the force plot
    shap.initjs()

    ## Generate the force plot
    plot = shap.force_plot(explainer.expected_value[pred_class], shap_values[pred_class], feature_names=list(vec.get_feature_names_out()), matplotlib=True, show=False)
    
    return plot, shap_values