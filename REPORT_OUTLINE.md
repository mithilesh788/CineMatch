# CineMatch — Report outline

1. **Introduction:** problem statement, objectives, scope and motivation for a movie recommendation system.
2. **Dataset and EDA:** MovieLens 100K overview, cleaning checks, genre features, limitations and 80:20 chronological per-user holdout.
3. **System design:** Streamlit structure, TMDb poster/catalogue integration, recommendation workflow and module layout.
4. **Existing recommendation approaches:** popularity, content-based filtering (TF-IDF/cosine similarity), user-based collaborative filtering and SVD-based collaborative filtering.
5. **Integrated ML recommendation engine:** K-NN item-item rating similarity; K-Means user-preference clusters; entropy-based Decision Tree; Random Forest; linear SVM; Linear Regression rating prediction. Explain how the six model rankings are normalized and blended in Discover, Mood, and Describe It.
6. **Evaluation methodology:** train/test separation, relevance threshold (rating >= 4), ranking metrics at K=10, training-only fitting and limitations.
7. **Results and discussion:** insert actual generated `model_comparison.csv` and `rating_prediction_metrics.csv` values; compare Precision@10, Recall@10, NDCG@10, MAE and RMSE where applicable.
8. **Model settings and tuning:** document baseline settings and any tuning search actually performed; do not describe models as tuned unless experiments support it.
9. **UI testing and deployment:** screenshots for Discover, Mood, Describe It, Model Analytics, TMDb catalogue, Cinema Journal and hosted app if deployed. There is no separate Lab Models tab.
10. **Conclusion, limitations, future scope and references:** discuss cold-start behavior, genre-feature limitations, runtime, tuning and possible further algorithms.

Do not invent metric results or citations. Re-run `python src/evaluate_models.py` after changes and use only the generated values in the report. The source data is downloaded locally and should not be committed into a public repository. Check with the instructor whether separate practical demonstrations must be submitted in addition to the mini-project.
