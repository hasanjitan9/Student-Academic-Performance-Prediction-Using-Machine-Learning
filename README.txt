Student Academic Performance Prediction Using Machine Learning

Submission package contents:

1. Student_Performance_Final_Report.pdf
   - PDF version of the final report for easy viewing.

2. Student_Performance_ML_Project.ipynb
   - Jupyter Notebook containing the full Python implementation.
   - Run it in Jupyter Notebook / Google Colab / VS Code.
   - Keep student_data.csv in the same folder as the notebook.

3. student_data.csv
   - Dataset used in the project.

4. outputs/
   - Contains generated figures, result tables, feature importance file, and confusion matrices.

Main model results:
- Logistic Regression and SVM achieved the best performance.
- Accuracy = 0.8734
- F1-score = 0.9000

Important explanation for discussion:
- G3 was converted into Pass/Fail.
- Pass = 1 if G3 >= 10, Fail = 0 if G3 < 10.
- G1 and G2 are previous grades and are the strongest predictors of final performance.
- A second experiment without G1 and G2 was included to show that prediction becomes harder without previous grade information.
