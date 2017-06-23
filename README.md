# nanodegree_capstone
Project for Udacity Machine Learning Engineer nanodegree.
### About
- This is a kaggle competition named [Sberbank Russian Housing Market](https://www.kaggle.com/c/sberbank-russian-housing-market)
- [Data](https://www.kaggle.com/c/sberbank-russian-housing-market/data) is already given, which inclue:
	- train.csv, test.csv: information about individual transactions. The rows are indexed by the "id" field, which refers to individual transactions (particular properties might appear more than once, in separate transactions). These files also include supplementary information about the local area of each property.
	- macro.csv: data on Russia's macroeconomy and financial sector (could be joined to the train and test sets on the "timestamp" column)
	- sample_submission.csv: an example submission file in the correct format
	- data_dictionary.txt: explanations of the fields available in the other data files
- Files:
	- ```EDA.ipyhnb```: some exploratary analysis and visualizations.
	- ```lv1_class_utilities.py```: a class to generate out-of-sample data and prediction csvs.
	- ```lv1_preprocessing.py```: prepare data for first level models.
	- ```lv1_training.py```: multiple models to be trained.
	- ```lv2_train.py```: train second level model and make a submit.
	- ```report.pdf```
	- ```proposal.pdf```
