# SoilClassification_annam

This Repo is about our Team The_Iterators work on the preliminary Round of Annam.AI hackathon. we have given solution for two Kaggle competition conducted where 1st one is about classification of soil type where 4 soil type where given and asked to classify the test images and 2nd one is about classifying wheather it is a soil or any other random image. 

Kaggle competition - 1:
  - here we have been asked to predict wheather the gvien image is of which soil type among (alluvial soil, clay soil, red soil , black soil) out of this four we need to train the model with given train iamges.
  - here i have used colab as my platfoem to train so the data is been uploaded to drive and loaded.
  - we used Random forest and XGboost model for this task and we scored F1 score of 93.33% in test images.

Kaggle competition - 2:
  - here we have been asked to predict wheather the gvien image is soil or any other random image we need to train the model with given train iamges full of different soil type and here we have images of one type if any other images encountered then the output has to be 0.
  - here we did use IsolatedForest model for mono class classification and got F1 score of 76%

⚙️ Setup and Run Instructions
Follow the steps below to set up the environment and run the soil type classification project:
📦 1. Clone the Repository

📑 2. Install Dependencies
   - Install all required Python packages using the requirements.txt file

📁 3. Access the Dataset
   - ⚠️ Note: The dataset used in this project is private and available only to registered participants. Place the dataset in the designated data/ directory after downloading. use you own dataset

📓 4. Run the Notebooks
The project contains the following main notebooks:
   - notebooks/annamkaggle(1,2).ipynb: Contains the full pipeline including data loading, preprocessing (image-to-array conversion and dimensionality reduction), model training using Random Forest, and evaluation.
