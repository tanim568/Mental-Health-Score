# Mental Health Score Prediction

A machine learning web application that predicts a reported mental health score from social media usage and daily-habit features. The project combines a scikit-learn regression pipeline, a FastAPI backend, and a browser-based frontend.

> **Important:** This is an educational machine learning project, not a medical assessment or mental health diagnosis. Predictions should not be used for clinical decisions.

## Live Demo

- **Website:** https://mental-health-score-1-104e.onrender.com
- **API base URL:** https://mental-health-score-3vfu.onrender.com
- **Interactive API documentation:** https://mental-health-score-3vfu.onrender.com/docs

## Features

- Explore and prepare a student social-media and mental-health dataset.
- Clean duplicate rows and handle negative physical-activity values.
- Preprocess numeric and categorical features with a scikit-learn pipeline.
- Compare Linear Regression and Random Forest regression models.
- Evaluate models with R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- Serve predictions through a FastAPI endpoint.
- Submit input values through a simple HTML, CSS, and JavaScript web interface.

## Model Results

The project notebook reports these held-out test results:

| Model | Test R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.7398 | 0.5362 | 0.6760 |
| Random Forest (default) | 0.8776 | 0.3472 | 0.4637 |
| Random Forest (tuned) | 0.8650 | 0.3689 | 0.4869 |

In these experiments, the default Random Forest achieved the strongest test-set results among the three models. These values describe the notebook's evaluation; they do not guarantee performance on other populations or future data.

## Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine learning:** scikit-learn, Joblib
- **Backend/API:** FastAPI, Pydantic, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render

## Project Structure

- `ML_Project.ipynb` — data exploration, preprocessing, model training, and evaluation.
- `Student Social Media And Mental Health Impact.csv` — dataset used by the notebook.
- `main.py` — FastAPI application and prediction endpoint.
- `Mental_Health_Model.pkl` — serialized model pipeline loaded by the API.
- `index.html`, `style.css`, `script.js` — web interface and client-side API requests.
- `requirements.txt` — API dependencies.
- `requirements-notebook.txt` — notebook-related dependencies.

## Run the API Locally

1. Clone the repository and open its folder.
2. Create and activate a Python virtual environment.
3. Install the API dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the API:

   ```bash
   uvicorn main:app --reload
   ```

5. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore and test the API.

The saved model file `Mental_Health_Model.pkl` must be present in the project directory for the API to load it. The deployed frontend sends prediction requests to the API URL configured in `script.js`.

## Limitations and Responsible Use

- The model estimates a score from patterns in the available dataset; it does not establish cause and effect.
- Results depend on the dataset, feature choices, preprocessing, and train-test split, and may not generalize to other student populations.
- A predicted score is not a diagnosis and should not replace professional mental-health support.
- Avoid using the model to make clinical, educational, or other high-stakes decisions about individuals.

## Future Improvements

- Add model validation on an independent dataset.
- Improve input validation and error handling in the web interface.
- Add automated tests for the API and preprocessing pipeline.
- Document the dataset source, licensing, and data limitations in more detail.
