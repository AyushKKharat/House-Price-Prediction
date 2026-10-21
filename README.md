# House Price Predictor

A small machine learning project that estimates what a house should cost from things a buyer actually cares about: size, locality, age of the building, distance from the city centre, and so on.

I built it to connect the statistics I had studied with a problem that feels real. The aim was never just a good score. It was to follow the whole path from a messy spreadsheet to a prediction I can explain in plain words.

## What the project does

1. Builds a realistic house dataset (2,000 homes).
2. Checks the data, fills the gaps and chooses the features worth using.
3. Trains four regression models and compares them fairly.
4. Compares predicted prices with actual prices on houses the model has never seen.
5. Saves the best model so a single command can price a new house.

## About the data

I generated the data myself instead of downloading it. Public housing datasets are often locked behind logins or licences, or come with so much cleaning that the modelling gets crowded out. Generating the data also let me control how it behaves.

`src/generate_data.py` creates 2,000 houses priced with rules a buyer would recognise:

- bigger homes cost more
- better localities cost more (Premium, Mid-range, Budget)
- older buildings are cheaper
- homes closer to the centre are dearer
- higher floors are worth more with a lift, and less without one
- furnished homes, parking and extra bathrooms add a little
- random noise is mixed in so no two similar houses cost exactly the same

About 2% of the age and distance values are left blank on purpose, so the data-preparation step has something real to deal with. Prices are in **lakhs of rupees** (1 lakh = 100,000).

| Column | Meaning |
|---|---|
| `locality` | Premium, Mid-range or Budget |
| `area_sqft` | Built-up area |
| `bedrooms`, `bathrooms` | Room counts |
| `age_years` | Age of the building |
| `floor` | Floor number (0 = ground) |
| `parking_spots` | 0, 1 or 2 |
| `distance_to_center_km` | Distance to the city centre |
| `furnishing` | Unfurnished, Semi-furnished or Furnished |
| `has_lift` | 1 if the building has a lift |
| `price_lakhs` | **What we predict** |

## How the model works

- **Cleaning:** missing numbers are filled with the column median, numeric columns are scaled, and text columns are one-hot encoded.
- **Log of the price:** a house that is 10% bigger costs roughly 10% more, not a fixed amount more. Training on the log of the price matches this, and predictions are converted back to lakhs.
- **Four models:** Linear Regression, Ridge Regression, Random Forest and Gradient Boosting.
- **Fair comparison:** I held back 20% of the houses as a test set. Models were ranked by 5-fold cross-validation on the training data only, so the test set stayed untouched until the final check.

## Results

Scores on the 400 houses held out for testing:

| Model | CV R² | Test R² | MAE (lakhs) | RMSE (lakhs) | Avg error |
|---|---|---|---|---|---|
| Linear Regression | 0.930 | 0.928 | 4.15 | 6.15 | 8.4% |
| Ridge Regression | 0.930 | 0.928 | 4.15 | 6.14 | 8.4% |
| Random Forest | 0.933 | 0.946 | 4.05 | 5.32 | 8.2% |
| **Gradient Boosting** | **0.954** | **0.965** | **3.26** | **4.26** | **6.6%** |

Gradient Boosting won. On unseen houses it is typically off by about 3.3 lakhs, or roughly 6.6% of the price.

![Model comparison](outputs/model_comparison.png)

![Predicted vs actual](outputs/predicted_vs_actual.png)

Points close to the dashed line are accurate predictions. The spread grows for the most expensive homes, which is expected: there are fewer of them to learn from.

### What drives the price

![Feature importance](outputs/feature_importance.png)

Area and locality do most of the work, followed by building age and distance to the centre. Bedrooms scores close to zero, but that does not mean bedrooms are irrelevant. Bedroom count mostly follows area, so once the model knows the area it learns almost nothing new from it. Shuffling a column only shows what the model loses *beyond what the other columns already tell it*.

The other charts (`price_distribution.png`, `correlation_heatmap.png`, `error_distribution.png`) are in the `outputs/` folder, along with `test_predictions.csv`, which lists every test house with its actual price, predicted price and error.

## Running it yourself

You need Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py      # creates data/houses.csv
python src/train.py              # trains, compares, saves model + charts
```

Price a new house:

```bash
python src/predict.py --area 1200
python src/predict.py --area 950 --bedrooms 2 --bathrooms 2 --age 8 \
    --floor 4 --parking 1 --distance 3.5 --locality Premium \
    --furnishing Semi-furnished --has-lift 1
```

Example output:

```
Estimated price: Rs. 68.16 lakhs (about Rs. 0.68 crore)
On unseen houses the model is typically off by about 6.6%, so expect roughly Rs. 63.7 to 72.7 lakhs.
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```
house-price-predictor/
├── data/houses.csv              generated dataset
├── models/                      saved model (joblib)
├── outputs/                     charts, metrics.json, test_predictions.csv, training_log.txt
├── src/
│   ├── config.py                paths and column names in one place
│   ├── generate_data.py         builds the synthetic dataset
│   ├── train.py                 prepares data, trains, compares, saves
│   └── predict.py               prices a single house from the command line
├── tests/test_project.py        15 tests for data, model and predictions
└── requirements.txt
```

## What I learned

- **Preparation takes more thought than modelling.** Missing values, scaling and encoding decide whether a model can use the data at all.
- **A fair test matters.** Choosing the winner on cross-validation and keeping the test set untouched gave me a number I can trust.
- **Simple models go a long way.** Plain linear regression reached R² of 0.93. Gradient Boosting added about 3.7 points of R², by catching interactions such as how much a square foot is worth in each locality.
- **Explaining beats scoring.** Predicted-vs-actual plots, error percentages and feature importance made the results something I could talk about, not just a number.

## Limitations

- The data is synthetic. Its patterns are cleaner than the real market, so these scores would be lower on real listings, which carry noise nobody wrote down (street, view, builder reputation, negotiation).
- The model only knows the features in the dataset and can be wrong for homes outside the sizes and localities it saw (350 to about 2,300 sq ft).
- The Rs. figures come from invented prices and say nothing about any actual city.

## Ideas for next steps

- Swap in a real dataset and see how much the scores drop.
- Tune hyperparameters with a grid or random search.
- Add a small web page or Streamlit app on top of `predict.py`.
