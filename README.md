# 🏀 March Madness Bracket Predictor

## Overview
This project predicts the outcomes of the NCAA Men’s Basketball Tournament from the **Round of 64 through the Championship** using a machine learning model trained on historical tournament data and team-level season statistics.

Rather than predicting the entire bracket at once, the system predicts **win probabilities for individual matchups** and simulates the tournament round by round using the **official NCAA bracket** as input.

---

## Problem Statement
March Madness is a single-elimination tournament with high variability and frequent upsets. The goal of this project is to model game outcomes probabilistically by learning patterns from historical tournaments and team performance metrics, then use those probabilities to generate a complete tournament bracket and championship odds.

---

## Approach

### Key Idea
- Train a supervised ML model to predict the probability that **Team A beats Team B**
- Use those probabilities to simulate the tournament round by round
- Aggregate results using Monte Carlo simulation to estimate advancement and title probabilities

---

## Data

### Required Datasets
- **Historical NCAA tournament game results**
- **Team season statistics (pre-tournament only)**
- **Tournament seeds**
- **Team metadata (IDs and names)**

### Data Handling
- Raw data is stored unchanged in `data/raw/`
- Cleaned and feature-engineered data is stored in `data/processed/`
- Only information available **before the tournament begins** is used to avoid data leakage

---

## Feature Engineering
Team-level statistics are converted into **game-level difference features**, including:
- Seed difference
- Offensive efficiency difference
- Defensive efficiency difference
- Win percentage difference
- Point margin difference

This allows the model to learn relative strength between teams in a matchup.

---

## Models
Models evaluated include:
- Logistic Regression (baseline, interpretable)
- Random Forest
- Gradient Boosting

Models are evaluated using:
- Log loss
- ROC-AUC
- Probability calibration on held-out seasons

---

## Bracket Simulation
- The official NCAA bracket (teams, seeds, regions) is used as input
- Matchups are simulated round by round:
  - Round of 64
  - Round of 32
  - Sweet 16
  - Elite 8
  - Final Four
  - Championship
- Monte Carlo simulation (thousands of runs) is used to estimate:
  - Championship probabilities
  - Final Four appearances
  - Upset likelihoods

---

## Project Structure
march-madness-predictor/
├── data/
│ ├── raw/
│ ├── processed/
├── notebooks/
│ ├── EDA.ipynb
│ ├── model_training.ipynb
├── src/
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── simulate.py
├── results/
├── README.md

---

## How to Run

1. Place raw datasets in `data/raw/`
2. Run exploratory analysis and model training in `notebooks/`
3. Use scripts in `src/` to:
   - Load and process data
   - Train the model
   - Simulate the tournament
4. View outputs in `results/`

---

## Limitations
- Single-elimination randomness introduces high variance
- Injuries and late-season roster changes are not fully captured
- College basketball has a relatively small sample size per season
- Predictions are probabilistic, not deterministic

---

## Future Improvements
- Incorporate player-level statistics
- Add conference strength modeling
- Ensemble multiple models
- Build a web interface for bracket generation
- Use real-time stat updates

---

## Disclaimer
This project is for educational and analytical purposes only. Predictions are not guaranteed and should not be used for gambling or financial decision-making.

---

## Author
Han Li  
University of Michigan — Computer Science