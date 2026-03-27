## YouTube Shorts Engagement Predictor

Predicts the view-to-like ratio of a YouTube Shorts video before posting.

Built on real channel data pulled via YouTube Data API v3.

**Pipeline:**
1. Collect video stats via YouTube Data API
2. Engineer features (duration, day, hour, title length)
3. Train Random Forest regression model
4. Serve predictions via FastAPI
5. Visualize via Streamlit dashboard
6. Deployed on Render.com

**Tech Stack:** Python, Pandas, Scikit-learn, FastAPI, Streamlit, Joblib
