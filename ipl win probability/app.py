import pandas as pd
import streamlit as st
import pickle

teams = ['Royal Challengers Bengaluru',
         'Lucknow Super Giants',
         'Gujarat Titans',
         'Kolkata Knight Riders',
         'Rajasthan Royals',
         'Mumbai Indians',
         'Chennai Super Kings',
         'Sunrisers Hyderabad',
         'Delhi Capitals',
         'Punjab Kings']

cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata',
          'Jaipur', 'Durban', 'Ahmedabad', 'Pune', 'Hyderabad', 'Abu Dhabi',
          'Unknown', 'Visakhapatnam', 'Bengaluru', 'Dubai', 'Sharjah',
          'Navi Mumbai', 'Lucknow', 'Mohali']

st.title('IPL Win Predictor')

with open('ipl.pkl', 'rb') as f:
    pipe = pickle.load(f)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
    
col3, col4 = st.columns(2)
with col3:
    selected_city = st.selectbox("Select host city", sorted(cities))

with col4:
    target = st.number_input('Target')

col5, col6, col7 = st.columns(3)

with col5:
    score = st.number_input('Score')

with col6:
    overs = st.number_input('Overs completed')

with col7:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'run_left': [runs_left],
        'ball_left': [balls_left],
        'wickets_left': [wickets_left],
        'target_runs': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    st.write("Input DataFrame:", input_df)

    result = pipe.predict_proba(input_df)
    loss = 0.99 if rrr == 30.00 else result[0][0]
    win = 0.01 if rrr == 30.00 else result[0][1]
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
