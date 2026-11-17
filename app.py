import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #81A1BD;
}

.main-title {
    font-size: 38px;
    font-weight: bold;
    color: #163A5F;
    text-align: center;
}

.subtitle {
    font-size: 17px;
    color: #64748B;
    text-align: center;
}

.result-card {
    background-color: #E0F2FE;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #BAE6FD;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("model.pkl not found!")
        st.stop()

    return joblib.load("model.pkl")

package = load_model()

model = package["model"]
thresholds = package["thresholds"]
features = package["features"]

# ==========================================
# WEATHER INFORMATION
# ==========================================

weather_info = {
    "sun": {
        "emoji": "☀️",
        "description": "Sunny weather with bright sunshine and generally clear skies."
    },
    "rain": {
        "emoji": "🌧️",
        "description": "Rainy weather with precipitation. Consider bringing an umbrella."
    },
    "snow": {
        "emoji": "❄️",
        "description": "Snowy weather with possible snowfall and cold conditions."
    },
    "fog": {
        "emoji": "🌫️",
        "description": "Foggy weather with reduced visibility."
    },
    "drizzle": {
        "emoji": "🌦️",
        "description": "Light rainfall or drizzle with potentially cloudy conditions."
    }
}

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🌦️ Weather System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Weather Prediction",
        "About the Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Weather Prediction "
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<p class="main-title">🌦️ Weather Prediction System</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Predict weather conditions.</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Welcome!")

    st.write(
        """
        This application uses a trained Random Forest
        Classification model to predict weather conditions.

        Users can answer four simple questions about
        weather characteristics to receive a predicted
        weather category.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction Classes", "5")

    with col2:
        st.metric("Input Questions", "4")

    with col3:
        st.metric("Algorithm", "Random Forest")

    st.markdown("---")

    st.subheader("Available Weather Predictions")

    weather_cols = st.columns(5)

    for i, (weather, info) in enumerate(weather_info.items()):
        with weather_cols[i]:
            st.markdown(f"### {info['emoji']}")
            st.write(weather.capitalize())

    st.markdown("---")

    st.info(
        "Go to Weather Prediction in the sidebar "
        "to start predicting weather."
    )

# ==========================================
# WEATHER PREDICTION PAGE
# ==========================================

elif page == "Weather Prediction":

    st.title("🌤️ Weather Prediction")

    st.write(
        "Answer the following YES/NO questions "
        "based on the weather conditions you want to evaluate."
    )

    st.markdown("---")

    st.subheader("Weather Condition Questionnaire")

    st.caption(
        "YES means the measurement is above the "
        "typical median value in the training dataset. "
        "NO means it is at or below that value."
    )

    questions = {
        "precipitation": (
            "🌧️ Is the precipitation higher than usual?"
        ),
        "temp_max": (
            "🌡️ Is the maximum temperature higher than usual?"
        ),
        "temp_min": (
            "🌡️ Is the minimum temperature higher than usual?"
        ),
        "wind": (
            "💨 Is the wind speed higher than usual?"
        )
    }

    answers = {}

    for feature, question in questions.items():

        st.markdown(f"**{question}**")

        answers[feature] = st.radio(
            label=f"Answer for {feature}",
            options=["No", "Yes"],
            horizontal=True,
            key=f"question_{feature}",
            label_visibility="collapsed"
        )

        st.caption(
            f"Reference median: {thresholds[feature]:.2f}"
        )

        st.markdown("")

    st.markdown("---")

    if st.button(
        "🔮 Predict Weather",
        type="primary",
        use_container_width=True
    ):

        # Convert answers into binary values
        input_data = pd.DataFrame([{
            feature: 1 if answers[feature] == "Yes" else 0
            for feature in features
        }])

        # Predict weather
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = max(probabilities) * 100

        info = weather_info.get(
            prediction,
            {
                "emoji": "🌤️",
                "description": "Predicted weather condition."
            }
        )

        st.markdown("---")

        st.success("Prediction Completed Successfully!")

        st.markdown(
            f"""
            <div class="result-card">
                <h2>Predicted Weather</h2>
                <h1 style="font-size:65px;">
                    {info['emoji']}
                </h1>
                <h1 style="color:#075985;">
                    {prediction.upper()}
                </h1>
                <p>{info['description']}</p>
                <h3>Model Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Prediction Probability")

        probability_df = pd.DataFrame({
            "Weather": model.classes_,
            "Probability": probabilities * 100
        }).sort_values(
            by="Probability",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Weather")
        )

        st.dataframe(
            probability_df.style.format({
                "Probability": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "The confidence value represents the model's "
            "estimated probability for its predicted class. "
            "It is not a guarantee of actual weather."
        )

# ==========================================
# ABOUT THE MODEL PAGE
# ==========================================

elif page == "About the Model":

    st.title("🧠 About the Machine Learning Model")

    st.write(
        """
        This project uses a Random Forest Classifier
        trained using the Kaggle Weather Prediction dataset.
        """
    )

    st.markdown("---")

    st.subheader("Prediction Classes")

    for weather, info in weather_info.items():

        st.markdown(
            f"**{info['emoji']} {weather.capitalize()}**"
        )

        st.write(info["description"])

    st.markdown("---")

    st.subheader("Project Dataset")

    st.markdown(
        "[View Original Kaggle Dataset](https://www.kaggle.com/"
        "datasets/ananthr1/weather-prediction)"
    )

    st.markdown("---")

    st.caption(
        "Weather Prediction System | Machine Learning Project"
    )
