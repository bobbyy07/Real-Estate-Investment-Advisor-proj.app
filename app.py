# ============================================================
# REAL ESTATE INVESTMENT ADVISOR
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

CURRENT_YEAR = 2026

CLASSIFIER_PATH = Path("models/classification_model.pkl")
REGRESSOR_PATH = Path("models/regression_model.pkl")


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .result-good {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border: 1px solid #81c784;
    }

    .result-warning {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff8e1;
        border: 1px solid #ffca28;
    }

    .metric-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {CLASSIFIER_PATH}"
        )

    if not REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Regressor model not found: {REGRESSOR_PATH}"
        )

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    return classifier, regressor


# ============================================================
# 5. LOAD MODEL SAFELY
# ============================================================

classifier_model = None
regressor_model = None
models_loaded = False

try:

    classifier_model, regressor_model = load_models()

    models_loaded = True

except Exception as e:

    st.error("❌ Model loading failed.")

    st.code(str(e))

    st.info(
        """
        Make sure your project contains:

        models/
        ├── classifier_pipeline.pkl
        └── regressor_pipeline.pkl
        """
    )


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 Real Estate Investment Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predict property investment potential and estimate the property value
    after 5 years using Machine Learning.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 7. MODEL STATUS
# ============================================================

if models_loaded:

    st.success("✅ Machine Learning models loaded successfully.")

else:

    st.warning(
        "⚠️ Models are not available. Prediction cannot be performed "
        "until the model files are placed in the models folder."
    )


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title("🏠 Property Information")

st.sidebar.subheader("📍 Location")

state = st.sidebar.text_input(
    "State",
    value="Telangana"
)

city = st.sidebar.text_input(
    "City",
    value="Hyderabad"
)

locality = st.sidebar.text_input(
    "Locality",
    value="Madhapur"
)


# ============================================================
# 9. PROPERTY DETAILS
# ============================================================

st.sidebar.subheader("🏡 Property Details")

property_type = st.sidebar.selectbox(
    "Property Type",
    [
        "Apartment",
        "Villa",
        "Independent House",
        "Builder Floor",
        "Plot",
        "Other"
    ]
)

bhk = st.sidebar.slider(
    "BHK",
    min_value=1,
    max_value=10,
    value=2
)

size_sqft = st.sidebar.number_input(
    "Size in SqFt",
    min_value=100.0,
    max_value=10000.0,
    value=1200.0,
    step=50.0
)

current_price = st.sidebar.number_input(
    "Current Price (Lakhs)",
    min_value=1.0,
    max_value=5000.0,
    value=80.0,
    step=1.0
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1950,
    max_value=CURRENT_YEAR,
    value=2020,
    step=1
)

floor_no = st.sidebar.number_input(
    "Floor Number",
    min_value=0,
    max_value=100,
    value=2,
    step=1
)

total_floors = st.sidebar.number_input(
    "Total Floors",
    min_value=1,
    max_value=150,
    value=10,
    step=1
)


# ============================================================
# 10. PROPERTY FEATURES
# ============================================================

st.sidebar.subheader("🛋️ Property Features")

furnished_status = st.sidebar.selectbox(
    "Furnished Status",
    [
        "Unfurnished",
        "Semi-Furnished",
        "Fully Furnished"
    ]
)

owner_type = st.sidebar.selectbox(
    "Owner Type",
    [
        "Owner",
        "Dealer",
        "Builder"
    ]
)

facing = st.sidebar.selectbox(
    "Facing",
    [
        "North",
        "South",
        "East",
        "West",
        "North-East",
        "North-West",
        "South-East",
        "South-West"
    ]
)

availability_status = st.sidebar.selectbox(
    "Availability Status",
    [
        "Ready to Move",
        "Under Construction"
    ]
)

security = st.sidebar.selectbox(
    "Security",
    [
        "Yes",
        "No"
    ]
)

amenities = st.sidebar.text_input(
    "Amenities",
    value="Parking, Lift, Gym"
)


# ============================================================
# 11. INFRASTRUCTURE
# ============================================================

st.sidebar.subheader("🏫 Infrastructure")

nearby_schools = st.sidebar.slider(
    "Nearby Schools",
    min_value=0,
    max_value=20,
    value=4
)

nearby_hospitals = st.sidebar.slider(
    "Nearby Hospitals",
    min_value=0,
    max_value=20,
    value=2
)

public_transport = st.sidebar.slider(
    "Public Transport Accessibility",
    min_value=0,
    max_value=5,
    value=4
)

parking_space = st.sidebar.selectbox(
    "Parking Space",
    [
        "Yes",
        "No"
    ]
)


# ============================================================
# 12. FEATURE ENGINEERING
# ============================================================

price_per_sqft = (
    current_price * 100000
) / size_sqft


age_of_property = (
    CURRENT_YEAR - year_built
)

if age_of_property < 0:
    age_of_property = 0


# ============================================================
# 13. DISPLAY DERIVED FEATURES
# ============================================================

st.sidebar.markdown("---")

st.sidebar.write("### 📊 Calculated Features")

st.sidebar.write(
    f"**Price per SqFt:** ₹{price_per_sqft:,.2f}"
)

st.sidebar.write(
    f"**Property Age:** {age_of_property} years"
)


# ============================================================
# 14. CREATE RAW INPUT DATA
# ============================================================

input_data = pd.DataFrame({

    "State": [state],

    "City": [city],

    "Locality": [locality],

    "Property_Type": [property_type],

    "BHK": [bhk],

    "Size_in_SqFt": [size_sqft],

    "Price_in_Lakhs": [current_price],

    "Price_per_SqFt": [price_per_sqft],

    "Year_Built": [year_built],

    "Furnished_Status": [furnished_status],

    "Floor_No": [floor_no],

    "Total_Floors": [total_floors],

    "Age_of_Property": [age_of_property],

    "Nearby_Schools": [nearby_schools],

    "Nearby_Hospitals": [nearby_hospitals],

    "Public_Transport_Accessibility": [public_transport],

    "Parking_Space": [parking_space],

    "Security": [security],

    "Amenities": [amenities],

    "Facing": [facing],

    "Owner_Type": [owner_type],

    "Availability_Status": [availability_status]

})


# ============================================================
# 15. FUNCTION TO PREPARE FEATURES
# ============================================================

def prepare_input_for_model(data, model):
    """
    Prepare input according to the saved model.

    If the saved object is a Pipeline, raw data is passed directly.

    If the saved object is a normal sklearn estimator with
    feature_names_in_, categorical variables are one-hot encoded
    and columns are aligned to the model's expected columns.
    """

    data = data.copy()

    # --------------------------------------------------------
    # CASE 1: Pipeline
    # --------------------------------------------------------

    if hasattr(model, "named_steps"):

        return data


    # --------------------------------------------------------
    # CASE 2: Model trained directly on encoded dataframe
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        expected_columns = list(
            model.feature_names_in_
        )

        encoded_data = pd.get_dummies(
            data,
            drop_first=True
        )

        # Add missing columns
        for column in expected_columns:

            if column not in encoded_data.columns:

                encoded_data[column] = 0


        # Keep only model columns
        encoded_data = encoded_data[
            expected_columns
        ]

        return encoded_data


    # --------------------------------------------------------
    # CASE 3: Unknown model structure
    # --------------------------------------------------------

    return data


# ============================================================
# 16. PREDICTION FUNCTION
# ============================================================

def make_predictions(data):

    # --------------------------------------------------------
    # Prepare classifier input
    # --------------------------------------------------------

    classifier_input = prepare_input_for_model(
        data,
        classifier_model
    )

    # --------------------------------------------------------
    # Prepare regressor input
    # --------------------------------------------------------

    regressor_input = prepare_input_for_model(
        data,
        regressor_model
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    prediction = classifier_model.predict(
        classifier_input
    )[0]

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = None

    if hasattr(
        classifier_model,
        "predict_proba"
    ):

        probabilities = classifier_model.predict_proba(
            classifier_input
        )[0]

        # Find probability of predicted class
        if hasattr(
            classifier_model,
            "classes_"
        ):

            classes = list(
                classifier_model.classes_
            )

            try:

                prediction_index = classes.index(
                    prediction
                )

                probability = probabilities[
                    prediction_index
                ]

            except ValueError:

                probability = max(
                    probabilities
                )

        else:

            probability = max(
                probabilities
            )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    future_price = regressor_model.predict(
        regressor_input
    )[0]

    return (
        prediction,
        probability,
        future_price
    )


# ============================================================
# 17. ANALYZE BUTTON
# ============================================================

st.markdown("---")

analyze_button = st.button(
    "🔍 Analyze Investment",
    type="primary",
    use_container_width=True
)


# ============================================================
# 18. RUN PREDICTION
# ============================================================

if analyze_button:

    if not models_loaded:

        st.error(
            "❌ Prediction cannot be performed because "
            "the ML models were not loaded."
        )

        st.stop()


    try:

        # ====================================================
        # RUN MODEL
        # ====================================================

        (
            prediction,
            probability,
            future_price
        ) = make_predictions(
            input_data
        )


        # ====================================================
        # NORMALIZE CLASSIFICATION RESULT
        # ====================================================

        prediction_text = str(
            prediction
        ).strip().lower()


        good_values = [
            "1",
            "yes",
            "true",
            "good",
            "good investment"
        ]


        is_good_investment = (
            prediction_text in good_values
        )


        # ====================================================
        # CALCULATE GROWTH
        # ====================================================

        if current_price > 0:

            growth_percentage = (
                (
                    future_price -
                    current_price
                )
                / current_price
            ) * 100

        else:

            growth_percentage = 0


        # ====================================================
        # RESULTS HEADER
        # ====================================================

        st.markdown(
            "## 📊 Investment Analysis"
        )


        # ====================================================
        # RESULT COLUMNS
        # ====================================================

        col1, col2, col3 = st.columns(3)


        # ====================================================
        # INVESTMENT RESULT
        # ====================================================

        with col1:

            if is_good_investment:

                st.success(
                    "🏠 Good Investment"
                )

            else:

                st.warning(
                    "⚠️ Not a Good Investment"
                )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        with col2:

            if probability is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{probability * 100:.2f}%"
                )

            else:

                st.metric(
                    "Prediction Confidence",
                    "N/A"
                )


        # ====================================================
        # FUTURE PRICE
        # ====================================================

        with col3:

            st.metric(
                "Estimated Price After 5 Years",
                f"₹{future_price:,.2f} Lakhs"
            )


        # ====================================================
        # SECOND ROW
        # ====================================================

        col4, col5, col6 = st.columns(3)


        with col4:

            st.metric(
                "Current Price",
                f"₹{current_price:,.2f} Lakhs"
            )


        with col5:

            st.metric(
                "Estimated Growth",
                f"{growth_percentage:.2f}%"
            )


        with col6:

            st.metric(
                "Property Age",
                f"{age_of_property} Years"
            )


        # ====================================================
        # PRICE PROJECTION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📈 5-Year Property Price Projection"
        )

        years = [
            "Current",
            "Year 1",
            "Year 2",
            "Year 3",
            "Year 4",
            "Year 5"
        ]

        # Interpolate between current and predicted
        prices = np.linspace(
            current_price,
            future_price,
            6
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            years,
            prices,
            marker="o"
        )

        ax.set_title(
            "Estimated Property Price Growth"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Price (Lakhs)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        # ====================================================
        # PROPERTY SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🏠 Property Summary"
        )

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write(
                f"**State:** {state}"
            )

            st.write(
                f"**City:** {city}"
            )

            st.write(
                f"**Locality:** {locality}"
            )

            st.write(
                f"**Property Type:** {property_type}"
            )

            st.write(
                f"**BHK:** {bhk}"
            )

            st.write(
                f"**Size:** {size_sqft:,.0f} SqFt"
            )


        with summary_col2:

            st.write(
                f"**Current Price:** ₹{current_price:,.2f} Lakhs"
            )

            st.write(
                f"**Price/SqFt:** ₹{price_per_sqft:,.2f}"
            )

            st.write(
                f"**Year Built:** {year_built}"
            )

            st.write(
                f"**Floor:** {floor_no} / {total_floors}"
            )

            st.write(
                f"**Furnished:** {furnished_status}"
            )

            st.write(
                f"**Availability:** {availability_status}"
            )


        # ====================================================
        # INFRASTRUCTURE PROFILE
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🏫 Infrastructure Profile"
        )

        infrastructure_data = pd.DataFrame({

            "Infrastructure": [
                "Nearby Schools",
                "Nearby Hospitals",
                "Public Transport"
            ],

            "Score": [
                nearby_schools,
                nearby_hospitals,
                public_transport
            ]

        })


        fig2, ax2 = plt.subplots(
            figsize=(9, 4)
        )

        ax2.bar(
            infrastructure_data["Infrastructure"],
            infrastructure_data["Score"]
        )

        ax2.set_title(
            "Property Infrastructure"
        )

        ax2.set_ylabel(
            "Availability / Score"
        )

        ax2.grid(
            axis="y",
            alpha=0.3
        )

        st.pyplot(
            fig2,
            use_container_width=True
        )

        plt.close(fig2)


        # ====================================================
        # PROPERTY FEATURES
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🔑 Property Features"
        )

        feature_col1, feature_col2 = st.columns(2)


        with feature_col1:

            st.write(
                f"**Parking:** {parking_space}"
            )

            st.write(
                f"**Security:** {security}"
            )

            st.write(
                f"**Facing:** {facing}"
            )

            st.write(
                f"**Owner Type:** {owner_type}"
            )


        with feature_col2:

            st.write(
                f"**Amenities:** {amenities}"
            )

            st.write(
                f"**Nearby Schools:** {nearby_schools}"
            )

            st.write(
                f"**Nearby Hospitals:** {nearby_hospitals}"
            )

            st.write(
                f"**Public Transport:** {public_transport}"
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown("---")

        with st.expander(
            "🔧 Model Information"
        ):

            st.write(
                "Classifier model:"
            )

            st.code(
                str(
                    type(classifier_model)
                )
            )

            st.write(
                "Regressor model:"
            )

            st.code(
                str(
                    type(regressor_model)
                )
            )

            if hasattr(
                classifier_model,
                "feature_names_in_"
            ):

                st.write(
                    "Classifier expected features:"
                )

                st.write(
                    list(
                        classifier_model.feature_names_in_
                    )
                )

            if hasattr(
                regressor_model,
                "feature_names_in_"
            ):

                st.write(
                    "Regressor expected features:"
                )

                st.write(
                    list(
                        regressor_model.feature_names_in_
                    )
                )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except ValueError as e:

        st.error(
            "❌ Feature mismatch error."
        )

        st.write(
            """
            The saved model expects different features from
            the features supplied by the Streamlit application.
            """
        )

        st.code(
            str(e)
        )

        st.info(
            """
            If this error appears, the model was most likely
            trained with a different preprocessing structure.
            The classifier and regressor should ideally be saved
            together with their preprocessing in a Pipeline.
            """
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(e)
        )


# ============================================================
# 19. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Real Estate Investment Advisor | "
    "Machine Learning + Streamlit"
)
