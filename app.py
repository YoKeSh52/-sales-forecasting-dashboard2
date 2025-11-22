import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import BytesIO
from prophet import Prophet

st.set_page_config(page_title="📈 Advanced Sales Forecasting", layout="wide")

# 📥 Upload CSV
st.title("📊 Sales Demand Forecasting Dashboard")
uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip().lower() for col in df.columns]

    if 'date' not in df.columns or 'sales' not in df.columns:
        st.error("CSV must contain 'Date' and 'Sales' columns.")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)

        # 📅 Date Filtering
        st.sidebar.header("📅 Date Range Filter")
        start_date = st.sidebar.date_input("Start Date", df.index.min())
        end_date = st.sidebar.date_input("End Date", df.index.max())
        filtered_df = df.loc[start_date:end_date]



        st.subheader("📈 Filtered Sales Data")
        st.line_chart(filtered_df['sales'])

        # 🔍 Decomposition
        st.subheader("🔍 Trend & Seasonality")
        try:
            decomposition = seasonal_decompose(filtered_df['sales'], model='additive', period=7)
            fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(filtered_df['sales'], label='Original')
            ax[1].plot(decomposition.trend, label='Trend')
            ax[2].plot(decomposition.seasonal, label='Seasonality')
            ax[3].plot(decomposition.resid, label='Residuals')
            for a in ax: a.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning("Decomposition failed: " + str(e))

        # 📌 Model Selection
        st.sidebar.header("⚙️ Forecast Settings")
        model_choice = st.sidebar.selectbox("Choose Forecasting Model", ["ARIMA", "Prophet"])
        forecast_period = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

        # 🔮 Forecasting
        st.subheader("🔮 Forecast Results")
        if model_choice == "ARIMA":
            p = st.sidebar.slider("ARIMA p", 0, 10, 5)
            d = st.sidebar.slider("ARIMA d", 0, 2, 1)
            q = st.sidebar.slider("ARIMA q", 0, 10, 0)

            model = ARIMA(filtered_df['sales'], order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_period)
            forecast_index = pd.date_range(start=filtered_df.index[-1] + pd.Timedelta(days=1), periods=forecast_period)
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

            st.line_chart(pd.concat([filtered_df['sales'], forecast_df['Forecast']]))

            # 📉 Residuals
            st.subheader("📉 Residual Analysis")
            residuals = model_fit.resid
            fig2, ax2 = plt.subplots()
            ax2.plot(residuals)
            ax2.set_title("Model Residuals")
            st.pyplot(fig2)

            # 📊 Error Metrics
            st.subheader("📊 Error Metrics")
            try:
                actual = filtered_df['sales'][-forecast_period:]
                predicted = forecast[:len(actual)]
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mae = mean_absolute_error(actual, predicted)
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("MAE", f"{mae:.2f}")
                st.metric("MAPE", f"{mape:.2f}%")
            except:
                st.warning("Not enough data for error metrics.")

        elif model_choice == "Prophet":
            prophet_df = filtered_df.reset_index().rename(columns={"date": "ds", "sales": "y"})
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_period)
            forecast = m.predict(future)
            forecast_df = forecast[['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'Forecast'})

            st.line_chart(pd.concat([filtered_df['sales'], forecast_df['Forecast']]))

        # 🚨 Anomaly Detection
        st.subheader("🚨 Anomaly Detection")
        threshold = st.slider("Z-score Threshold", 1.0, 3.0, 2.0)
        z_scores = (filtered_df['sales'] - filtered_df['sales'].mean()) / filtered_df['sales'].std()
        anomalies = filtered_df[np.abs(z_scores) > threshold]
        st.write("Detected Anomalies", anomalies)

        # 📥 Download Forecast
        def convert_df(df):
            buffer = BytesIO()
            df.to_csv(buffer)
            buffer.seek(0)
            return buffer

        st.download_button(
            label="📥 Download Forecast CSV",
            data=convert_df(forecast_df),
            file_name="forecast.csv",
            mime="text/csv"
        )