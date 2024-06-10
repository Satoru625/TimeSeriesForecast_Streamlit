import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from urllib.parse import quote

# データの読み込み
def load_data(file):
    df = pd.read_csv(file)
    return df

# メインの関数
def main():
    st.title("Forecast App")
    
    # ファイルのアップロード
    file = st.file_uploader("Upload CSV file", type="csv")
    
    if file is not None:
        # データの読み込み
        df = load_data(file)
        
        # 時間列と値列の指定
        time_col = st.selectbox("Select time column", df.columns)
        value_col = st.selectbox("Select value column", df.columns)
        df.index = pd.to_datetime(df[time_col])
        df.index.freq ='MS'
        df = df.resample('MS').asfreq()
        
        # 予測期間の指定
        periods = st.number_input("Number of periods to forecast", min_value=1, max_value=100, value=12)
        
        # 周期性の指定
        seasonal = st.checkbox("Include seasonality")
        
        # 季節性とトレンド性の指定
        trend = st.checkbox("Include trend")
        
        # ボタンを押すとモデルが実行されるようにする
        if st.button("Run Model"):
            # モデルの作成
            model = ExponentialSmoothing(df[value_col],
                                         seasonal='add' if seasonal else None,
                                         trend='add' if trend else None,
                                         seasonal_periods=12,
                                         initialization_method='estimated')

            # 予測結果の取得
            forecast = model.fit().forecast(periods)

            # 予測結果の可視化
            chart_data = pd.concat([df[value_col], pd.Series(forecast, index=pd.date_range(start=df.index[-1], periods=periods, freq='MS'))])

            fig, ax = plt.subplots()
            ax.plot(chart_data, color='black', label='Act')
            ax.plot(chart_data[-periods:], color='blue', label='Forecast')

            # Set the chart title and labels
            ax.set_title('Forecast Visualization')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')

            # Format the x-axis as YYYY-MM
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            # Set the y-axis as integer
            ax.ticklabel_format(style='plain',axis='y')

            # Display the legend
            ax.legend()

            # Display the chart
            st.pyplot(fig)

            # 予測結果の表形式表示
            forecast_dates = pd.date_range(start=df.index[-1], periods=periods, freq='MS').strftime('%Y-%m-%d')
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
            forecast_df.set_index('Date', inplace=True)
            st.table(forecast_df)

            # 予測結果のCSV出力
            csv = forecast_df.to_csv(index=True, encoding='utf-8-sig')
            filename = 'result.csv'
            href = f'<a href="data:text/csv;charset=utf-8,{quote(csv)}" download="{filename}">Download the Table</a>'
            st.markdown(href, unsafe_allow_html=True)


        
if __name__ == "__main__":
    main()