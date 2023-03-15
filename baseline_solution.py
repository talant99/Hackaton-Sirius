from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


FORECAST_HORIZON = 90  # 90 days


def hyperbolic_equation(t: float, qi: float, b: float, di: float) -> float:
    """Hyperbolic decline curve equation.
    :param t: Number of days since the well started operating
    :param qi: Maximum oil production level per day
    :param b: Hyperbolic decline constant
    :param di: Nominal decline at time step t=0
    :return: Expected oil production at time step t
    """
    return qi / ((1.0 + b * di * t) ** (1.0 / b))


def make_prediction(historical_values: pd.DataFrame,
                    forecast_horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Function creates oil production forecast for a single well
    fitting decline curve to previous data.
    :param historical_values: DataFrame containing parameters for one well
    :param forecast_horizon: Number of steps (days) in the forecast
    :returns: DataFrame containing forecast
    """
    time_series = historical_values['Дебит нефти'].values
    peak_index = np.argmax(time_series)
    qi = time_series[peak_index]

    latest_period = time_series[peak_index:]
    days = [i for i in range(1, len(latest_period) + 1)]

    popt, pcov = curve_fit(hyperbolic_equation, days, latest_period, bounds=(0, [qi, 2, 20]))
    qi, b, di = popt
    print(f'Fit Curve Variables: qi={qi}, b={b}, di={di}')

    pred_start = max(days) + 1
    pred_end = pred_start + forecast_horizon
    pred_days = np.array([i for i in range(pred_start, pred_end)], dtype=np.float)
    forecast = hyperbolic_equation(pred_days, *popt)

    date_range = pd.date_range(start='1992-04-11', freq='1D', periods=forecast_horizon)
    forecast_df = pd.DataFrame({'datetime': date_range, 'forecast': forecast})

    return forecast_df


def process_data():
    """Function loads training data from file and
    iterates over unique wells in the dataset
    making forecast for each time series.
    """
    train_path = '/Users/User/Documents/Sirius/data/train.csv'
    train_df = pd.read_csv(train_path)
    print(f'Loaded training data. Shape: {train_df.shape}')

    wells = list(train_df['Номер скважины'].unique())
    print(f'Number of unique wells: {len(wells)}')

    all_forecasts = []
    with tqdm(total=len(wells)) as pbar:
        for well in wells:
            print(f'Started processing well ID: {well}')
            well_df = train_df[train_df['Номер скважины'] == well]

            # Make prediction using Decline Curve Analysis
            forecats_df = make_prediction(well_df)
            forecats_df['Номер скважины'] = [well] * len(forecats_df)
            all_forecasts.append(forecats_df)

            pbar.update(1)

    all_forecasts = pd.concat(all_forecasts)
    print(f'Completed data processing. Forecast shape: {all_forecasts.shape}')
    print(f'Number of unique wells: {len(all_forecasts["Номер скважины"].unique())}')

    all_forecasts.to_csv('baseline_forecast.csv', index=False, encoding="utf-8")
    print('Saved forecast to "baseline_forecast.csv"')

if __name__ == '__main__':
    process_data()
