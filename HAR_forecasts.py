import numpy as np
import pandas as pd
import statsmodels.api as sm

# Forecast horizon
forwin = 1

# Roling wstimiation window length
initwin = 1000

# Read in data
dats = pd.read_csv('SP500.csv')

# Convert to a datetime variable
dats['dts'] = pd.to_datetime(dats['dts'], format='%d/%m/%Y')

# Construct HAR variables
# Get Y variable
y = np.asarray(dats['rv'])
har_df = pd.DataFrame({'y': y})

# Get X variables
# .shift(1) gives lag of 1 day
har_df['rv_d'] = dats['rv'].shift(1)
har_df['rv_w'] = dats['rv'].rolling(5).mean().shift(1)
har_df['rv_m'] = dats['rv'].rolling(22).mean().shift(1)

# Remove lag days with NA
har_df = har_df.dropna().reset_index(drop=True)

# Adjust for forecast horizon
# Prediction target is average over horizon
nobs = har_df.shape[0] - forwin + 1
y_1 = np.zeros(nobs)
for i in np.arange(0, nobs, 1):
    y_1[i] = np.mean(har_df['y'].iloc[i:i+forwin])

har_df_1 = pd.DataFrame({'y': y_1})
har_df_1['rv_d'] = har_df['rv_d'].iloc[:nobs]
har_df_1['rv_w'] = har_df['rv_w'].iloc[:nobs]
har_df_1['rv_m'] = har_df['rv_m'].iloc[:nobs]

har_df_1 = har_df_1.dropna().reset_index(drop=True)

# Rolling forecasting loop
nfore = nobs - initwin

target = np.zeros((nfore, 1))
fore_har = np.zeros((nfore, 1))

for i in np.arange(0, nfore, 1):

    target[i] = har_df_1['y'].iloc[initwin + i]
    tmp_df = har_df_1.iloc[i:i+initwin,:]

    # Estimate
    mod = sm.OLS.from_formula('y ~ 1 + rv_d + rv_w + rv_m', data=tmp_df)
    har_res = mod.fit()

    # Forecast
    fore_d = tmp_df['y'].iloc[-1]
    fore_w = np.mean(tmp_df['y'].iloc[-5:])
    fore_m = np.mean(tmp_df['y'].iloc[-22:])
    x_fore = np.array([1, fore_d, fore_w, fore_m])

    fore_har[i] = x_fore @ har_res.params

# Wrte out forecasts
tmp_out = {'target': target.squeeze(), 'forecast': fore_har.squeeze()}
out_df = pd.DataFrame(tmp_out)
out_df.to_csv('fore_RV.csv', index=False)

