# Notes on articles in discussion section

**Chart of data size including sizes of train/test/validation datasets**

## Bai2019 - "Hourly PM2.5 concentration forecast , stacked ae model, emph on seasonality

* Uses seasonal predictions to provide an early warning system for pm2.5 levels

* pm2.5 is particularly dangerous in ambient air quality measures

* Knowledge-driven forecasting method. Requires a strong background in atmospheric science

* Data-driven approach is attractive in that it doesn't necessarily require as much hands-on experience and expertise in the field to be effective

* Paper considers data both seasonality and temporality

* Hourly data included pm2.5 levels as well as types of meteorological data: mean temp, mean wind speed, precipitation, and relative humidity.

* Split data into four seasonal categories:
    - Spring: March, Arpil, May
    - Summer: June, July, and August
    - Fall: September, October, November
    - Winter: December, January, February

* Findings: worst to best in terms of pollution were, winter -> spring -> fall -> summer
    - Winter months had higher pm2.5 concentrations than any other time
    - High temps, humidity, and strong thermal convection causes more pm2.5 diffusion

* Architecture: simple stacked autoencoder (SAE)

* SAE
    1. Collect historical data including pm2.5 levels and meteorological data
    2. Determine the seasonal group & analyze correlations between meteorological conditions of four seasons respectively
    3. normalize data
    4. train model (basically 4 different models) representing the four respective seasons
    5. forecast the next hours pm2.5 concentration
    6. output result of different seasons

* Winter and spring have higher proportion of human activities (at least in this area), so the shallow learning approach has difficulty discerning these discrepencies. Deep learning can capture the instinct information hidden in the historical data, and can thus obtain much more accurate forecasting results

* Used a shallow learning feed-forward network for comparison

## Quilodran - Urban Air Pollution Forecasts Generated from Latent Space Representations

* Autoencoder based on full-rank PCA to replicate computational fluid dynamic simulations of air pollution

* Layered approach: First reduction is using PCA, second is from a fully connected autoencoder (FCAE) that takes PCA as input, and uses it as a target and output
    - FCAE input and output use LeakyReLU activation funcion and Batch Normalization for faster convergence after each dense layer

* Once latent space of full-rank Pcs is obtained, it is used in tandom with a LSTM in order to make a prediction of the next time-step

* CFA is described by a differential equation that describes pollution diffusal 

* Full space corresponds to 256.4m trainable params, FCAE reduces it to just 98,828, which is far cheaper and easier to run 






