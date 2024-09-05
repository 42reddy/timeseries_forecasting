# timeseries_forecasting
Stock price prediction using deep learning and frequency decomposition

Time series data of S&P 500 index has been decompsed into its dominant frequencies and the resultant series are fed into a a CNN-LSTM network. The architecture is primarily inspired from Stock price prediction using deep learning and frequency decomposition, Hadi Rezaei et al. https://doi.org/10.1016/j.eswa.2020.114332.

Time series data of stock closing prices are decomposed using STL decomposition, further used as input to stacked CNN-LSTM architecture.

hyperparameters are manually optimised for the best fit. the model seemed to struggle fitting the data during the covid pandemic, which indeed is an anamoly compared to the rest of the pattern. alternatively, data from multiple instruments excluding the pandemic time period were used to train the model and the RMS error has improved to prior.

alternate decomposition methods could be explored as well as adapting bayesian optimization to optimize the parameters.

