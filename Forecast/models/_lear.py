# This file is part of epftoolbox (https://github.com/jeslago/epftoolbox)
# Modified by Maria Margarida Mascarenhas, 2025
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0).

import numpy as np
import pandas as pd
from statsmodels.robust import mad
import os
import holidays
import joblib


from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.linear_model import LassoCV
from ..data import scaling
from ..data import read_data
from ..evaluation import MAE, sMAPE

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import logging
import warnings

logging.captureWarnings(True)


class LEAR(object):
    """Class to build a LEAR model, recalibrate it, and use it to predict DA electricity prices.
    
    An example on how to use this class is provided :ref:`here<learex2>`.
    
    Parameters
    ----------
    calibration_window : int, optional
        Calibration window (in days) for the LEAR model.
        
    """
    
    def __init__(self, calibration_window=364 * 3, recalibrate_frequency='daily'):

        # Attribute to store the recalibration frequency: 'daily', 'weekly', 'monthly', or 'once'
        self.recalibrate_frequency = recalibrate_frequency

        # Attribute to store the last date on which we recalibrated the model
        self.last_recalibration_date = None
        # Alternatively, we can store a boolean if the user only wants a single calibration
        self.has_recalibrated_once = False
        
        # Calibration window in hours
        self.calibration_window = calibration_window
        
    
    def recalibrate(self, Xtrain, Ytrain):
        """Function to recalibrate the LEAR model using LassoCV for cross-validated selection of alpha.

        Parameters
        ----------
        Xtrain : numpy.array
            Input in training dataset. It should be of size [n,m] where n is the number of days
            in the training dataset and m the number of input features.

        Ytrain : numpy.array
            Output in training dataset. It should be of size [n,24] where n is the number of days 
            in the training dataset and 24 are the 24 prices of each day.

        Returns
        -------
        numpy.array
            The prediction of day-ahead prices after recalibrating the model.
        """

        # Applying Invariant, aka asinh-median transformation to the prices
        [Ytrain], self.scalerY = scaling([Ytrain], 'Norm')

        # Rescaling all inputs except dummies (8 last features = 7 day_of_week + 1 is_holiday) 
        [Xtrain_no_dummies], self.scalerX = scaling([Xtrain[:, :-8]], 'Norm')
        Xtrain[:, :-8] = Xtrain_no_dummies

        self.models = {}
        for h in range(24):
            # Using LassoCV for alpha selection and model fitting
            model = LassoCV(cv=5, max_iter=5000, tol=1e-3, verbose=True)  # cv=5 specifies the number of folds in cross-validation
            model.fit(Xtrain, Ytrain[:, h])

            self.models[h] = model

    def predict(self, X):
        """Function that makes a prediction using some given inputs.
        
        Parameters
        ----------
        X : numpy.array
            Input of the model.
        
        Returns
        -------
        numpy.array
            An array containing the predictions.
        """

        # Predefining predicted prices
        Yp = np.zeros(24)

        # Rescaling all inputs except dummies (8 last features = 7 day_of_week + 1 is_holiday) 
        X_no_dummies = self.scalerX.transform(X[:, :-8])
        X[:, :-8] = X_no_dummies

        # Predicting the current date using a recalibrated LEAR
        for h in range(24):

            # Predicting test dataset and saving
            Yp[h] = self.models[h].predict(X)
        
        Yp = self.scalerY.inverse_transform(Yp.reshape(1, -1))

        return Yp
    
    def save_model(self, next_day_date):

        date_str = next_day_date.strftime('%Y-%m-%d')
        models_dir = 'saved_models/LEAR'

        model_data = {
            'models': self.models,
            'scalerX': self.scalerX,
            'scalerY': self.scalerY
        }

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        try:
            joblib.dump(model_data, f'{models_dir}/LEAR_CW{self.calibration_window}_{date_str}.joblib')
        except Exception as e:
            print(f"Failed to save model: {e}")
        
    def recalibrate_predict(self, Xtrain, Ytrain, Xtest, next_day_date):
        """Function that first recalibrates the LEAR model and then makes a prediction.

        The function receives the training dataset, and trains the LEAR model. Then, using
        the inputs of the test dataset, it makes a new prediction.
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input of the training dataset.
        Xtest : numpy.array
            Input of the test dataset.
        Ytrain : numpy.array
            Output of the training dataset.
        
        Returns
        -------
        numpy.array
            An array containing the predictions in the test dataset.
        """
        
        if self.recalibrate_frequency == 'once':
            # Recalibrate only if we haven't done it yet
            if not self.has_recalibrated_once:
                self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain)
                self.has_recalibrated_once = True
                print(f'\n recalibrated at: {next_day_date}\n')

        elif self.recalibrate_frequency == 'weekly':
            # Recalibrate if last_recalibration_date is None (never done)
            # or if at least 7 days have passed
            if (self.last_recalibration_date is None) \
               or ((next_day_date - self.last_recalibration_date).days >= 7):
                self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain)
                self.last_recalibration_date = next_day_date
                print(f'\n recalibrated at: {next_day_date}\n')


        elif self.recalibrate_frequency == 'monthly':
            # Recalibrate if last_recalibration_date is None
            # or the month changed
            if (self.last_recalibration_date is None) \
               or (next_day_date.year != self.last_recalibration_date.year) \
               or (next_day_date.month != self.last_recalibration_date.month):
                self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain)
                self.last_recalibration_date = next_day_date
                print(f'\n recalibrated at: {next_day_date}\n')
        else:
            # Default: 'daily' – recalibrate every day
            self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain)
            self.last_recalibration_date = next_day_date
            print(f'\n recalibrated at: {next_day_date}\n')

        Yp = self.predict(X=Xtest)

        #self.save_model(next_day_date)

        return Yp

    def _build_and_split_XYs(self, df_train, df_test=None, date_test=None):
        
        """Internal function that generates the X,Y arrays for training and testing based on pandas dataframes
        
        Parameters
        ----------
        df_train : pandas.DataFrame
            Pandas dataframe containing the training data
        
        df_test : pandas.DataFrame
            Pandas dataframe containing the test data
        
        date_test : datetime, optional
            If given, then the test dataset is only built for that date
        
        Returns
        -------
        list
            [Xtrain, Ytrain, Xtest] as the list containing the (X,Y) input/output pairs for training, 
            and the input for testing
        """

        # Checking that the first index in the dataframes corresponds with the hour 00:00 
        if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
            print('Problem with the index')

        # 
        # Defining the number of Exogenous inputs
        n_exogenous_inputs = len(df_train.columns) - 1

        # 96 prices + n_exogenous * (24 * 3 exogeneous) + 7 weekday dummies
        # Price lags: D-1, D-2, D-3, D-7
        # Exogeneous inputs lags: D, D-1, D-7
        n_features = 96 + 7 + n_exogenous_inputs * 72 + 1 


        # Extracting the predicted dates for testing and training. We leave the first week of data
        # out of the prediction as we the maximum lag can be one week
        
        # We define the potential time indexes that have to be forecasted in training
        # and testing
        indexTrain = df_train.loc[df_train.index[0] + pd.Timedelta(weeks=1):].index

        # For testing, the test dataset is different whether depending on whether a specific test
        # dataset is provided
        if date_test is None:
            indexTest = df_test.loc[df_test.index[0] + pd.Timedelta(weeks=1):].index
        else:
            indexTest = df_test.loc[date_test:date_test + pd.Timedelta(hours=23)].index

        # We extract the prediction dates/days.
        predDatesTrain = indexTrain.round('1H')[::24]                
        predDatesTest = indexTest.round('1H')[::24]

        # We create two dataframe to build XY.
        # These dataframes have as indices the first hour of the day (00:00)
        # and the columns represent the 23 possible horizons/dates along a day
        indexTrain = pd.DataFrame(index=predDatesTrain, columns=['h' + str(hour) for hour in range(24)])
        indexTest = pd.DataFrame(index=predDatesTest, columns=['h' + str(hour) for hour in range(24)])
        for hour in range(24):
            indexTrain.loc[:, 'h' + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
            indexTest.loc[:, 'h' + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)

        
        # Preallocating in memory the X and Y arrays          
        Xtrain = np.zeros([indexTrain.shape[0], n_features])
        Xtest = np.zeros([indexTest.shape[0], n_features])
        Ytrain = np.zeros([indexTrain.shape[0], 24])

        # Index that 
        feature_index = 0
        
        #
        # Adding the historial prices during days D-1, D-2, D-3, and D-7
        #

        # For each hour of a day
        for hour in range(24):
            # For each possible past day where prices can be included
            for past_day in [1, 2, 3, 7]:

                # We define the corresponding past time indexs using the auxiliary dataframses 
                pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                    pd.Timedelta(hours=24 * past_day)
                pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                    pd.Timedelta(hours=24 * past_day)

                # We include the historical prices at day D-past_day and hour "h" 
                Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Price']
                Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Price']
                feature_index += 1

        #
        # Adding the exogenous inputs during days D, D-1,  D-7
        #
        # For each hour of a day
        for hour in range(24):
            # For each possible past day where exogenous inputs can be included
            for past_day in [1, 7]:
                # For each of the exogenous input
                for exog in range(1, n_exogenous_inputs + 1):

                    # Definying the corresponding past time indexs using the auxiliary dataframses 
                    pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                        pd.Timedelta(hours=24 * past_day)
                    pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                        pd.Timedelta(hours=24 * past_day)

                    # Including the exogenous input at day D-past_day and hour "h" 
                    Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Exogenous ' + str(exog)]                    
                    Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Exogenous ' + str(exog)]
                    feature_index += 1

            # For each of the exogenous inputs we include feature if feature selection indicates it
            for exog in range(1, n_exogenous_inputs + 1):
                
                # Definying the corresponding future time indexs using the auxiliary dataframses 
                futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
                futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

                # Including the exogenous input at day D and hour "h" 
                Xtrain[:, feature_index] = df_train.loc[futureIndexTrain, 'Exogenous ' + str(exog)]        
                Xtest[:, feature_index] = df_test.loc[futureIndexTest, 'Exogenous ' + str(exog)] 
                feature_index += 1

        #
        # Adding the dummy variables that depend on the day of the week. Monday is 0 and Sunday is 6
        #
        # For each day of the week
        for dayofweek in range(7):
            Xtrain[indexTrain.index.dayofweek == dayofweek, feature_index] = 1
            Xtest[indexTest.index.dayofweek == dayofweek, feature_index] = 1
            feature_index += 1

        # Extracting the predicted values Y
        for hour in range(24):
            # Defining time index at hour h
            futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
            futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

            # Extracting Y value based on time indexs
            Ytrain[:, hour] = df_train.loc[futureIndexTrain, 'Price']
                
        belgium_holidays = holidays.Belgium()

        def is_holiday(date):
            return date in belgium_holidays

        # Adding the holiday dummy variable for Xtrain
        for i, date in enumerate(indexTrain.index.date):
            Xtrain[i, -1] = is_holiday(date)

        # Adding the holiday dummy variable for Xtest
        for i, date in enumerate(indexTest.index.date):
            Xtest[i, -1] = is_holiday(date)

        return Xtrain, Ytrain, Xtest


    def recalibrate_and_forecast_next_day(self, df, calibration_window, next_day_date):
        """Easy-to-use interface for daily recalibration and forecasting of the LEAR model.
        
        The function receives a pandas dataframe and a date. Usually, the data should
        correspond with the date of the next-day when using for daily recalibration.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of historical data containing prices and *N* exogenous inputs. 
            The index of the dataframe should be dates with hourly frequency. The columns 
            should have the following names ``['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']``.
        
        calibration_window : int
            Calibration window (in days) for the LEAR model.
        
        next_day_date : datetime
            Date of the day-ahead.
        
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices.
        """

        # We define the new training dataset and test datasets 
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]
    
        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features. 
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]


        # Generating X,Y pairs for predicting prices
        Xtrain, Ytrain, Xtest, = self._build_and_split_XYs(
            df_train=df_train, df_test=df_test, date_test=next_day_date)

        # Recalibrating the LEAR model and extracting the prediction
        Yp = self.recalibrate_predict(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, next_day_date=next_day_date)

        return Yp 