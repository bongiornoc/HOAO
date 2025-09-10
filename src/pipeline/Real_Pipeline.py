import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pk
import joblib
import time
import scipy.stats as st
from .Sampler import WindowBitsetSampler

def fetch_stock_returns(file_experiment=None,experiment=None,returns_df=None, prices_df=None, split_factor_df=None, rng=None, annualize=False,shuffle=False, shift=1):
    """
    Fetches stock returns from a given experiment file.
    Args:
        file_experiment (str): Path to the experiment file.
        experiment (dict): Experiment configuration dictionary.
        returns_df (pandas.DataFrame): DataFrame containing stock returns data.
        prices_df (pandas.DataFrame): DataFrame containing stock prices data.
        rng (numpy.random.Generator): Random number generator instance.
        annualize (bool, optional): Whether to annualize the in-sample returns. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        shift (int, optional): Number of periods to shift the returns. Defaults to 1.
    Yields:
        tuple: A tuple containing:
            - rin (numpy.ndarray): Returns for the input period.
            - rout (numpy.ndarray): Returns for the output period.
    Raises:
        FileNotFoundError: If the experiment file does not exist.
        KeyError: If required keys are missing in the experiment configuration.
        ValueError: If both file_experiment and experiment are provided, or if neither is provided.
    Note:
        The experiment file is expected to be a pickled dictionary containing the following
        keys:
            - 'selected_stocks': A list of integers representing the indices of the selected stocks.
            - 'selected_days': A list of strings representing the selected trade dates.
            - 'configs': A dictionary containing the following keys:
                - 'dtin': An integer representing the number of in-sample days.
                - 'dtout': An integer representing the number of out-of-sample days.
                - 'data_file': A string representing the path to the data file.
    """
    if prices_df is not None and split_factor_df is None:
        raise ValueError("split_factor must be provided when prices_df is not None.")
    
    if file_experiment is None and experiment is None:
        raise ValueError("Either file_experiment or expriment must be provided.")
    elif file_experiment is not None and experiment is not None:
        raise ValueError("Only one of file_experiment or expriment should be provided.")
    elif file_experiment is not None:
        if not file_experiment.endswith('.pkl'):
            raise ValueError("file_experiment must be a .pkl file.")
        with open(file_experiment, 'rb') as f:
            experiment = pk.load(f)
        
    # Load experiment data
    selected_stock_indices = experiment['selected_stocks']
    selected_trade_dates = pd.to_datetime(experiment['selected_days'])
    
    # Load configs
    n_days = experiment['configs']['dtin']
    n_days_out = experiment['configs']['dtout']
    data_file = experiment['configs']['data_file']
    
    # Load data
    if returns_df is not None:
        dates = returns_df.index
        returns = returns_df.values
        if prices_df is not None:
            prices = prices_df.values
            split_factor = split_factor_df.values
    elif data_file.endswith('.npz'):
        data = np.load(data_file, allow_pickle=True)
        dates = pd.DatetimeIndex(data['dates']).date
        returns = data['returns']
    else:
        AllData = joblib.load(data_file)
        returns = AllData.values
        indices = AllData.index
        dates = pd.DatetimeIndex(indices).date

    if rng is None:
        rng = np.random.default_rng()
    
    # Create date index map
    date_index_map = dict(zip(dates, range(len(dates))))
    # Yield stock returns
    for i in range(len(selected_stock_indices)):
        t_index = date_index_map[  selected_trade_dates[i] ]
        return_in_out = returns[t_index-n_days:t_index+n_days_out+shift, selected_stock_indices[i]]

        if shuffle:
            return_in_out = return_in_out[rng.permutation(n_days+n_days_out+shift)]

        rin, rout = return_in_out[:n_days].T, return_in_out[n_days+shift:].T
        if annualize==True:
            rin *= 252
        if prices_df is not None:
            yesterday_prices = prices[t_index-1, selected_stock_indices[i]] * split_factor[t_index, selected_stock_indices[i]]
            today_prices = prices[t_index, selected_stock_indices[i]]
            yield rin, rout, yesterday_prices, today_prices
        else:
            yield rin, rout

def dataset_experiment(file_experiment=None, returns_df=None, batch_size=1, experiment=None, annualize=True,shuffle=False, n_batches=None, shift=1):
    """
    Creates a batched TensorFlow dataset from stock returns data.
    Args:
        file_experiment (str): Path to the experiment data file.
        returns_df (pandas.DataFrame): DataFrame containing stock returns data.
        batch_size (int): Number of samples per batch.
        experiment (dict): Experiment configuration dictionary.
        shuffle (bool): Whether to shuffle the dataset. Defaults to False.
        annualize (bool, optional): Whether to annualize the in-sample returns. Defaults to True.
        n_batches (int, optional): Number of batches to generate. If None, all data is used.
        shift (int): Number of periods to shift the returns.
    Returns:
        tf.data.Dataset: A dataset yielding batches of stock returns data.
    """
    tf.get_logger().setLevel('ERROR')
    dataset = tf.data.Dataset.from_generator(
        lambda: fetch_stock_returns(file_experiment=file_experiment, returns_df=returns_df, experiment=experiment, annualize=annualize, shuffle=shuffle, shift=shift),
        output_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32)
        )
    ).batch(batch_size)
    
    if n_batches is not None:
        dataset = dataset.take(n_batches)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def generate_real_test_dataset(data_params,validation_steps):
    """
    Generates a real test dataset using the provided data parameters and validation steps.
    Args:
        data_params (dict): A dictionary containing parameters for the data pipeline. 
                            Expected keys include 'batch_size', 'n_stocks', and 'n_days'.
        validation_steps (int): The number of validation steps to generate batches for.
    Returns:
        tf.data.Dataset: A TensorFlow dataset object that yields batches of data for testing.
                            Each batch contains a tuple of (returns_in, cov_out) where:
                            - returns_in: A tensor of shape (batch_size, n_stocks, n_days) representing input returns.
                            - cov_out: A tensor of shape (batch_size, n_stocks, n_stocks) representing output covariances.
    """    
    
    if validation_steps == 0:
        raise ValueError("validation_steps must be greater than 0 to generate a test dataset.")

    data,_ = real_data_pipeline(**data_params)
    
    batches = []
    for _ in range(validation_steps):
        returns_in,cov_out = next(iter(data))
        batches.append((returns_in,cov_out))
    
    dataset_test = tf.data.Dataset.from_generator(lambda: iter(batches), 
                                                   output_signature=(
                                                       tf.TensorSpec(shape=(data_params['batch_size'], 
                                                                            data_params.get('n_stocks'), 
                                                                            data_params.get('n_days')), 
                                                                     dtype=tf.float32),
                                                       tf.TensorSpec(shape=(data_params['batch_size'], 
                                                                            data_params.get('n_stocks'), 
                                                                            data_params.get('n_stocks')), 
                                                                     dtype=tf.float32)
                                                       )
                                                   ).repeat()
    return dataset_test
    

def prepare_dataset(filename_returns, filename_available_stocks, n_days, n_days_out,  datatime_range, annualize=True, 
                    shift=1, calibration=True, top=None, include_sic=False):
    """
    Prepares the dataset for the given parameters.
    Parameters:
    filename_returns (str): Path to the file containing stock returns data.
    filename_available_stocks (str): Path to the file containing available stocks data.
    n_days (int): Number of days to include before the target date.
    n_days_out (int): Number of days to include after the target date.
    datatime_range (tuple): A tuple containing the start and end dates (d0, d1) for the data selection.
    annualize (bool, optional): Whether to annualize the returns. Defaults to True.
    shift (int optional): Number of days to shift the output data. Defaults to 1.
    top (int, optional): Maximum number of stocks to return. Defaults to None.
    Returns:
    tuple: A tuple containing:
        - returns (numpy.ndarray): The array of returns for the selected date range.
        - date_stock_mapping (list): A list of available stocks for each date in the selected date range.
    """    
    start_date, end_date = datatime_range

    # if shift!=1:
    #     raise ValueError("Shift different than 1 is not supported.")

    bundle = joblib.load(filename_returns,mmap_mode='r')
    returns = bundle.returns
    if include_sic:
        sic = bundle.sic.loc[:, returns.columns]
        
    with open(filename_available_stocks, 'rb') as f:
        preprocessed = pk.load(f)
    
    # Filter available days
    available_stocks = preprocessed['available_stocks']

    codes, unique_stocks = pd.factorize(available_stocks.to_numpy().ravel())

    # Rimappiamo i codici alla forma originale del DataFrame
    available_stocks = pd.DataFrame(codes.reshape(available_stocks.shape),
                                    index=pd.to_datetime(available_stocks.index),
                                    columns=available_stocks.columns)

    # Seleziona le colonne di returns in base ai nomi unici ottenuti
    returns = returns.loc[:, unique_stocks]
    if include_sic:
        sic = sic.loc[:, unique_stocks]

    available_stocks = available_stocks.loc[start_date:end_date]
    if calibration==True:
        available_stocks = available_stocks.iloc[:-n_days_out]

    if available_stocks.shape[0] == 0:
        raise ValueError("No data available for the specified date range.")
    
    first_valid_index = returns.index.get_indexer([start_date], method="bfill")[0]
    if first_valid_index < 0:
        raise ValueError("No valid index found for the specified start date.")
    start_cal = returns.index[first_valid_index - n_days]

    if annualize:
        returns *= 252

    if calibration==True:
        returns = returns.loc[start_cal:end_date]
        available_stocks = available_stocks.values.tolist()
        returns = returns.values
        if include_sic:
            sic = sic.loc[start_cal:end_date].values
            return returns, available_stocks, sic
        else:
            return returns, available_stocks, None
    else:
        returns = returns.loc[start_cal:]
        if top is not None:
            available_stocks = available_stocks.iloc[:,:top]
        returns.columns = np.arange(unique_stocks.shape[0])
        available_stocks.columns = [s.replace('permno','index') for s in available_stocks.columns]
        if not include_sic:
            return returns, available_stocks, None
        else:
            sic = sic.loc[start_cal:]
            sic = sic.iloc[:,:unique_stocks.shape[0]]
            sic = sic.values
            if top is not None:
                sic = sic.iloc[:,:top]
            sic.columns = [s.replace('permno','index') for s in sic.columns]
            return returns, available_stocks, sic

def real_data_pipeline(batch_size, n_days, n_days_out, filename_returns, filename_available_stocks ,
                       datatime_range, n_stocks=None, n_stocks_range=None, n_days_range=None, annualize=False, shuffle=False, shift=0,
                       scale_covariance_output=False, target_return=None, sequential=False, rng = np.random.default_rng(), 
                       oos_covariance=True, include_sic=False, return_generator=True, **kwargs):
    """
    Generates a dataset for real data pipeline based on the given parameters.
    Parameters:
    batch_size (int): The size of each batch of data.
    n_days (int): The number of days of historical data to include in each sample.
    n_days_out (int): The number of days to predict.\
    filename_returns (str): Path to the file containing stock returns data.
    filename_available_stocks (str): Path to the file containing available stocks data.
    datatime_range (tuple): A tuple containing the start and end dates for the data.
    n_stocks (int, optional): The number of stocks to include in each sample. Defaults to None.
    n_stocks_range (tuple, optional): A tuple containing the range of stocks to include. Defaults to None.
    annualize (bool, optional): Whether to annualize the returns. Defaults to True.
    shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
    shift (int, optional): The number of days to shift the output data. Defaults to 0.
    scale_covariance_output (bool, optional): Whether to scale the covariance output. Defaults to False.
    **kwargs: Additional keyword arguments.
    Returns:
    tuple: A tuple containing the dataset and an empty list.
    """


    historicalData, available_stocks, sic = prepare_dataset(filename_returns=filename_returns, filename_available_stocks=filename_available_stocks, n_days=n_days, n_days_out=n_days_out,
                                                       datatime_range=datatime_range, annualize=annualize, shift=shift, include_sic=include_sic)
    
    
    dataset = real_data_producer(batch_size=batch_size, n_days_in=n_days, n_days_out=n_days_out, historicalData=historicalData,
                                 available_stocks=available_stocks, n_stocks=n_stocks, n_stocks_range=n_stocks_range, 
                                 rng=rng, shuffle=shuffle, shift=shift, scale_covariance_output=scale_covariance_output, 
                                 target_return=target_return, n_days_range=n_days_range, sequential=sequential, oos_covariance=oos_covariance,sic=sic, 
                                 return_generator=return_generator, annualized=annualize)
    
    return dataset
    
    
def real_data_producer(batch_size, n_days_in, n_days_out, historicalData, available_stocks, rng=None,
                        n_stocks=None, n_stocks_range=None, shuffle=False, shift=0, scale_covariance_output=False,
                        target_return=None, n_days_range=None, sequential=False, oos_covariance=True, sic=None, return_generator=False, annualized=True):
    """
    Generates a TensorFlow dataset that produces batches of real stock data for training models.
    Args:
        batch_size (int): The number of samples per batch.
        n_days_in (int): The number of days of historical data to use as input.
        n_days_out (int): The number of days of future data to use as output.
        historicalData (np.ndarray): A 3D array of historical stock data with shape (time, stocks, features).
        available_stocks (dict): A dictionary where keys are time indices and values are lists of available stock indices at those times.
        rng (np.random.Generator, optional): A random number generator instance. If None, a default generator is used.
        n_stocks (int, optional): The number of stocks to select. If None, n_stocks_range must be provided.
        n_stocks_range (tuple, optional): A tuple specifying the range (min, max) for the number of stocks to select. If None, n_stocks must be provided.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        shift (int, optional): The number of days to shift the output data. Defaults to 0.
        scale_covariance_output (bool, optional): Whether to scale the covariance output. Defaults to False.
        target_return (float, optional): A target return value to subtract from the output returns. If None, the mean return is subtracted.
        n_days_range (tuple, optional): A tuple specifying the range (min, max) for the number of days to include in the input data. If None, all n_days_in are used.
        sequential (bool, optional): If True, the selected timesteps in a batch will be sequential. If False, they will be randomly selected.
    Raises:
        ValueError: If both n_stocks and n_stocks_range are None.
    Returns:
        tf.data.Dataset: A TensorFlow dataset that yields tuples of (returns_in, cov_out), where:
            - returns_in (tf.Tensor): A tensor of shape (n_stocks, n_days_in) containing the input returns.
            - cov_out (tf.Tensor): A tensor of shape (n_stocks, n_stocks) containing the covariance matrix of the output returns.
    """
    
    if n_stocks is None and n_stocks_range is None:
        raise ValueError("Both n_stocks and n_stocks_range cannot be None")
    
    if target_return is not None and scale_covariance_output==True:
        raise ValueError("target_return and scale_covariance_output cannot be used together.")
    
    if sequential and n_days_out > 1:
        raise ValueError("Sequential mode is not supported for n_days_out > 1. Please set sequential=False.")
    
    if rng is None:
        rng = np.random.default_rng()
        
    if sequential:
        sampler = WindowBitsetSampler(available_stocks, use_numba=True, warmup=True)  # Set up the rng 
        
            
    def data_generator():

        time_offsets = np.arange(0 ,n_days_in+n_days_out+shift).reshape(1, 1, n_days_in + n_days_out + shift)

        while True:
            if n_stocks_range is not None:
                n_stocks_local = rng.integers(*n_stocks_range, size=1)
            else:
                n_stocks_local = n_stocks
            
            if sequential:
                selected_timesteps = np.arange(batch_size) + rng.integers(0, len(available_stocks) - batch_size)
                selected_stocks = sampler.sample_chain_window(start=selected_timesteps[0], k=batch_size, m=n_stocks_local)

            else:
                selected_timesteps = rng.integers(0,len(available_stocks), size=batch_size)
                selected_stocks = np.array([
                    rng.choice(available_stocks[t], n_stocks_local, replace=False)
                    for t in selected_timesteps
                ])
            
            if sic is not None:
                selected_sics = sic[selected_timesteps]
                
            time_indices = time_offsets + selected_timesteps[:, None, None]
            
            if isinstance(historicalData, np.ndarray):
                returns_inout = historicalData[time_indices, selected_stocks[:, :, None]]
            else:
                change_dict = {col:indx for indx,col in enumerate(historicalData.columns)}
                mapped_selected_stocks = np.array([[change_dict[s] for s in row] for row in selected_stocks])
                values = historicalData.to_numpy()
                returns_inout = values[time_indices, mapped_selected_stocks[:, :, None]]
            
            if shuffle:
                returns_inout = returns_inout.sample(frac=1, random_state=rng)

            returns_in, returns_out = returns_inout[:, :, :n_days_in], returns_inout[:, :, n_days_in + shift:]
            
            if n_days_range is not None:
                dt = rng.integers(n_days_range[0], n_days_range[1] + 1)
                returns_in = returns_in[:, :, -dt:]

            if n_days_out == 1 or oos_covariance==False:
                if sic is not None:
                    selected_sics = np.take_along_axis(selected_sics, selected_stocks, axis=1)
                    yield (returns_in, selected_sics.astype(np.int32)), returns_out/252,
                else:
                    if annualized:
                        yield returns_in, returns_out/252
                    else:
                        yield returns_in, returns_out
                continue
            
            if target_return is None:
                returns_out -= returns_out.mean(axis=2, keepdims=True)
            else:
                returns_out -= target_return

            cov_out = returns_out @ returns_out.transpose(0, 2, 1) / returns_out.shape[-1] 

            if scale_covariance_output:
                r_in_demode = returns_in - returns_in.mean(axis=2, keepdims=True)
                cov_in = r_in_demode @ r_in_demode.transpose(0, 2, 1) / n_days_in
                cost = np.linalg.inv(cov_in).sum(axis=(-1, -2), keepdims=True)
                cov_out *= cost
            if sic is not None:
                selected_sics = np.take_along_axis(selected_sics, selected_stocks, axis=1)
                yield (returns_in, selected_sics.astype(np.int32)), cov_out
            else:
                yield returns_in, cov_out

    if return_generator:
        return data_generator   
    batch_day = None if n_days_in is not None else n_days_in
    batch_out = n_stocks if n_days_out>1 and oos_covariance==True else n_days_out
    if sic is None:
        # Create a dataset from the generator function
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=[batch_size,n_stocks, batch_day], dtype=tf.float32),
                tf.TensorSpec(shape=[batch_size,n_stocks, batch_out], dtype=tf.float32)
            )
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                (tf.TensorSpec(shape=[batch_size, n_stocks, batch_day], dtype=tf.float32),
                 tf.TensorSpec(shape=[batch_size, n_stocks], dtype=tf.int32)),
                tf.TensorSpec(shape=[batch_size, n_stocks, batch_out], dtype=tf.float32)
            )
        )

    # Prefetch the dataset to improve performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
