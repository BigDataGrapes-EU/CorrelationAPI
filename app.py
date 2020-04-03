import os
import time
import zipfile

from flask import Flask, send_file
import datetime
import scipy
import numpy as np
import pandas as pd
import datetime
from dateutil import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
import inspect

app = Flask(__name__)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


@app.route('/symbeeosis/correlate')
def symbeeosis():
    import scipy
    import numpy as np
    import pandas as pd
    import datetime
    from dateutil import relativedelta
    import seaborn as sns
    import matplotlib.pyplot as plt
    import inspect

    def get_maceration_property_dict():
        maceration_property_dict = {}

        for prop in list_of_maceration_properties:
            if prop == 'Sample':
                continue
            maceration_property_dict[prop] = []
            for parcel_id in list_of_parcel_ids:
                sample_id = int(parcel_to_sample[str(parcel_id)])
                df_parcel_maceration = df_maceration[df_maceration['Sample'] == sample_id]
                maceration_property_dict[prop].append(float(df_parcel_maceration[prop]))

        # print(maceration_property_dict)
        return maceration_property_dict

    def get_product_type_signals_dict(agg_value_column, agg_time_function, satellite_source):
        product_type_signals_dict = {}

        for product_type in list_of_product_types:
            if product_type == 'variations':
                continue

            product_type_signals_dict[product_type] = []
            for parcel_id in list_of_parcel_ids:
                # properties filter
                df_parcels_filtered = df_parcels[
                    (df_parcels.parcel_id == parcel_id) & (df_parcels.product_type == product_type) & (
                            df_parcels.source == satellite_source)]
                # exclude non
                df_parcels_filtered = df_parcels_filtered[df_parcels_filtered[agg_value_column].notna()]
                # getting the start and the end dates
                df_parcels_filtered = agg_time_function([parcel_id, df_parcels_filtered, agg_value_column])
                # series
                se_product_type_agg_value = df_parcels_filtered[agg_value_column]
                se_product_type_agg_value = pd.to_numeric(pd.Series(se_product_type_agg_value), errors='coerce')
                product_type_signals_dict[product_type].append(se_product_type_agg_value.mean())

        return product_type_signals_dict

    def fn_correlation_matrix(agg_value_column, agg_time_function, satellite_source, verbose=False):
        product_type_signals_dict = get_product_type_signals_dict(agg_value_column, agg_time_function, satellite_source)
        maceration_property_dict = get_maceration_property_dict()

        correlation_matrix = np.ones((len(maceration_property_dict.keys()), len(product_type_signals_dict.keys())),
                                     dtype=np.float64)

        for i, observed_variable in enumerate(maceration_property_dict.keys()):
            for j, current_signal in enumerate(product_type_signals_dict.keys()):
                np_observed_array = np.array(maceration_property_dict[observed_variable])
                np_signal_array = np.array(product_type_signals_dict[current_signal])

                # np.corrcoef: returns pearson product-moment correlation coefficients.
                current_correlation = np.corrcoef(np_observed_array, np_signal_array)[0, 1]
                correlation_matrix[i][j] = current_correlation
                if verbose:
                    print("correlation between {} and {} {}: {}".format(observed_variable, current_signal,
                                                                        agg_value_column,
                                                                        current_correlation))

        y_labels = ['TFC', 'ABTS', 'Refractive', 'TPC', 'pH', 'DPPH']  # maceration_property_dict.keys()
        x_labels = product_type_signals_dict.keys()

        return correlation_matrix, x_labels, y_labels

    def plot_heat_map(correlation_matrix, x_labels, y_labels, title, plot_shape, idx_plot):
        plt.subplot(plot_shape[0], plot_shape[1], idx_plot)
        ax = sns.heatmap(correlation_matrix, linewidth=0.5, center=0, cmap="RdBu")
        ax.set_title(title)
        ax.set_yticklabels(y_labels, rotation=0, fontsize="10", va="center")
        ax.set_xticklabels(x_labels, rotation=0, fontsize="10", va="center")

    def get_the_most_correlated_variables(correlation_matrix, x_labels, y_labels):
        abs_matrix = np.abs(correlation_matrix)
        max_corr_idx = np.argmax(abs_matrix, axis=1)
        results = []

        for i, idx_max in enumerate(max_corr_idx):
            item = (y_labels[i], x_labels[idx_max], correlation_matrix[i][idx_max], abs_matrix[i][idx_max])
            results.append(item)

        df = pd.DataFrame(results,
                          columns=['Maceration', 'Product_Type', 'Highest_Corr_Value', 'Highest_Corr_Value_Abs'])
        return df

    def interval_from_30_days_before_crop(params_list, verbose=False):
        parcel_id = params_list[0]
        df_timeseries = params_list[1]

        days_before_the_crop = 30
        date_crop = parcel_date_crop_dict[parcel_id]
        date_end = date_crop
        date_start = date_end - datetime.timedelta(days=days_before_the_crop)

        # temporal filter
        df_filtered = df_timeseries[(df_timeseries.date >= date_start) & (df_timeseries.date <= date_end)]

        if verbose:
            print("parcel_id {} start {} end {} crop {} len {}".format(parcel_id, date_start.strftime("%d-%m-%Y"),
                                                                       date_end.strftime("%d-%m-%Y"),
                                                                       date_crop.strftime("%d-%m-%Y"),
                                                                       len(df_filtered)))

        return df_filtered

    # From 1/1/XXXX to the sample date in the same XXXX year (different for each y)
    def interval_from_the_begging_of_the_year_until_crop_date(params_list, verbose=False):
        parcel_id = params_list[0]
        df_timeseries = params_list[1]

        date_crop = parcel_date_crop_dict[parcel_id]
        date_end = date_crop
        date_start = datetime.datetime(date_end.year, month=1, day=1)

        # temporal filter
        df_filtered = df_timeseries[(df_timeseries.date >= date_start) & (df_timeseries.date <= date_end)]

        if verbose:
            print("parcel_id {} start {} end {} crop {} len {}".format(parcel_id, date_start.strftime("%d-%m-%Y"),
                                                                       date_end.strftime("%d-%m-%Y"),
                                                                       date_crop.strftime("%d-%m-%Y"),
                                                                       len(df_filtered)))

        return df_filtered

    # From 1/1/XXXX to the min of samples dates in the same year (the same for each y)
    def interval_from_the_begging_of_the_year_until_minvalue_observation_date(params_list, verbose=False):
        parcel_id = params_list[0]
        df_timeseries = params_list[1]
        agg_value_column = params_list[2]

        date_crop = parcel_date_crop_dict[parcel_id]
        date_first_of_the_year = datetime.datetime(date_crop.year, month=1, day=1)
        date_start = date_first_of_the_year

        df_filtered = df_timeseries[(df_timeseries.date >= date_start) & (df_timeseries.date < date_crop)]
        df_filtered = df_filtered.reset_index(drop=True)

        idx_min = df_filtered[agg_value_column].idxmin()
        date_end = df_filtered.iloc[idx_min]['date']

        df_filtered = df_filtered[(df_filtered.date <= date_end)]
        assert len(df_filtered) > 0, 'The interval resulted in a empty number of observation'

        if verbose:
            print("parcel_id {} start {} end {} crop {} len {}".format(parcel_id, date_start.strftime("%d-%m-%Y"),
                                                                       date_end.strftime("%d-%m-%Y"),
                                                                       date_crop.strftime("%d-%m-%Y"),
                                                                       len(df_filtered)))

        return df_filtered

    # From the time we identify the min value of y since the beginning of the sample season
    def interval_from_minvalue_observation_date_until_the_crop_date(params_list, verbose=False):
        parcel_id = params_list[0]
        df_timeseries = params_list[1]
        agg_value_column = params_list[2]

        date_crop = parcel_date_crop_dict[parcel_id]
        date_first_of_the_year = datetime.datetime(date_crop.year, month=1, day=1)

        df_filtered = df_timeseries[(df_timeseries.date >= date_first_of_the_year) & (df_timeseries.date < date_crop)]
        df_filtered = df_filtered.reset_index(drop=True)

        idx_min = df_filtered[agg_value_column].idxmin()
        date_start = df_filtered.iloc[idx_min]['date']
        date_end = date_crop
        df_filtered = df_filtered[(df_filtered.date >= date_start) & (df_filtered.date <= date_end)]
        assert len(
            df_filtered) > 0, 'The interval between {} and {} resulted in a empty number of observation for the parcel_id {}'.format(
            date_start.strftime("%d-%m-%Y"), date_end.strftime("%d-%m-%Y"), parcel_id)

        if verbose:
            print("parcel_id {} start {} end {} crop {} len {}".format(parcel_id, date_start.strftime("%d-%m-%Y"),
                                                                       date_end.strftime("%d-%m-%Y"),
                                                                       date_crop.strftime("%d-%m-%Y"),
                                                                       len(df_filtered)))

        return df_filtered

    df_samples = pd.read_csv("static/symbeeosis/samples.csv", sep=';', parse_dates=['Sample collection day'])

    # df_maceration = pd.read_csv('maceration.csv', sep=';')
    df_maceration = pd.read_csv('static/symbeeosis/ultrasound.csv', sep=';')

    df_maceration.drop("Total microbial count", axis=1, inplace=True)
    df_maceration.drop("Toxicity on skin cells (MTT assay)", axis=1, inplace=True)
    df_maceration.drop("Gene expression (SIRT1) on skin cells", axis=1, inplace=True)
    df_maceration.drop("Yeasts and moulds", axis=1, inplace=True)

    list_of_maceration_properties = df_maceration.columns

    # read timeseries
    df_parcels = pd.read_csv('static/symbeeosis/symbeeosis_timeseries_data_api.csv', sep=';', parse_dates=['date'])

    def adjust_dataframe_float_types(df, column_name):
        df[column_name].replace('None', np.nan, inplace=True)
        return df[column_name].astype('float64')

    for col_name in ['max_value', 'min_value', 'mean_value', 'std_value', 'count_value', 'sum_value']:
        df_parcels[col_name] = adjust_dataframe_float_types(df_parcels, col_name)

    # df_parcels['month'] = df_parcels.date.apply(lambda x: x.month)
    # df_parcels['year'] = df_parcels.date.apply(lambda x: x.year)
    # df_parcels["year/month"] = df_parcels["year"].map(str) + '-' + df_parcels["month"].map(str)

    print("df_dtypes", df_parcels.dtypes)

    df_parcels = df_parcels[df_parcels.parcel_id != 135975]
    df_parcels = df_parcels[df_parcels.parcel_id != 135974]

    list_of_parcel_ids = df_parcels.parcel_id.unique()
    list_of_product_types = df_parcels.product_type.unique()

    print(list_of_parcel_ids)
    print(list_of_product_types)

    sample_to_parcel = {}
    parcel_to_sample = {}

    file_sample = open("static/symbeeosis/samples.csv", 'r')
    file_sample.readline()

    for item in file_sample:
        fields = item.strip().split(';')
        sample_to_parcel[fields[0]] = fields[2]
        parcel_to_sample[fields[2]] = fields[0]

    file_sample.close()

    print("sample_to_parcel", sample_to_parcel)
    print("parcel_to_sample", parcel_to_sample)

    parcel_date_crop_dict = {}

    for parcel_id in list_of_parcel_ids:
        date_crop = df_samples[df_samples["Parcel ID"] == parcel_id]["Sample collection day"].max()
        date_crop = datetime.datetime(date_crop.year, date_crop.month, date_crop.day)
        parcel_date_crop_dict[parcel_id] = date_crop

    plt.figure(figsize=(19, 9))

    ### >>>>> @parameters
    list_of_agg_value_columns = ['max_value', 'min_value', 'mean_value', 'std_value', 'count_value', 'sum_value']
    list_of_agg_time_functions = [interval_from_30_days_before_crop,
                                  interval_from_the_begging_of_the_year_until_crop_date
        , interval_from_the_begging_of_the_year_until_minvalue_observation_date,
                                  interval_from_minvalue_observation_date_until_the_crop_date]

    list_of_agg_time_functions = [interval_from_minvalue_observation_date_until_the_crop_date]

    satellite_source = 'sentinel2'  # ['sentinel2', 'landsat8']

    idx_plot = 1
    for agg_value_column in list_of_agg_value_columns:
        for agg_time_function in list_of_agg_time_functions:
            correlation_matrix, x_labels, y_labels = fn_correlation_matrix(agg_value_column, agg_time_function,
                                                                           satellite_source, verbose=False)
            # plot
            plot_title = "Agg.: {}. Intrv: {}".format(agg_value_column,
                                                      agg_time_function.__name__.replace("interval_", ""))
            plot_title = plot_title.replace("from_the_begging_of_the_year", "from_01/X/X")
            plot_title = plot_title.replace("observation_", "")
            plot_heat_map(correlation_matrix, x_labels, y_labels, plot_title, (2, 3), idx_plot)
            idx_plot = idx_plot + 1

    plt.savefig('plots/heatmap_0.png')
    plt.show()

    plt.figure(figsize=(35, 28))

    ### >>>>> @parameters
    list_of_agg_value_columns = ['max_value', 'min_value', 'mean_value', 'std_value', 'count_value', 'sum_value']
    list_of_agg_time_functions = [interval_from_30_days_before_crop,
                                  interval_from_the_begging_of_the_year_until_crop_date
        , interval_from_the_begging_of_the_year_until_minvalue_observation_date,
                                  interval_from_minvalue_observation_date_until_the_crop_date]
    satellite_source = 'sentinel2'  # ['sentinel2', 'landsat8']

    idx_plot = 1
    for agg_value_column in list_of_agg_value_columns:
        for agg_time_function in list_of_agg_time_functions:
            correlation_matrix, x_labels, y_labels = fn_correlation_matrix(agg_value_column, agg_time_function,
                                                                           satellite_source, verbose=False)
            # plot
            plot_title = "Agg.: {}. Intrv: {}".format(agg_value_column,
                                                      agg_time_function.__name__.replace("interval_", ""))
            plot_title = plot_title.replace("from_the_begging_of_the_year", "from_01/XX/XXXX")
            plot_heat_map(correlation_matrix, x_labels, y_labels, plot_title,
                          (len(list_of_agg_value_columns), len(list_of_agg_time_functions)), idx_plot)
            idx_plot = idx_plot + 1

    plt.savefig('plots/heatmap_1.png')
    plt.show()

    import seaborn as sns

    sns.set(style="whitegrid")

    product_types = ['ndvi', 'vitality']
    satellite_source = 'sentinel2'
    agg_value_column = 'mean_value'
    list_of_agg_time_functions = [interval_from_30_days_before_crop,
                                  interval_from_the_begging_of_the_year_until_crop_date
        , interval_from_the_begging_of_the_year_until_minvalue_observation_date,
                                  interval_from_minvalue_observation_date_until_the_crop_date]

    plt.figure(figsize=(35, 12))

    idx_plot = 1

    for product_type in product_types:
        for agg_time_function in list_of_agg_time_functions:
            list_of_dfs = []

            df_observations = df_parcels[
                (df_parcels.product_type == product_type) & (df_parcels.source == satellite_source)]
            df_observations = df_observations[df_observations[agg_value_column].notna()]

            len_before_drop_duplication = len(df_observations)
            df_observations = df_observations.drop_duplicates(subset=['parcel_id', 'date'])
            len_after_drop_duplication = len(df_observations)

            if len_before_drop_duplication < len_after_drop_duplication:
                print("There are duplicated register in the dataset.")

            print("{} {}".format(product_type, agg_time_function.__name__))
            for parcel_id in list_of_parcel_ids:
                df_observations_parcel = df_observations[df_observations.parcel_id == parcel_id]
                df = agg_time_function([parcel_id, df_observations_parcel, agg_value_column], verbose=True)
                list_of_dfs.append(df)

            df_result = pd.concat(list_of_dfs)

            value_counts = df_result.parcel_id.value_counts()
            df_val_counts = pd.DataFrame(value_counts)
            df_val_counts = df_val_counts.reset_index()
            df_val_counts.columns = ['parcel_id', 'number of observations']

            plt.subplot(len(product_types), len(list_of_agg_time_functions), idx_plot)
            ax = sns.barplot(x="parcel_id", y="number of observations", data=df_val_counts)
            plot_title = "{}".format(agg_time_function.__name__)
            ax.set_title(plot_title)
            idx_plot = idx_plot + 1

    plt.savefig('plots/obs_hist.png')
    plt.show()

    list_of_agg_value_columns = ['max_value', 'min_value', 'mean_value', 'std_value', 'count_value', 'sum_value']
    satellite_source = 'sentinel2'
    list_of_agg_time_functions = [interval_from_30_days_before_crop,
                                  interval_from_the_begging_of_the_year_until_crop_date
        , interval_from_the_begging_of_the_year_until_minvalue_observation_date,
                                  interval_from_minvalue_observation_date_until_the_crop_date]

    df_highest_corr = None

    for agg_value_column in list_of_agg_value_columns:
        for agg_time_function in list_of_agg_time_functions:
            correlation_matrix, x_labels, y_labels = fn_correlation_matrix(agg_value_column, agg_time_function,
                                                                           satellite_source,
                                                                           verbose=False)  # fn_correlation_matrix(agg_function, satellite_source, days_before_the_crop, verbose=False)
            df_result = get_the_most_correlated_variables(correlation_matrix, list(x_labels), y_labels)
            df_result['Agg_value_column'] = agg_value_column
            df_result['Agg_time_function'] = agg_time_function.__name__
            df_result['Source'] = satellite_source

            if df_highest_corr is None:
                df_highest_corr = df_result
            else:
                df_highest_corr = pd.concat([df_highest_corr, df_result])

    pd.set_option('max_colwidth', 800)

    df_highest_corr_by_Maceration = df_highest_corr.sort_values(by=['Maceration', 'Highest_Corr_Value_Abs'],
                                                                ascending=False)
    df_highest_corr_by_Maceration = df_highest_corr_by_Maceration.drop_duplicates(subset=['Maceration'], keep='first')
    df_highest_corr_by_Maceration.rename(columns={'Product_Type': 'Most_correlated_Product_Type'}, inplace=True)
    print(df_highest_corr_by_Maceration[
              ['Maceration', 'Most_correlated_Product_Type', 'Highest_Corr_Value', 'Highest_Corr_Value_Abs',
               'Agg_time_function', 'Agg_value_column', 'Source']])

    pd.set_option('max_colwidth', 800)

    df_highest_corr_by_Satellity_Index = df_highest_corr.sort_values(by=['Product_Type', 'Highest_Corr_Value_Abs'],
                                                                     ascending=False)
    df_highest_corr_by_Satellity_Index = df_highest_corr_by_Satellity_Index.drop_duplicates(subset=['Product_Type'],
                                                                                            keep='first')
    df_highest_corr_by_Satellity_Index.rename(columns={'Maceration': 'Most_correlated_Maceration_Prop'}, inplace=True)
    print(df_highest_corr_by_Satellity_Index[
              ['Product_Type', 'Most_correlated_Maceration_Prop', 'Highest_Corr_Value', 'Highest_Corr_Value_Abs',
               'Agg_time_function', 'Agg_value_column', 'Source']])

    filename = str(time.time()) + '.zip'
    zipf = zipfile.ZipFile('correlations/' + filename, 'w', zipfile.ZIP_DEFLATED)
    zipdir('plots/', zipf)
    zipf.close()

    return send_file('correlations/' + filename,
                     attachment_filename=filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
