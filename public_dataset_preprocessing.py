import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import pickle


# Assign datatime object to raw data
def assign_pd_time(x_1, x_2):
    day = x_1
    time_str = x_2
    if day == 1:
        date_str = "2020-08-01 " + str(time_str)
    elif day == 2:
        date_str = "2020-08-02 " + str(time_str)
    else:
        date_str = "2020-08-02 " + str(time_str)
    return pd.to_datetime(date_str)


# Use the rmssd as the HRV value
def get_subject_hrv_rmssd_pd(rr_pd_raw_input):
    #
    rr_pd = rr_pd_raw_input.copy()

    del rr_pd['Unnamed: 0']  # delete the first column

    rr_pd["time_pd"] = rr_pd.apply(lambda x: assign_pd_time(x["day"], x["time"]), axis=1)

    df_rr = rr_pd.copy()
    df_rr['ibi_s'] = [x if x < 2.1 else np.nan for x in df_rr['ibi_s']]
    df_rr['ibi_s'] = [x if x > 0.3 else np.nan for x in df_rr['ibi_s']]
    df_rr = df_rr.dropna()

    df_rr_5min_list = [g for n, g in df_rr.groupby(pd.Grouper(key='time_pd', freq='5min'))]

    t_list = []
    rmssd_list = []
    sdnn_list = []

    for t_i, week_pd in enumerate(df_rr_5min_list):
        if week_pd.empty:
            pass
        else:
            nn_intervals = list(1000 * week_pd['ibi_s'].dropna().values)
            diff_nni = np.diff(nn_intervals)
            rmssd = np.sqrt(np.mean(diff_nni ** 2))
            sdnn = np.std(nn_intervals, ddof=1)
            time_pd_datatime = week_pd["time_pd"].mean()
            t_list.append(time_pd_datatime)
            rmssd_list.append((rmssd))
            sdnn_list.append((sdnn))

    hrv_df = pd.DataFrame(list(zip(t_list, sdnn_list)), columns=["time_pd", "value"])
    hrv_df["log_value"] = np.log(hrv_df["value"])
    return hrv_df


# functional for lambda
def get_time_delta_hour(x):
    t_d = (x - pd.Timestamp("2020-08-01 00:00:00"))
    hour_sec = t_d.seconds / 3600
    hour_day = t_d.days * 24
    return hour_sec + hour_day


# sleep data
def get_subject_sleep_hour(sleep_pd):
    return sleep_pd["Total Minutes in Bed"].sum() / 60.0


def assign_pd_time_activity(start, end):
    return pd.to_timedelta(end + ":00") - pd.to_timedelta(start + ":00")


def get_activity_value(activity_pd_input):
    activity_pd = activity_pd_input.copy().dropna()
    del activity_pd["Unnamed: 0"]
    activity_pd["time_start_pd"] = activity_pd.apply(lambda x: assign_pd_time_activity(x["Start"], x["End"]), axis=1)
    activity_pd["time_end_pd"] = activity_pd.apply(lambda x: assign_pd_time_activity(x["Start"], x["End"]), axis=1)
    activity_pd["time_last_hour"] = activity_pd["time_end_pd"].apply(lambda x : x.seconds / 3600)
    activity_pd_345 = activity_pd.loc[activity_pd["Activity"].isin([4, 5])]
    if activity_pd_345.empty:
        return 0
    else:
        return activity_pd_345["time_last_hour"].sum()


if __name__ == "__main__":
    data_subject_dict = {}

    for i in range(1, 23):
        print()
        print("i =", i)
        user_folder = "../MMASH/DataPaper/user_" + str(i)

        # sleep
        sleep_file = user_folder + "/sleep.csv"
        sleep_pd = pd.read_csv(sleep_file, header=0)
        sleep_value = get_subject_sleep_hour(sleep_pd)

        # exercise
        activity_file = user_folder + "/Activity.csv"
        activity_pd = pd.read_csv(activity_file, header=0)
        activity_value = get_activity_value(activity_pd)

        # demographics
        demographics_file = user_folder + "/user_info.csv"
        dem_pd = pd.read_csv(demographics_file, header=0)
        age_height_weight_values = dem_pd[["Age", "Height", "Weight"]].loc[0].values
        age_height_weight = np.array([age_height_weight_values[0],
                                      age_height_weight_values[1] / 100,
                                      age_height_weight_values[2]])
        print("age_height_weight =", age_height_weight)

        # stress
        qtn_file = user_folder + "/questionnaire.csv"
        qtn_pd = pd.read_csv(qtn_file, header=0)
        stress_value = qtn_pd["Daily_stress"].loc[0]

        # HRV
        rr_file = user_folder + "/RR.csv"
        rr_pd = pd.read_csv(rr_file, header=0)
        subject_hrv_pd = get_subject_hrv_rmssd_pd(rr_pd)

        low_noise_index = subject_hrv_pd["value"] < 55
        low_noise_num = len(subject_hrv_pd[low_noise_index].index)
        low_noise_df = subject_hrv_pd[low_noise_index].sample(int(0.85 * low_noise_num))
        subject_hrv_pd = subject_hrv_pd[~subject_hrv_pd.index.isin(low_noise_df.index)]
        z_score_msk = (np.abs(stats.zscore(subject_hrv_pd["value"])) < 2.5)
        subject_hrv_pd = subject_hrv_pd.loc[z_score_msk]


        subject_hrv_pd = subject_hrv_pd.loc[subject_hrv_pd["value"] > 12]
        subject_hrv_pd["t_hour"] = subject_hrv_pd["time_pd"].apply(lambda x: get_time_delta_hour(x))

        s_vector = np.hstack([age_height_weight, activity_value, sleep_value, stress_value])
        t_raw = subject_hrv_pd["t_hour"].values
        y_raw = subject_hrv_pd["value"].values
        y_mean = subject_hrv_pd["value"].mean()
        print("y_mean =", y_mean)
        data_subject_dict[i] = [t_raw, y_raw, s_vector]

        print("sleep_value =", sleep_value)
        print("activity_value =", activity_value)
        print("age_height_weight =", age_height_weight)
        print("stress_value =", stress_value)
        subject_hrv_pd.plot(x="t_hour", y="value", style=".")
        plt.xlim(8, 36)
        plt.title(i)
        plt.show()
        print()

    pickle_name = "data_and_pickle/public_data_for_MTL.pkl"
    file_open = open(pickle_name, "wb")
    pickle.dump(data_subject_dict, file_open)
    file_open.close()