import warnings
import time
from Community_scale_class import *

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None

    load_file_dir = './Example_files/SRP_Daily_Data_8'
    month_lst = ['8']

    # Summer rate
    rate = [0.077] * 3 + [0.2215] * 6 + [0.077] * 3 + [0.0619] * 6 + [0.077] * 6

    # Winter rate
    # rate = [0.0737] * 5 + [0.0951] * 4 + [0.0737] * 2 + [0.0575] * 6 + [0.0951] * 4 + [0.0737] * 3

    # CO2 factors
    co2 = [0.715, 0.72, 0.715, 0.71, 0.705, 0.69, 0.685, 0.69, 0.71, 0.74, 0.795, 0.87,
           0.87, 0.865, 0.845, 0.81, 0.765, 0.73, 0.7, 0.675, 0.66, 0.66, 0.68, 0.7]

    t_start = time.time()

    case_lst = ['Case1', 'Case2', 'Case3']

    for case_num in case_lst:
        for month in month_lst:
            day_folders = os.listdir(f'{load_file_dir}/{month}')
            t_month = time.time()
            for date in day_folders:
                file_lst = os.listdir(f'{load_file_dir}/{month}/{date}')
                account_lst = [x.split('.')[0] for x in file_lst]  # str list
                iteration = 0
                rev_new = []
                rev_old = []
                old_iter_profits = []
                new_iter_profits = []
                old_sum_cost = []
                new_sum_cost = []
                old_iter_rev = []
                new_iter_rev = []
                old_iter_cost = pd.DataFrame()
                new_iter_cost = pd.DataFrame()
                old_iter_loads = pd.DataFrame()
                new_iter_loads = pd.DataFrame()
                co2_change_lst = []
                dif_lst = []

                df_price = pd.DataFrame(np.array(rate * len(account_lst)).reshape(len(account_lst), 24))
                old_price = df_price.iloc[:len(account_lst), :]

                while iteration <= 5:
                    t_iteration = time.time()
                    t_gurobi = time.time()
                    for account_idx in range(len(account_lst)):
                        try:
                            SRPIndividual(load_file_dir, df_price, case_num, month,
                                          date, iteration, account_lst, account_idx)
                        except:
                            pass
                    t_grid = time.time()
                    print(f'Start grid opt iteration {iteration}')
                    df_price, output_dict = SRPUrbanGrid(month, date, case_num, account_lst, iteration, df_price).grid_main()

                    old_iter_profits.append(output_dict.get('Old profit'))
                    new_iter_profits.append(output_dict.get('New profit'))
                    old_sum_cost.append(sum(output_dict.get('Old cost')))
                    new_sum_cost.append(sum(output_dict.get('New cost')))
                    old_iter_rev.append(output_dict.get('Old rev'))
                    new_iter_rev.append(output_dict.get('New rev'))
                    old_iter_cost = pd.concat([old_iter_cost, pd.Series(output_dict.get('Old cost'))])
                    new_iter_cost = pd.concat([new_iter_cost, pd.Series(output_dict.get('New cost'))])
                    old_iter_loads = pd.concat([old_iter_loads, pd.Series(output_dict.get('Old load'))])
                    new_iter_loads = pd.concat([new_iter_loads, pd.Series(output_dict.get('New load'))])

                    print(f'Finish grid opt iteration {iteration}, time use {round(time.time() - t_grid, 2)} s')

                    print(f'Finish iteration {iteration}, time use {round(time.time() - t_iteration, 2)} s')

                    # Check co2 change
                    co2_change = np.zeros(24)
                    for i in range(24):
                        co2_change[i] = co2[i] * (output_dict.get('New load')[i] - output_dict.get('Old load')[i])
                    sum_change = sum(co2_change)
                    co2_change_lst.append(sum_change)

                    try:
                        dif = abs(new_iter_profits[-1] - new_iter_profits[-2]) / new_iter_profits[-2]
                    except:
                        dif = new_iter_profits[-1]
                    dif_lst.append(dif)
                    if dif <= 0.05:
                        break
                    else:
                        iteration += 1

                df_profit = pd.DataFrame()
                df_profit['old_profit'] = old_iter_profits
                df_profit['new_profit'] = new_iter_profits
                df_profit['old_cost'] = old_sum_cost
                df_profit['new_cost'] = new_sum_cost
                df_profit['old_rev'] = old_iter_rev
                df_profit['new_rev'] = new_iter_rev

                df_profit.to_csv(f'./Community_level/{case_num}/{month}/{date}/Profit_profile.csv', index=False, header=True)

                df_cost = pd.DataFrame()
                df_cost['old_cost'] = old_iter_cost
                df_cost['new_cost'] = new_iter_cost
                df_cost['old_load'] = old_iter_loads
                df_cost['new_load'] = new_iter_loads

                df_cost.to_csv(f'./Community_level/{case_num}/{month}/{date}/Cost_profile.csv', index=False, header=True)

                df_co2 = pd.DataFrame()
                df_co2['co2'] = co2_change_lst

                df_co2.to_csv(f'./Community_level/{case_num}/{month}/{date}/CO2_profile.csv', index=False, header=True)

            print(f'Finish month {month}, time use{round(time.time() - t_month, 2)} s')

        print(f'Finish {case_num}! Time use {round((time.time() - t_start) / 60, 2)} min')
