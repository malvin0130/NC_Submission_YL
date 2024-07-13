import pandas as pd
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib
import os
import cvxpy as cp
import warnings
import multiprocessing as mp


def occ_schedule(df_day):
    df_day.index = pd.to_datetime(df_day.iloc[:, 0])
    try:
        occ_ = df_day[df_day['occ_new'] == 1]
    except:
        occ_ = df_day[df_day['occ'] == 1]

    occ_['grp'] = (occ_.index.to_series().diff().dt.total_seconds() / 3600).ne(1).cumsum()
    counts = []
    for grp in occ_['grp'].unique():
        items = occ_[occ_['grp'] == grp]
        count = len(items)
        counts.append(count)

    grp_num = counts.index(max(counts)) + 1
    occ_final = occ_[occ_['grp'] == grp_num]

    if len(occ_final) <= 19:
        occ_range = len(occ_final)
        start_index, end_index = occ_final.index.values[0], occ_final.index.values[-1]

    if len(occ_final) > 19:
        occ_range = 19
        start_index, end_index = df_day.index.values[0], df_day.index.values[18]

    # df_stat = pd.DataFrame({'day': day, 'start_index': start_index, 'end_index': end_index}, index=[0])
    # df_stat.to_csv(f'./29_7.4_7.9_temp/{account_id}/{account_id}_week{week}_day{day}_occ_stat.csv', index=False, header=True)

    return occ_range, start_index, end_index


def data_prepare(df_day, rate):
    # df_day = df.iloc[day * 24:day * 24 + 24]
    df_day['Price'] = rate

    rev_measured = np.zeros(24)
    rev_old = np.zeros(24)
    # rev_shift = np.zeros(24)

    measured_load = df_day['Load'].to_list()
    pred_result = df_day['Base_load'].to_list()
    # shift_load = df_day['Shift'].to_list()

    for i in range(len(df_day)):
        rev_measured[i] = rate[i] * measured_load[i]
        rev_old[i] = rate[i] * pred_result[i]
        # rev_shift[i] = rate[i] * shift_load[i]

    df_day['rev_measured'] = rev_measured
    df_day['rev_old'] = rev_old
    # df_day['rev_shift'] = rev_shift

    return df_day


def gurobi_individual_model(df_day, occ_range, start_index, end_index, account_id, month, day_folder, price_, iteration, co2):
    occ_day = pd.DataFrame()
    occ_day['occ'] = np.ones(int(occ_range))
    occ_day.index = pd.date_range(start_index, end_index, freq='H')
    df_day['occ_temp'] = np.zeros(len(df_day))
    df_day.loc[occ_day.index, 'occ_temp'] = occ_day['occ']

    I = range(24)
    L_predict = df_day['Base_load'].tolist()
    ev_load = df_day['Load'].tolist()
    occ = df_day['occ_temp'].tolist()
    ev_price = df_day['Price'].tolist()
    # rev_min = df_day['rev_shift'][start_index:end_index].tolist()
    # rate_old = df_day['Price'][start_index:end_index].tolist()
    bill_old = sum(pd.Series(ev_load)*pd.Series(ev_price))
    df_day['price_outcome'] = price_
    rate_old = df_day['price_outcome'].tolist()
    miu = 0.9
    peak_load = max(ev_load)
    min_load = min(ev_load)

    # Calculate bat init
    mask_temp = (df_day['charging'] == 1) & (df_day['occ'] == 1)
    df_charging = df_day.loc[mask_temp]
    try:  # Charging happened
        bat_use = df_charging.dif.sum()
        if bat_use < 48:
            bat_init = 60*0.9 - bat_use
        else:
            bat_init = 0
    except:  # No charging happened
        bat_init = 60*0.9

    # Create model
    m = gp.Model()

    # Add variables
    a = [m.addVar(vtype=GRB.BINARY, name='a({})'.format(i)) for i in I]
    b = [m.addVar(name='b({})'.format(i)) for i in I]
    c = [m.addVar(name='c({})'.format(i)) for i in I]
    p = [m.addVar(name='p({})'.format(i)) for i in I]
    bat = [m.addVar(name='bat({})'.format(i)) for i in range(25)]
    load = [m.addVar(name='load({})'.format(i)) for i in I]
    max_load = m.addVar(name='max_load')

    # Normalization
    rate_old_norm = []
    co2_norm = []
    ev_load_norm = []
    for i_ in I:
        rate_i = (rate_old[i_]-min(rate_old))/(max(rate_old)-min(rate_old))
        co2_i = (co2[i_] - min(co2)) / (max(co2) - min(co2))
        ev_load_i = (ev_load[i_] - min(ev_load)) / (max(ev_load) - min(ev_load))

        rate_old_norm.append(rate_i)
        co2_norm.append(co2_i)
        ev_load_norm.append(ev_load_i)

    # Obj Func: CO2; charge and discharge
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                occ_range * ((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I) +
    #                gp.quicksum(co2[i] * (a[i] * p[i] / miu - b[i] * miu) for i in I))

    # Obj Func: CO2; load dif
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                occ_range * ((bat[-1] - 0.9) * 100) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I) +
    #                9*gp.quicksum(co2[i] * (load[i] - ev_load[i]) for i in I))

    # Obj Func: No CO2
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                occ_range * 4*((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I))

    # Case 1
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                occ_range*((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I))

    # Case 2
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                occ_range*((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I) +
    #                9*gp.quicksum(co2[i]*(load[i] - ev_load[i]) for i in I))

    # Case 1 Norm
    # m.setObjective(gp.quicksum((load[i] - min_load) / (peak_load - min_load) * rate_old_norm[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24*((bat[-1] - 0.9) * 10) ** 2 -
    #                (peak_load-max_load)/(peak_load-min(ev_load)))
    # Case 2 Norm
    # m.setObjective(gp.quicksum((load[i] - min_load) / (peak_load - min_load) * rate_old_norm[i] for i in I) +
    #                gp.quicksum(c[i] for i in I)/24 +
    #                ((bat[-1] - 0.9) * 10) ** 2 +
    #                9*gp.quicksum(co2_norm[i] *
    #                              ((load[i] - min_load)/(peak_load-min_load)-ev_load_norm[i])
    #                              for i in I) -
    #                9*(peak_load-max_load)/(peak_load-min(ev_load)))
    # Case 3 Norm
    m.setObjective(5 * gp.quicksum((load[i] - min_load) / (peak_load - min_load) * rate_old_norm[i] for i in I) +
                   gp.quicksum(c[i] for i in I)/24 +
                   ((bat[-1] - 0.9) * 10) ** 2 +
                   5 * gp.quicksum(co2_norm[i] *
                                   ((load[i] - min_load)/(peak_load-min_load)-ev_load_norm[i])
                                   for i in I) -
                   5*(peak_load-max_load)/(peak_load-min_load))

    m.ModelSense = GRB.MINIMIZE

    # Add constraints
    for i in I:
        m.addConstr(bat[i+1] == bat[i] + a[i] * p[i] * miu / 60 * occ[i] - b[i] * occ[i] / (60*miu))
    m.addConstr(bat[0] == bat_init / 60 + 0.1 * c[0])
    m.addConstr(bat[-1] >= 0.88)
    # m.addConstr(bat[-1] - 0.1 * c[-1] == 0.9)
    # m.addConstr(d >= 0)
    # m.addConstr(d <= 0.1)

    for i in I:
        m.addConstr(c[i] >= 0)
        m.addConstr(c[i] <= 1 * occ[i])
        m.addConstr(p[i] >= 0)
        m.addConstr(p[i] <= 7.4 * occ[i])
        m.addConstr(b[i] >= 0)
        m.addConstr(b[i] <= L_predict[i] * occ[i])
        m.addConstr(b[i] + (0.2 - 0.1 * c[i]) * 60 * occ[i] <= bat[i] * 60 * occ[i])
        # m.addConstr(pdc[i] >= 0)
        # m.addConstr(pdc[i] <= L_predict[i])
        m.addConstr(bat[i] + 0.1 * c[i] >= 0.2)
        m.addConstr(bat[i] - 0.1 * c[i] <= 0.9)
        # m.addConstr(bat[i] >= 0.2)
        # m.addConstr(bat[i] <= 0.9)
        m.addConstr(a[i] * b[i] == 0)
        m.addConstr(load[i] == L_predict[i] + a[i] * p[i]/miu - b[i]*miu)
        m.addConstr(load[i] >= 0)
        m.addConstr(load[i] <= peak_load)
        # m.addConstr(sell_price[i] >= 0.0619)
        # m.addConstr(sell_price[i] <= 0.2215)
        # m.addConstr(rev[i] == load[i] * sell_price[i])
        # m.addConstr(F_cost[i] == load[i] * sell_price[i] + 0.375 * load[i])
    #     m.addConstr(F_emission[i] == load[i] * co2[i])
    # #
    # m.addConstr(F_emission_max == gp.max_(F_emission))
    # m.addConstr(F_emission_min == gp.min_(F_emission))
    #
    # m.addConstr(gp.quicksum(rev) <= sum(rev_max))
    # m.addConstr(gp.quicksum(rev) <= sum(rev_max))
    # m.addConstr(r1 == F_emission_max - F_emission_min)
    # m.addConstr(r2 * r1 == 1)
    # for i in I:
    #     m.addConstr(load[i] * co2[i] + s2[i] == F_emission_max - (r1 / (len(I) + 1)) * i)
    # m.addConstr(b[0] == 0)
    # m.addConstr(a[0] == 0)
    m.addConstr(max_load == gp.max_(load))

    m.update()
    m.params.NormAdjust = 0
    m.params.NonConvex = 2
    m.Params.timelimit = 100
    # m.Params.IterationLimit = 1
    m.optimize()

    # Output variables
    x_list = []
    y_list = []

    for v in m.getVars():
        x_list.append(v.VarName)
        y_list.append(v.X)
        # print('%s %g' % (v.VarName, v.X))

    df_temp = pd.DataFrame({'VarName': x_list, 'Value': y_list})
    occ_range_ = 24

    a1 = df_temp['Value'][:occ_range_].tolist()
    b1 = df_temp['Value'][occ_range_:occ_range_ * 2].tolist()
    c1 = df_temp['Value'][occ_range_ * 2:occ_range_ * 3].tolist()
    P = df_temp['Value'][occ_range_ * 3:occ_range_ * 4].tolist()
    bat1 = df_temp['Value'][occ_range_ * 4+1:occ_range_ * 5+1].tolist()
    loads = df_temp['Value'][occ_range_ * 5+1:occ_range_ * 6+1].tolist()
    # prices = df_temp['Value'][occ_range * 6:occ_range * 7].tolist()
    # rev1 = df_temp['Value'][occ_range * 6:occ_range * 7].tolist()
    # F_emission1 = df_temp['Value'][-occ_range:].tolist()

    # F_cost_ = np.zeros(occ_range)
    # for i in I:
    #     F_cost_[i] = 10e-4 * (rate_old[i] - 0.025) * loads[i] * loads[i] + 0.375 * loads[i]
    # F_cost1 = F_cost_.tolist()

    ev_charging = np.zeros(occ_range_)
    ev_discharging = np.zeros(occ_range_)

    for i in I:
        ev_charging[i] = a1[i] * P[i]
        ev_discharging[i] = -b1[i]

    ev_power = np.zeros(occ_range_)

    for i in I:
        ev_power[i] = ev_charging[i] + ev_discharging[i]

    # Put everything in a dataframe
    df_2 = pd.DataFrame(np.zeros(24))
    df_2.index = df_day.index

    # loads_day = df_day['Load'][:start_index - 1].tolist() + loads + df_day['Load'][
    #                                                                        end_index + 1:].tolist()
    loads_day = loads
    # price_day = df_day['Price'][:start_index - 1].tolist() + prices + df_day['Price'][end_index + 1:].tolist()
    charging_day = np.zeros(24)
    discharging_day = np.zeros(24)
    # revenue_day = revenues + f1_day['rev_old'][20:].tolist()
    soc_day = np.ones(24)
    df_2['Optimized load'] = loads_day
    df_2['Measured load'] = df_day['Load'].tolist()
    # df_2['New price'] = price_day
    df_2['Charging'] = charging_day
    df_2['Discharging'] = discharging_day
    df_2['rev_measured'] = df_day['rev_measured'].tolist()

    df_2['Charging'] = ev_charging
    df_2['Discharging'] = ev_discharging

    df_2['EV power'] = df_2['Charging'] + df_2['Discharging']

    df_2['soc'] = soc_day
    df_2['soc'] = bat1
    df_2['Price_signal'] = price_
    # df_2['soc'][:start_index] = bat1[0]
    # df_2['soc'][end_index:] = bat1[-1]

    # f_cost_old = np.zeros(24)
    # f_emission_old = np.zeros(24)
    # old_load = df_day['Load'].tolist()
    # for i in range(24):
    # #     f_cost_old[i] = 10e-4 * (rate[i] - 0.025) * old_load[i] * old_load[i] + 0.375 * old_load[i]
    #     f_emission_old[i] = old_load[i] * co2[i]
    #
    # # f_cost_new = f_cost_old.copy()
    # f_emission_new = f_emission_old.copy()
    #
    # df_obj = pd.DataFrame()
    # df_obj.index = df_day.index
    #
    # df_obj['cost_old'] = f_cost_old / 230
    # df_obj['cost_new'] = f_cost_new / 230
    # df_2['emission_old'] = f_emission_old / 225
    # df_2['emission_new'] = f_emission_new / 225
    #
    # df_obj['cost_new'][start_index:end_index] = np.array(F_cost1) / 230
    # df_2['emission_new'][start_index:end_index] = np.array(F_emission1) / 230
    co2_new = np.zeros(24)
    # co2_old = np.zeros(24)
    # for i in range(24):
    #     co2_new[i] = 0.0042*4.5*1e-7*loads_day[i]*loads_day[i] + 0.33*4.5*1e-4*loads_day[i] + 13.86*0.45
    #     co2_old[i] = 0.0042*4.5*1e-7*old_load[i]**2 + 0.33*4.5*1e-4*old_load[i] + 13.86*0.45
    for i in range(24):
        co2_new[i] = co2[i]*df_2['EV power'].tolist()[i]
        # co2_old[i] = co2[i]*old_load[i]

    # df_2['co2_old'] = co2_old
    df_2['co2_new'] = co2_new

    if not os.path.exists(f'./5.27_Case3_55/{month}/{day_folder}/{account_id}'):
        os.makedirs(f'./5.27_Case3_55/{month}/{day_folder}/{account_id}')

    df_2.to_csv(f'./5.27_Case3_55/{month}/{day_folder}/{account_id}/iteration{iteration}_profile.csv',
                index=True, header=True)

    return df_2


def get_load(account_id, month, day_folder, iteration):
    df_load_ = pd.read_csv(
        f'./5.27_Case3_55/{month}/{day_folder}/{account_id}/iteration{iteration}_profile.csv')
    loads = pd.DataFrame(df_load_['Optimized load'])
    old_load = pd.DataFrame(df_load_['Measured load'])

    return loads, old_load


def agg_load(account_lst, month, day_folder, iteration):
    df_aggload = pd.DataFrame()
    old_aggload = pd.DataFrame()
    for account_id in account_lst:
        try:
            loads, old_load = get_load(account_id, month, day_folder, iteration)
        except:
            loads = pd.Series([0] * 24)
            old_load = pd.Series([0] * 24)
        df_aggload = pd.concat([df_aggload, loads], axis=1)
        old_aggload = pd.concat([old_aggload, old_load], axis=1)

    return df_aggload, old_aggload


def grid_model(account_lst, month, day_folder, df_price, iteration):
    # Time steps
    T = range(24)
    # How many accounts
    I = range(len(account_lst))
    df_aggload, old_aggload = agg_load(account_lst, month, day_folder, iteration)
    load_matrix = np.zeros([len(account_lst), 24 * len(account_lst)])
    for i in range(len(df_aggload.T)):
        load_matrix[i, i * 24:(i + 1) * 24] = df_aggload.T.iloc[i].to_numpy()
    # rate_ = df_price
    rev_old = (df_price.iloc[:len(account_lst), :].to_numpy() @ old_aggload.to_numpy()).diagonal()
    rev_old1 = (df_price.iloc[iteration*len(account_lst):(iteration+1)*len(account_lst), :].to_numpy() @
                df_aggload.to_numpy()).diagonal()
    # rev_old = (df_price.to_numpy() @ df_aggload.to_numpy()).diagonal()
    old_rate = np.array(rate)
    # print(sum(rev_old), sum(rev_old1))

    # New load
    lst_sum = df_aggload.sum(axis=1).tolist()
    # Old load
    lst_old_sum = old_aggload.sum(axis=1).tolist()
    # New cost
    cost_matrix = 0.0001 * (old_rate - 0.025) * (np.array(lst_sum)) ** 2
    cost = sum(cost_matrix)
    # Old cost
    cost_old_matrix = 0.0001*(old_rate - 0.025)*(np.array(lst_old_sum)) ** 2
    cost_old = sum(cost_old_matrix)
    # Old profit
    profit_old = sum(rev_old) - cost_old

    # CVXPY
    price = cp.Variable((len(T) * len(I), 1))
    profit_obj = cp.sum(load_matrix @ price) - cost

    # Case 1
    # varphi = cp.Variable(len(account_lst))
    # obj = cp.Maximize(profit_obj - cp.sum(varphi) * profit_old)
    #
    # constraints = [price >= 0.0575 * 0.8,
    #                price <= 0.2215 * 1.2,
    #                profit_obj >= profit_old,
    #                varphi >= 0,
    #                varphi <= 1]
    # for i in I:
    #     load_val = load_matrix[i][i * 24:(i + 1) * 24]
    #     price_vals = price[i * 24:(i + 1) * 24]
    #     constraints += [load_val @ price_vals <=
    #                     rev_old.reshape(len(I), 1)[i] * (1 + varphi[i])]

    # Case 2
    # obj = cp.Maximize(profit_obj)
    #
    # constraints = [price >= 0.0575 * 0.8,
    #                price <= 0.2215 * 1.2]
    # for i in I:
    #     load_val = load_matrix[i][i * 24:(i + 1) * 24]
    #     price_vals = price[i * 24:(i + 1) * 24]
    #     constraints += [load_val @ price_vals <= rev_old.reshape(len(I), 1)[i]]

    # Case 3
    varphi = cp.Variable(len(account_lst))
    c = cp.Variable()
    obj = cp.Maximize(profit_obj + c*profit_old -
                      cp.sum(varphi)*profit_old)

    constraints = [price >= 0.0575*0.8,
                   price <= 0.2215*1.2,
                   profit_obj >= c*profit_old,
                   varphi >= 0,
                   varphi <= 1,
                   c >= 0,
                   c <= 2]
    for i in I:
        load_val = load_matrix[i][i*24:(i+1)*24]
        price_vals = price[i*24:(i+1)*24]
        constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i]*(1+varphi[i])]

    prob = cp.Problem(obj, constraints)
    val = prob.solve()

    df_price = price.value.reshape(len(I), 24)

    return df_price


def results_plot(df_new, month, day_folder, start_index, end_index, account_id, df_day, iteration, df_price, account_idx, account_lst):
    occ_range = pd.Timedelta(end_index - start_index).seconds / 3600 + 1
    occ_day = pd.DataFrame()
    occ_day['occ'] = np.ones(int(occ_range))
    occ_day.index = pd.date_range(start_index, end_index, freq='H')
    df_day['occ_temp'] = np.zeros(len(df_day))
    df_day.loc[occ_day.index, 'occ_temp'] = occ_day['occ']

    occ_status = df_day['occ_temp'].to_numpy()
    occ_index = np.array(np.where(occ_status != 0))
    occ_group = np.split(occ_index, np.where(np.diff(occ_index) != 1)[1] + 1, axis=1)

    plt.rcParams['figure.figsize'] = [18, 9]
    fig_day2, axs = plt.subplots(3, 1, sharex=True)
    axs[0].set_title('Loads', fontsize=14)
    axs[0].plot(df_new.index, df_new['Measured load'], 'g', marker='.', label="Measured load", lw=5,
                drawstyle='steps-post')
    axs[0].plot(df_new.index, df_new['Optimized load'], 'r', marker='.', label="Optimized load", lw=5,
                drawstyle='steps-post')
    axs[0].plot(df_new.index, df_new['EV power'], 'b--', label="EV power", lw=5, alpha=0.8, drawstyle='steps')
    # if pd.to_datetime(start_index).hour == 12:
    #     axs[0].axvspan(start_index, end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    # else:
    #     axs[0].axvspan(start_index + pd.Timedelta(hours=-1), end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    # axs[0].plot(df_2.index, df_day['occ_new'], 'grey', label="Occ", lw=5, alpha=0.8, drawstyle='steps')
    for j in range(len(occ_group)):
        axs[0].axvspan(df_new.index[max(occ_group[j][0][0] - 1, 0)], df_new.index[occ_group[j][0][-1]], alpha=0.5,
                       color='#A9D9D0', label='Occupied' if j == 0 else '')
    # axs[0].axvspan(start_index + pd.Timedelta(hours=-1), end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    axs[0].set_ylabel("Load (kWh)", fontsize=14)
    axs[0].xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    axs[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b%d %H'))
    # axs[0].set_xticks(x)
    # axs[0].set_xticklabels(time_ticks)
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    axs[0].tick_params(axis='both', labelsize=14)

    axs[1].set_title('Prices', fontsize=14)
    # axs[1].plot(df_new.index, df_new['New price'], 'k', label='New price', lw=5, drawstyle='steps')
    axs[1].plot(df_new.index, df_price.iloc[iteration * len(account_lst) + account_idx], 'k--', label='Price signal',
                lw=5, alpha=0.6, drawstyle='steps')
    # if pd.to_datetime(start_index).hour == 12:
    #     axs[1].axvspan(start_index, end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    # else:
    #     axs[1].axvspan(start_index + pd.Timedelta(hours=-1), end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    for j in range(len(occ_group)):
        axs[1].axvspan(df_new.index[max(occ_group[j][0][0] - 1, 0)], df_new.index[occ_group[j][0][-1]], alpha=0.5,
                       color='#A9D9D0', label='Occupied' if j == 0 else '')
    axs[1].set_ylabel('Price ($)', fontsize=14)
    axs[1].xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b%d %H'))
    # axs[1].set_xticks(x)
    # axs[1].set_xticklabels(time_ticks)
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    axs[1].tick_params(axis='both', labelsize=14)

    axs[2].set_title('Battery SOC', fontsize=14)
    axs[2].plot(df_new.index, df_new['soc'], 'y', label='Battery', lw=5, alpha=0.8)
    # axs[2].plot(dates_day, ev_power_day_soc, 'b--', label='EV power', lw=5, alpha=0.8, drawstyle='steps')
    # if pd.to_datetime(start_index).hour == 12:
    #     axs[2].axvspan(start_index, end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    # else:
    #     axs[2].axvspan(start_index + pd.Timedelta(hours=-1), end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    for j in range(len(occ_group)):
        axs[2].axvspan(df_new.index[max(occ_group[j][0][0] - 1, 0)], df_new.index[occ_group[j][0][-1]], alpha=0.5,
                       color='#A9D9D0', label='Occupied' if j == 0 else '')
    axs[2].set_ylabel('SOC', fontsize=14)
    axs[2].xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    axs[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b%d %H'))
    # axs[2].set_xticks(x)
    # axs[2].set_xticklabels(time_ticks)
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    axs[2].tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    # plt.show()
    # fig_day2.savefig(f'./29_7.4_7.9_temp/{account_id}/{account_id}_week{week}_day{day}_iteration{iteration}.png', facecolor='w')
    fig_day2.savefig(f'./5.27_Case3_55/{month}/{day_folder}/{account_id}/iteration{iteration}.png',
                     facecolor='w')
    plt.close()


def simulation(df_day_, rate, account_id, month, day_folder, df_price, iteration, co2, account_idx, account_lst):
    try:
        df_day = data_prepare(df_day_, rate)
        occ_range, start_index, end_index = occ_schedule(df_day)
        df_new = gurobi_individual_model(df_day, occ_range, start_index, end_index,
                                         account_id, month, day_folder,
                                         price_=df_price.
                                         iloc[iteration * len(account_lst) + account_idx].tolist(),
                                         iteration=iteration,
                                         co2=co2)
    # if iteration >= 2:
    #     results_plot(df_new, month, day_folder, start_index, end_index, account_id, df_day,
    #                  iteration, df_price, account_idx, account_lst)
    # else:
    #     pass
    # load = df_new['Optimized load'].to_numpy()
    # old_load = df_new['Measured load'].to_numpy()
    # rev1 = df_price.iloc[iteration * len(account_lst) + account_idx].tolist() @ load
    # rev_new.append(rev1)
    # rev2 = df_price.iloc[iteration * len(account_lst) + account_idx].tolist() @ old_load
    # rev_old.append(rev2)
    except:
        pass


# Main
if __name__ == '__main__':

    matplotlib.use('qt5agg')

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None
    # plt.rcParams['figure.figsize'] = [18, 9]

    # Summer
    rate_s = [0.077] * 3 + [0.2215] * 6 + [0.077] * 3 + [0.0619] * 6 + [0.077] * 6
    # Winter
    rate_w = [0.0737] * 5 + [0.0951] * 4 + [0.0737] * 2 + [0.0575] * 6 + [0.0951] * 4 + [0.0737] * 3

    co2 = [0.715, 0.72, 0.715, 0.71, 0.705, 0.69, 0.685, 0.69, 0.71, 0.74, 0.795, 0.87,
           0.87, 0.865, 0.845, 0.81, 0.765, 0.73, 0.7, 0.675, 0.66, 0.66, 0.68, 0.7]

    load_file_dir = './Example_files/SRP_Daily_Data_8'
    # month_lst = [1, 2, 3, 6, 7, 8]
    month_lst = [8, 1]
    # month_lst = [1, 2, 3]
    # month_lst = [6, 7, 8]

    t_start = time.time()

    for month in month_lst:
        if month <= 3:
            rate = rate_w
        else:
            rate = rate_s
        day_folders = os.listdir(f'{load_file_dir}/{month}')
        t_month = time.time()
        for day_folder in day_folders:
            file_lst = os.listdir(f'{load_file_dir}/{month}/{day_folder}')
            if len(file_lst) >= 200:
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

                while iteration <= 3:
                    t_iteration = time.time()
                    t_gurobi = time.time()
                    config = []
                    for account_idx in range(len(account_lst)):
                        account_id = account_lst[account_idx]
                        df_day_ = pd.read_csv(f'{load_file_dir}/{month}/{day_folder}/{account_id}.csv')
                        config.append((df_day_, rate, account_id, month, day_folder, df_price,
                                       iteration, co2, account_idx, account_lst))
                    with mp.get_context("spawn").Pool(processes=60) as Pool:
                        Pool.starmap(simulation, config)
                        Pool.close()
                        Pool.join()
                        # try:
                        #     df_day = data_prepare(df_day_, rate)
                        #     occ_range, start_index, end_index = occ_schedule(df_day)
                        #     df_new = gurobi_individual_model(df_day, occ_range, start_index, end_index,
                        #                                      account_id, month, day_folder,
                        #                                      price_=df_price.
                        #                                      iloc[iteration * len(account_lst) + account_idx].tolist(),
                        #                                      iteration=iteration,
                        #                                      co2=co2)
                        #     if iteration >= 2:
                        #         results_plot(df_new, month, day_folder, start_index, end_index, account_id, df_day,
                        #                      iteration, df_price, account_idx, account_lst)
                        #     else:
                        #         pass
                        #     load = df_new['Optimized load'].to_numpy()
                        #     old_load = df_new['Measured load'].to_numpy()
                        #     rev1 = df_price.iloc[iteration * len(account_lst) + account_idx].tolist() @ load
                        #     rev_new.append(rev1)
                        #     rev2 = df_price.iloc[iteration * len(account_lst) + account_idx].tolist() @ old_load
                        #     rev_old.append(rev2)
                        # except:
                        #     pass
                    print(f'Finish Gurobi iteration {iteration}, time use {round(time.time() - t_gurobi, 2)} s')
                    df_aggload, old_aggload = agg_load(account_lst, month, day_folder, iteration)
                    print(f'Start grid opt iteration {iteration}')
                    t_grid = time.time()
                    new_price = grid_model(account_lst, month, day_folder, df_price, iteration)
                    df_price = pd.concat([df_price, pd.DataFrame(new_price)])
                    lst_sum = df_aggload.sum(axis=1).tolist()
                    # new_price1 = new_price
                    # df_aggload1 = df_aggload
                    old_load_sum = old_aggload.sum(axis=1).tolist()
                    new_cost = np.zeros(24)
                    old_cost = np.zeros(24)
                    for t in range(24):
                        new_cost[t] = 0.0001 * (rate[t] - 0.025) * lst_sum[t] * lst_sum[t]
                        old_cost[t] = 0.0001 * (rate[t] - 0.025) * old_load_sum[t] * old_load_sum[t]
                        # new_cost[t] = 0.00001 * lst_sum[t] ** 2
                        # old_cost[t] = 0.00001 * old_load_sum[t] ** 2
                    old_cal_profit = sum(rev_new[-len(account_lst):]) - sum(old_cost)
                    old_cal_profit1 = sum(rev_old[-len(account_lst):]) - sum(old_cost)

                    new_profit = np.trace(new_price @ df_aggload.to_numpy()) - sum(new_cost)
                    old_rev_t = np.trace(df_price.iloc[:len(account_lst), :].
                                         to_numpy() @ old_aggload.to_numpy())
                    old_profit = old_rev_t - sum(old_cost)
                    new_rev_t = np.trace(new_price @ df_aggload.to_numpy())
                    # print(new_price @ df_aggload.to_numpy())
                    print(f'Old load: {sum(old_load_sum)}, New load: {sum(lst_sum)}')
                    print(f'Old cost: {sum(old_cost)}, New cost: {sum(new_cost)}')
                    print(f'Old rev: {old_rev_t}, New rev: {new_rev_t}')
                    # print(f'Old profit: {old_profit}, Old cal profit: {old_cal_profit}, '
                    #       f'Old cal profit1: {old_cal_profit1}, New profit: {new_profit}')
                    print(f'Old profit: {old_profit}, New profit: {new_profit}')

                    old_iter_profits.append(old_profit)
                    new_iter_profits.append(new_profit)
                    old_sum_cost.append(sum(old_cost))
                    new_sum_cost.append(sum(new_cost))
                    old_iter_rev.append(old_rev_t)
                    new_iter_rev.append(new_rev_t)
                    old_iter_cost = pd.concat([old_iter_cost, pd.Series(old_cost)])
                    new_iter_cost = pd.concat([new_iter_cost, pd.Series(new_cost)])
                    old_iter_loads = pd.concat([old_iter_loads, pd.Series(old_load_sum)])
                    new_iter_loads = pd.concat([new_iter_loads, pd.Series(lst_sum)])

                    print(f'Finish grid opt iteration {iteration}, time use {round(time.time() - t_grid, 2)} s')

                    print(f'Finish iteration {iteration}, time use {round(time.time() - t_iteration, 2)} s')

                    # Check co2 change
                    co2_change = np.zeros(24)
                    # co2_change_ = np.zeros(24)
                    for i in range(24):
                        co2_change[i] = co2[i] * (lst_sum[i] - old_load_sum[i])
                    sum_change = sum(co2_change)
                    co2_change_lst.append(sum_change)

                    # Check profit dif
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

                df_profit.to_csv(f'./5.27_Case3_55/{month}/{day_folder}/Profit_profile.csv', index=False, header=True)

                df_cost = pd.DataFrame()
                df_cost['old_cost'] = old_iter_cost
                df_cost['new_cost'] = new_iter_cost
                df_cost['old_load'] = old_iter_loads
                df_cost['new_load'] = new_iter_loads

                df_cost.to_csv(f'./5.27_Case3_55/{month}/{day_folder}/Cost_profile.csv', index=False, header=True)

                df_co2 = pd.DataFrame()
                df_co2['co2'] = co2_change_lst

                df_co2.to_csv(f'./5.27_Case3_55/{month}/{day_folder}/CO2_profile.csv', index=False, header=True)

            else:
                pass

        print(f'Finish month {month}, time use{round(time.time() - t_month, 2)} s')

    print(f'Finish ALL! Time use {round((time.time() - t_start)/60, 2)} min')
