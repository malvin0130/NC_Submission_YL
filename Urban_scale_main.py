import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib
import warnings
from datetime import timedelta
import os
import time
import cvxpy as cp
import random
import shutil
import sys
import multiprocessing as mp


# Calculate distance
# def haversine_np(lon1, lat1, lon2, lat2):
#   """
#   Calculate the great circle distance between two points
#   on the earth (specified in decimal degrees)
#   All args must be of equal length.
#   double check here: https://www.movable-type.co.uk/scripts/latlong.html
#   """
#   lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
#
#   dlon = lon2 - lon1
#   dlat = lat2 - lat1
#
#   a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
#
#   c = 2 * np.arcsin(np.sqrt(a))
#   km = 6367 * c
#   m = km * 1000
#
#   return m


def load_file(load_file_path, date, bldg_num):
    file_path = os.path.join(load_file_path, date, bldg_num)

    # Open load file
    df_load = pd.read_csv(f'{file_path}/single_family_house_mtr_revised.csv')
    bldg_load = df_load.iloc[:, -2]
    # bldg_occ = df_load.iloc[:, -1]

    new_bldg_load = bldg_load[-12:].tolist() + bldg_load[:12].tolist()
    # new_bldg_occ = bldg_occ[-12:].tolist() + bldg_occ[:12].tolist()

    month = date.split('_')[0]
    day = date.split('_')[1]
    try:
        hours = pd.date_range(start=f'2017-{month}-{day} 12:0', end=f'2017-{month}-{int(day) + 1} 11:0', freq='h')
    except:
        hours = pd.date_range(start=f'2017-{month}-{day} 12:0', end=f'2017-{int(month)+1}-1 11:0', freq='h')
    # df_load_new = pd.DataFrame({'Time': hours,
    #                             'Load': new_bldg_load,
    #                             'Occ': new_bldg_occ})
    df_load_new = pd.DataFrame({'Time': hours,
                                'Load': new_bldg_load})

    return df_load_new


def ev_status(occ_path, date, bldg_num):
    # Locate occ files
    occ_date_folder = os.path.join(occ_path, date)
    users_file = pd.read_csv(f'{occ_date_folder}/{bldg_num}.csv')
    # users_ID = users_file.iloc[:, 0]
    # users_list = users_ID.tolist()
    # users_file.index = users_file.iloc[:, 0]
    # users_file = users_file.iloc[:, 1:]

    # Use individual occ status as EV status (First leave)
    column = 0
    users_leave = pd.Series()
    while column in range(len(users_file.columns)):
        temp = users_file.iloc[:, column]
        if 0 not in temp.tolist():
            column += 1
        else:
            users_leave = temp
            break

    ev_user = users_leave[users_leave == 0].index.values[0]
    ev_user_occ = users_file.loc[ev_user]
    new_ev_user_occ = ev_user_occ[-12:].tolist() + ev_user_occ[:12].tolist()

    return new_ev_user_occ


# def bat_initial(ev_user, user_occ_path, date):
#     user_loc_file = pd.read_csv(f'{user_occ_path}/{date}/{ev_user}')
#     home_locations = user_loc_file.query('location_type == 0')
#     other_locations = user_loc_file.query('location_type == 2')
#     # Choose most frequent location
#     home_lon = home_locations['lon'].mode()[0]
#     home_lat = home_locations['lat'].mode()[0]
#     other_lon = other_locations['lon'].mode()[0]
#     other_lat = other_locations['lat'].mode()[0]
#     d1 = haversine_np(home_lon, home_lat, other_lon, other_lat)  # Unit: m
#     # travel distance * 2; 0.2 kWh/km
#     bat_usage = d1*2/1000*0.2
#     if bat_usage < 48:
#         bat_init = 48 - bat_usage
#     else:
#         bat_init = 12
#
#     return bat_init


def bat_initial(date):
    df_bat_init = pd.read_csv(f'./EV_bat/{date}_bat_init.csv')
    bat_init_lst = df_bat_init.iloc[:, 0].tolist()
    bat_init = random.choice(bat_init_lst)

    return bat_init


def data_prepare(df_load_new, new_ev_user_occ, bat_init, bldg_num):
    df_day = df_load_new.copy()
    df_day['EV_sch'] = new_ev_user_occ
    base_load = df_day['Load'].tolist()

    # Add charging load to load profile (charge immediately as EV arrives)
    # Random charging power
    cp = np.random.uniform(6, 7.4)
    # Total load need for charging
    tl = 48 - bat_init
    # EV_load[t] = base_load[t] + cp
    EV_load = np.zeros(24)
    # t = 0
    for t in range(len(df_day)):
        if tl > 0:
            # Charge during off-peak hour
            if t <= 6:
                EV_load[t] = base_load[t]
            if t > 6:
                if tl < cp:
                    EV_load[t] = base_load[t] + tl*new_ev_user_occ[t]
                    tl = tl - cp*new_ev_user_occ[t]
                if tl >= cp:
                    EV_load[t] = base_load[t] + cp * new_ev_user_occ[t]
                    tl = tl - cp*new_ev_user_occ[t]
        else:
            EV_load[t] = base_load[t]

    dif = np.zeros(24)
    for i in range(24):
        dif[i] = EV_load[i] - base_load[i]

    df_day['EV_load'] = EV_load
    df_day['Charging'] = dif

    df_day['Price'] = rate

    save_path = r"C:\apps-su\Yuewei\EV_Opt_E+\3.11_df_day"

    if not os.path.exists(os.path.join(save_path, date, bldg_num)):
        os.makedirs(os.path.join(save_path, date, bldg_num))

    df_day.to_csv(f'./3.11_df_day/{date}/{bldg_num}/df_day.csv', header=True, index=False)

    return df_day


def gurobi_individual_model(date, bldg_num, price_, iteration, bat_init, co2):
    I = range(24)
    df_day = pd.read_csv(f'./2.25_df_day/{date}/{bldg_num}/df_day.csv')
    L_predict = df_day['Load'].tolist()
    occ = df_day['EV_sch'].tolist()
    ev_load = df_day['EV_load'].tolist()
    ev_price = df_day['Price'].tolist()
    # rev_min = df_day['rev_shift'][start_index:end_index].tolist()
    # rate_old = df_day['Price'][start_index:end_index].tolist()
    # bill_old = sum(pd.Series(ev_load)*pd.Series(ev_price))
    df_day['price_outcome'] = price_
    rate_old = df_day['price_outcome'].tolist()
    peak_load = max(ev_load)

    miu = 0.9

    # read bat init
    # try:
    #     temp = pd.read_csv(f'./29_7.4_NEW/{account_id}/{account_id}_week{week}_initial.csv')
    #     bat_init = temp['Initial'][day]
    # except:
    #     bat_init = 12
    # bat_init = bat_initial(ev_user, user_occ_path, date)
    # bat_init = 12

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

    # Create model
    m = gp.Model()

    # Add variables
    a = [m.addVar(vtype=GRB.BINARY, name='a({})'.format(i)) for i in I]
    b = [m.addVar(name='b({})'.format(i)) for i in I]
    c = [m.addVar(name='c({})'.format(i)) for i in I]
    p = [m.addVar(name='p({})'.format(i)) for i in I]
    bat = [m.addVar(name='bat({})'.format(i)) for i in range(25)]
    load = [m.addVar(name='load({})'.format(i)) for i in I]
    max_load = m.addVar(name='Peak_Load')
    # load_min = m.addVar(name='Load_min')
    # load_max = m.addVar(name='Load_max')
    # factor = m.addVar(name='varphi')
    # sell_price = [m.addVar(name='price({})'.format(i)) for i in I]
    # pdc = [m.addVar(name='pdc({})'.format(i)) for i in I]
    # s2 = [m.addVar(name='slack({})'.format(i)) for i in I]
    # F_cost = [m.addVar(name='F_cost({})'.format(i)) for i in I]
    # rev = [m.addVar(name='Rev({})'.format(i)) for i in I]
    # pdc = [m.addVar(name='pdc({})'.format(i)) for i in I]
    # F_emission_max = m.addVar(name='Emission_max')
    # F_emission_min = m.addVar(name='Emission_min')
    # r1 = m.addVar(name='load_range')
    # r2 = m.addVar(name='r2')
    # F_emission = [m.addVar(name='F_emission({})'.format(i)) for i in I]
    # d = m.addVar(name='d')

    # m.setObjective(0.5*gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                occ_range * ((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I) +
    #                100*gp.quicksum(0.0042*4.5*1e-7*load[i]*load[i] + 0.33*4.5*1e-4*load[i] for i in I))
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24 * ((bat[-1] - 0.9) * 10) ** 2 +
    #                10*gp.quicksum(co2[i] * (a[i] * p[i]/miu - b[i]*miu) for i in I) -
    #                5*gp.quicksum(b[i] * rate_old[i] for i in I))
    # m.setObjective(4*gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24 * ((bat[-1] - 0.9) * 10) ** 2 +
    #                10*gp.quicksum(co2[i] * (load[i] - ev_load[i]) for i in I) -
    #                4*gp.quicksum(b[i] * rate_old[i] for i in I))
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24 * ((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I) +
    #                9*gp.quicksum(co2[i] * (a[i] * p[i]/miu - b[i]*miu) for i in I))
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24 * ((bat[-1] - 0.9) * 10) ** 2 +
    #                9*gp.quicksum(co2[i] * (load[i] - ev_load[i]) for i in I))
    # m.setObjective(gp.quicksum((load[i]-ev_load[i]) * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24*((bat[-1] - 0.9) * 10) ** 2 +
    #                9*gp.quicksum(0.76*co2[i] * (load[i] - ev_load[i]) for i in I) -
    #                gp.quicksum(b[i] * rate_old[i] for i in I))

    # Case 1 NEW obj
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I)/bill_old +
    #                gp.quicksum(c[i] for i in I)/24 +
    #                ((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * rate_old[i] for i in I)/bill_old)

    # Case 2&3 NEW obj (co2_1)
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I)/bill_old +
    #                gp.quicksum(c[i] for i in I)/24 +
    #                ((bat[-1] - 0.9) * 10) ** 2 +
    #                gp.quicksum((co2[i]-min(co2))/(max(co2)-min(co2)) * (load[i] - ev_load[i])/max(ev_load) for i in I) -
    #                gp.quicksum(b[i] * rate_old[i] for i in I)/bill_old)

    # Case 2 NEW
    # m.setObjective(0.1*gp.quicksum(load[i] *
    #                                (rate_old[i] - min(rate_old))/(max(rate_old)-min(rate_old)) for i in I) +
    #                gp.quicksum(c[i] for i in I)/24 +
    #                ((bat[-1] - 0.9) * 10) ** 2 -
    #                gp.quicksum(b[i] * (rate_old[i]-min(rate_old))/(max(rate_old)-min(rate_old)) for i in I) +
    #                0.9*gp.quicksum((co2[i]-min(co2))/(max(co2)-min(co2)) *
    #                                load[i] for i in I) -
    #                0.9*gp.quicksum((co2[i] - min(co2))/(max(co2)-min(co2)) *
    #                                ev_load[i] for i in I))

    # Case 3 NEW
    # Case 2&3 NEW obj (co2_9)
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24*((bat[-1] - 0.9) * 10) ** 2 +
    #                9*gp.quicksum(co2[i] * (load[i] - ev_load[i]) for i in I) -
    #                gp.quicksum(b[i] * rate_old[i] for i in I))

    # Case 1 Norm
    # m.setObjective(gp.quicksum((load[i]-min(ev_load))/(max(ev_load)-min(ev_load)) *
    #                            (rate_old[i]-min(rate_old))/(max(rate_old)-min(rate_old))
    #                            for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24 * ((bat[-1] - 0.9) * 10) ** 2)

    # Case 2 Norm
    # m.setObjective(gp.quicksum((load[i])/(max(ev_load)) *
    #                            (rate_old[i]-min(rate_old))/(max(rate_old)-min(rate_old))
    #                            for i in I) +
    #                3*gp.quicksum(c[i] for i in I) +
    #                ((bat[-1] - 0.9) * 10) ** 2 +
    #                9*gp.quicksum(co2_norm[i] *
    #                              ((load[i]/max(ev_load)) - (ev_load[i]/max(ev_load)))
    #                              for i in I))

    # Case 3 Norm
    m.setObjective(3*gp.quicksum((load[i])/(max(ev_load)) *
                                 (rate_old[i]-min(rate_old))/(max(rate_old)-min(rate_old))
                                 for i in I) +
                   5*gp.quicksum(c[i] for i in I) +
                   ((bat[-1]-0.9)*10)**2 +
                   7*gp.quicksum(co2_norm[i] *
                                 ((load[i])/(max(ev_load)) - (ev_load[i]/max(ev_load)))
                                 for i in I))

    # Case 2 Non-norm
    # m.setObjective(gp.quicksum(load[i] * rate_old[i] for i in I) +
    #                gp.quicksum(c[i] for i in I) +
    #                24*((bat[-1] - 0.9) * 10) ** 2 +
    #                9*gp.quicksum(co2[i] * (load[i] - ev_load[i]) for i in I))

    m.ModelSense = GRB.MINIMIZE

    # Add constraints
    for i in I:
        m.addConstr(bat[i+1] == bat[i] + a[i] * p[i] * miu / 60 * occ[i] - b[i] * occ[i] / (60 * miu))
    m.addConstr(bat[0] == bat_init / 60 + 0.1 * c[0])
    # m.addConstr(bat[-1] >= 0.88)
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
        m.addConstr(b[i] <= 7.4 * occ[i])
        m.addConstr(b[i] + (0.2 - 0.1 * c[i]) * 60 * occ[i] <= bat[i] * 60 * occ[i])
        # m.addConstr(pdc[i] >= 0)
        # m.addConstr(pdc[i] <= L_predict[i])
        m.addConstr(bat[i] + 0.1 * c[i] >= 0.2)
        m.addConstr(bat[i] - 0.1 * c[i] <= 0.9)
        # m.addConstr(bat[i] >= 0.2)
        # m.addConstr(bat[i] <= 0.9)
        m.addConstr(a[i] * b[i] == 0)
        m.addConstr(load[i] == L_predict[i] + a[i] * p[i] - b[i])
        m.addConstr(load[i] >= 0)
        m.addConstr(load[i] <= max(ev_load))

        # m.addConstr(sell_price[i] >= 0.0619)
        # m.addConstr(sell_price[i] <= 0.2215)
        # m.addConstr(rev[i] == load[i] * sell_price[i])
        # m.addConstr(F_cost[i] == load[i] * sell_price[i] + 0.375 * load[i])
    #     m.addConstr(F_emission[i] == load[i] * co2[i])

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

    # m.addConstr(load_max == gp.max_(load))
    # m.addConstr(load_min == gp.min_(load))
    # m.addConstr(r1 == load_max-load_min)
    # m.addConstr(r2*r1 == 1)
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
    occ_range = 24

    a1 = df_temp['Value'][:occ_range].tolist()
    b1 = df_temp['Value'][occ_range:occ_range * 2].tolist()
    c1 = df_temp['Value'][occ_range * 2:occ_range * 3].tolist()
    P = df_temp['Value'][occ_range * 3:occ_range * 4].tolist()
    bat1 = df_temp['Value'][occ_range * 4+1:occ_range * 5+1].tolist()
    loads = df_temp['Value'][occ_range * 5+1:occ_range * 6+1].tolist()
    # prices = df_temp['Value'][occ_range * 6:occ_range * 7].tolist()
    # rev1 = df_temp['Value'][occ_range * 6:occ_range * 7].tolist()
    # F_emission1 = df_temp['Value'][-occ_range:].tolist()

    # F_cost_ = np.zeros(occ_range)
    # for i in I:
    #     F_cost_[i] = 10e-4 * (rate_old[i] - 0.025) * loads[i] * loads[i] + 0.375 * loads[i]
    # F_cost1 = F_cost_.tolist()

    ev_charging = np.zeros(occ_range)
    ev_discharging = np.zeros(occ_range)

    for i in I:
        ev_charging[i] = a1[i] * P[i]
        ev_discharging[i] = -b1[i]

    ev_power = np.zeros(occ_range)

    for i in I:
        ev_power[i] = ev_charging[i] + ev_discharging[i]

    # Put everything in a dataframe
    df_day.index = pd.to_datetime(df_day.iloc[:, 0])
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
    df_2['Measured load'] = df_day['EV_load'].tolist()
    # df_2['New price'] = price_day
    df_2['Charging'] = charging_day
    df_2['Discharging'] = discharging_day
    # df_2['rev_measured'] = df_day['rev_measured'].tolist()

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
    # co2_new = np.zeros(24)
    # co2_old = np.zeros(24)
    # for i in range(24):
    #     co2_new[i] = 0.0042*4.5*1e-7*loads_day[i]*loads_day[i] + 0.33*4.5*1e-4*loads_day[i] + 13.86*0.45
    #     co2_old[i] = 0.0042*4.5*1e-7*old_load[i]**2 + 0.33*4.5*1e-4*old_load[i] + 13.86*0.45
    # for i in range(24):
    #     co2_new[i] = co2[i]*df_2['EV power'].tolist()[i]
        # co2_old[i] = co2[i]*old_load[i]

    # df_2['co2_old'] = co2_old
    # df_2['co2_new'] = co2_new

    # df_2['occ'] = df_day['Occ']*len(df_day['Occ'].unique())
    df_2['EV_sch'] = occ

    save_path = r"C:\apps-su\Yuewei\EV_Opt_E+\6.3_TRY_Case3"

    if not os.path.exists(os.path.join(save_path, date, bldg_num)):
        os.makedirs(os.path.join(save_path, date, bldg_num))

    df_2.to_csv(f'./6.3_TRY_Case3/{date}/{bldg_num}/{bldg_num}_iteration{iteration}_profile.csv',
                index=True, header=True)
    # df_obj.to_csv(f'./10.13_temp/{account_id}/{account_id}_week{week}_day{day}_iteration{iteration}_obj.csv', index=True, header=True)

    return df_2


def get_load(bldg_num, date, iteration):
    df_load_ = pd.read_csv(f'./6.3_TRY_Case3/{date}/{bldg_num}/{bldg_num}_iteration{iteration}_profile.csv')
    loads = pd.DataFrame(df_load_['Optimized load'])
    old_load = pd.DataFrame(df_load_['Measured load'])

    return loads, old_load


# Get bldg_lst (str list)
def get_bldg_lst(load_file_path, date, BR_lst, ori_lst):
    bldg_lst = []
    for BR_num in BR_lst:
        for ori in ori_lst:
            bldg_folder = os.listdir(os.path.join(load_file_path, date, BR_num, ori))
            bldg_lst += bldg_folder

    return bldg_lst


def agg_load(date, iteration, bldg_lst):
    df_aggload = pd.DataFrame()
    old_aggload = pd.DataFrame()

    for bldg_num in bldg_lst:
        try:
            loads, old_load = get_load(bldg_num, date, iteration)
        except:
            loads = pd.Series([0] * 24)
            old_load = pd.Series([0] * 24)
        df_aggload = pd.concat([df_aggload, loads], axis=1)
        old_aggload = pd.concat([old_aggload, old_load], axis=1)

    return df_aggload, old_aggload


def grid_model(bldg_lst, date, df_price, iteration, rate):
    # Time steps
    T = range(24)
    # How many accounts
    I = range(len(bldg_lst))
    df_aggload, old_aggload = agg_load(date, iteration, bldg_lst)
    load_matrix = np.zeros([len(bldg_lst), 24 * len(bldg_lst)])
    for i in range(len(df_aggload.T)):
        load_matrix[i, i * 24:(i + 1) * 24] = df_aggload.T.iloc[i].to_numpy()
    # rate_ = df_price
    rev_old = (df_price.iloc[:len(bldg_lst), :].to_numpy() @ old_aggload.to_numpy()).diagonal()
    # rev_old1 = (df_price.iloc[iteration*len(bldg_lst):(iteration+1)*len(bldg_lst), :].to_numpy() @
    #             df_aggload.to_numpy()).diagonal()
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
    cost_old_matrix = 0.0001 * (old_rate - 0.025) * (np.array(lst_old_sum)) ** 2
    cost_old = sum(cost_old_matrix)
    # Old profit
    profit_old = sum(rev_old) - cost_old

    # CVXPY
    price = cp.Variable((len(T) * len(I), 1))
    profit_obj = cp.sum(load_matrix @ price) - cost
    # new_rev = load_matrix @ price

    # Case 1 obj
    # factor = cp.Variable(len(bldg_lst))
    # obj = cp.Maximize(profit_obj - cp.sum(factor)*profit_old)
    # # Case 1: Harm no profit
    # constraints = [price >= 0.0575*0.8,
    #                price <= 0.2215*1.2,
    #                profit_obj >= profit_old,
    #                factor >= 0,
    #                factor <= 1]
    # for i in I:
    #     load_val = load_matrix[i][i*24:(i+1)*24]
    #     price_vals = price[i*24:(i+1)*24]
    #     constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i] * (1+factor[i])]

    # Case 2 obj
    # obj = cp.Maximize(profit_obj)
    # # Case 2: Benefit CO2 (No const profit)
    # constraints = [price >= 0.0619*0.8,
    #                price <= 0.2215*1.2,
    #                cp.sum(load_matrix@price) <= sum(rev_old.reshape(len(I), 1))]

    # Case 3 obj
    # factor = cp.Variable(len(bldg_lst))
    # obj = cp.Maximize(profit_obj - 500*cp.sum(factor))
    # # Case 3: Balance profit and CO2
    # # Household-level: add weights for CO2
    # # Distribution-level: allow violation on revenue
    # constraints = [price >= 0.0619 * 0.8,
    #                price <= 10,
    #                profit_obj >= profit_old,
    #                factor >= 0,
    #                factor <= 1]
    # for i in I:
    #     load_val = load_matrix[i][i*24:(i+1)*24]
    #     price_vals = price[i*24:(i+1)*24]
    #     constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i] * (1+factor[i])]

    # Case 1 NEW
    # factor = cp.Variable(len(bldg_lst))
    # obj = cp.Maximize(profit_obj - 1000*cp.sum(factor))
    #
    # constraints = [price >= 0.0619*0.8,
    #                price <= 0.2215*1.2,
    #                profit_obj >= profit_old,
    #                factor >= 0,
    #                factor <= 1]
    # for i in I:
    #     load_val = load_matrix[i][i*24:(i+1)*24]
    #     price_vals = price[i*24:(i+1)*24]
    #     constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i] * (1+factor[i])]

    # Case 2 NEW
    # obj = cp.Maximize(profit_obj)
    #
    # constraints = [price >= 0.0619*0.8,
    #                price <= 0.2215*1.2]
    # for i in I:
    #     load_val = load_matrix[i][i*24:(i+1)*24]
    #     price_vals = price[i*24:(i+1)*24]
    #     constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i]]

    # Case 3 NEW
    # factor = cp.Variable(len(bldg_lst))
    # c = cp.Variable()
    # obj = cp.Maximize(c - cp.sum(factor)/len(bldg_lst))
    #
    # constraints = [price >= 0.0619*0.8,
    #                price <= 0.2215*1.2,
    #                profit_obj >= c*profit_old,
    #                factor >= 0,
    #                factor <= 1,
    #                c >= 0,
    #                c <= 1]
    # for i in I:
    #     load_val = load_matrix[i][i*24:(i+1)*24]
    #     price_vals = price[i*24:(i+1)*24]
    #     constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i]*(1+factor[i])]

    # Case 1 NEW-NEW
    # obj = cp.Maximize(profit_obj)
    #
    # constraints = [price >= 0.0619*0.8,
    #                price <= 0.2215*1.2,
    #                profit_obj >= profit_old]
    # for i in I:
    #     load_val = load_matrix[i][i*24:(i+1)*24]
    #     price_vals = price[i*24:(i+1)*24]
    #     constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i]]

    # Case 2 NEW-NEW
    # obj = cp.Maximize(profit_obj)
    # constraints = [price >= 0.0575*0.8,
    #                price <= 0.2215*1.2]
    # for i in I:
    #     load_val = load_matrix[i][i*24:(i+1)*24]
    #     price_vals = price[i*24:(i+1)*24]
    #     constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i]]

    # Case 3 NEW-NEW
    varphi = cp.Variable(len(bldg_lst))
    c = cp.Variable()
    obj = cp.Maximize(profit_obj +
                      c*profit_old -
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


def results_plot(df_new, bldg_num, iteration, df_price, price_idx, date):
    # occ_range = pd.Timedelta(end_index - start_index).seconds / 3600 + 1
    # occ_day = pd.DataFrame()
    # occ_day['occ'] = np.ones(int(occ_range))
    # occ_day.index = pd.date_range(start_index, end_index, freq='H')
    # df_day['occ_temp'] = np.zeros(len(df_day))
    # df_day.loc[occ_day.index, 'occ_temp'] = occ_day['occ']
    df_day = pd.read_csv(f'./2.25_df_day/{date}/{bldg_num}/df_day.csv')

    occ_status = df_day['EV_sch'].to_numpy()
    occ_index = np.array(np.where(occ_status != 0))
    occ_group = np.split(occ_index, np.where(np.diff(occ_index) != 1)[1] + 1, axis=1)

    fig_day2, axs = plt.subplots(3, 1, sharex=True)
    axs[0].set_title('Loads', fontsize=14)
    axs[0].plot(df_new.index, df_new['Measured load'], 'g', marker='.', label="Measured load", lw=5,
                drawstyle='steps')
    axs[0].plot(df_new.index, df_new['Optimized load'], 'r', marker='.', label="Optimized load", lw=5,
                drawstyle='steps')
    axs[0].plot(df_new.index, df_new['EV power'], 'b--', label="EV power", lw=5, alpha=0.8, drawstyle='steps')
    # if pd.to_datetime(start_index).hour == 12:
    #     axs[0].axvspan(start_index, end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    # else:
    #     axs[0].axvspan(start_index + pd.Timedelta(hours=-1), end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    # axs[0].plot(df_2.index, df_day['occ_new'], 'grey', label="Occ", lw=5, alpha=0.8, drawstyle='steps')
    for j in range(len(occ_group)):
        axs[0].axvspan(df_new.index[max(occ_group[j][0][0] - 1, 0)], df_new.index[occ_group[j][0][-1]], alpha=0.5,
                       color='#A9D9D0', label='EV at home' if j == 0 else '')
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
    axs[1].plot(df_new.index, df_price.iloc[price_idx], 'k--', label='Price signal',
                lw=5, alpha=0.6, drawstyle='steps')
    # if pd.to_datetime(start_index).hour == 12:
    #     axs[1].axvspan(start_index, end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    # else:
    #     axs[1].axvspan(start_index + pd.Timedelta(hours=-1), end_index, alpha=0.5, color='#A9D9D0', label='Occupied')
    for j in range(len(occ_group)):
        axs[1].axvspan(df_new.index[max(occ_group[j][0][0] - 1, 0)], df_new.index[occ_group[j][0][-1]], alpha=0.5,
                       color='#A9D9D0', label='EV at home' if j == 0 else '')
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
                       color='#A9D9D0', label='EV at home' if j == 0 else '')
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
    fig_day2.savefig(f'./6.3_TRY_Case3/{date}/{bldg_num}/{bldg_num}_iteration{iteration}.png',
                     facecolor='w')
    plt.close()


def main(bldg_num, date, df_price, iteration, co2):
    try:
        bat_init = bat_initial(date)
        #
        # df_day = data_prepare(df_load_new, new_ev_user_occ, bat_init, bldg_num)

        df_new = gurobi_individual_model(date, bldg_num,
                                         df_price.loc[f'{bldg_num}_iteration{iteration}'].tolist(),
                                         iteration, bat_init, co2)

    # results_plot(df_new, bldg_num, iteration, df_price, price_idx, date)

    except:
        pass


if __name__ == '__main__':
    matplotlib.use('qt5agg')

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None
    plt.rcParams['figure.figsize'] = [18, 9]

    # Summer rate
    rate = [0.077] * 3 + [0.2215] * 6 + [0.077] * 3 + [0.0619] * 6 + [0.077] * 6

    # Winter rate
    # rate = [0.0737] * 5 + [0.0951] * 4 + [0.0737] * 2 + [0.0575] * 6 + [0.0951] * 4 + [0.0737] * 3

    # Old
    co2 = [0.715, 0.72, 0.715, 0.71, 0.705, 0.69, 0.685, 0.69, 0.71, 0.74, 0.795, 0.87,
           0.87, 0.865, 0.845, 0.81, 0.765, 0.73, 0.7, 0.675, 0.66, 0.66, 0.68, 0.7]
    # New
    # co2 = [0.39, 0.376, 0.372, 0.363, 0.358, 0.358, 0.363, 0.367, 0.363, 0.367, 0.372, 0.381,
    #        0.376, 0.381, 0.381, 0.363, 0.349, 0.322, 0.299, 0.308, 0.319, 0.386, 0.399, 0.399]

    occ_path = './Example_files/GAN_bldgs'
    load_file_path = './Example_files/GAN_bldgs_E+'
    dates_lst = os.listdir(load_file_path)
    date_lst = [x for x in dates_lst if x[:2] == '8_']
    # dates_lst = ['1_30']
    # BR_lst = ['BR423', 'BR1765', 'BR1945', 'BR2876', 'BR2877', 'BR3677']
    # ori_lst = ['E', 'N', 'S', 'W']

    # dates_lst = ['1_2', '8_12']

    t_start = time.time()

    # for date in date_lst:
    #     bldg_files = os.listdir(os.path.join(load_file_path, date))
    #     for bldg_num in bldg_files:
    #         try:
    #             df_load_new = load_file(load_file_path, date, bldg_num)
    #
    #             new_ev_user_occ = ev_status(occ_path, date, bldg_num)
    #
    #             bat_init = bat_initial(date)
    #
    #             df_day = data_prepare(df_load_new, new_ev_user_occ, bat_init, bldg_num)
    #         except:
    #             pass

    for date in date_lst:
        try:
            bldg_files = os.listdir(os.path.join(load_file_path, date))
            df_price = pd.DataFrame(np.array(rate*len(bldg_files)).reshape(len(bldg_files), 24))
            bldg_files_str = [x + '_iteration0' for x in bldg_files]
            df_price.index = bldg_files_str
            # df_price.index = bldg_files
            # price_idx = 0
            iteration = 0

            # rev_new = []
            # rev_old = []
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
            dif_lst = []
            co2_change_lst = []
            bug_bldg_lst = []

            while iteration <= 5:
                t_iteration = time.time()
                t_gurobi = time.time()
                config = []
                for bldg_num in bldg_files:
                    config.append((bldg_num, date, df_price, iteration, co2))
                with mp.get_context("spawn").Pool(processes=60) as Pool:
                    Pool.starmap(main, config)  # chunk submitted to the processor
                    Pool.close()  # terminate worker processes when all work already assigned has completed
                    Pool.join()  # wait all processes to terminate

                print(f'Finish Gurobi iteration {iteration}, time use {round(time.time() - t_gurobi, 2)} s')

                df_aggload, old_aggload = agg_load(date, iteration, bldg_files)
                print(f'Start grid opt iteration {iteration}')
                t_grid = time.time()
                new_price = grid_model(bldg_files, date, df_price, iteration, rate)
                new_price_df = pd.DataFrame(new_price)
                new_price_str = [x + f'_iteration{iteration+1}' for x in bldg_files]
                new_price_df.index = new_price_str
                df_price = pd.concat([df_price, new_price_df])
                # df_price.reset_index(inplace=True, drop=True)

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
                # old_cal_profit = sum(rev_new[-len(bldg_lst):]) - sum(old_cost)
                # old_cal_profit1 = sum(rev_old[-len(bldg_lst):]) - sum(old_cost)

                new_profit = np.trace(new_price @ df_aggload.to_numpy()) - sum(new_cost)
                old_rev_t = np.trace(df_price.iloc[:len(bldg_files), :].
                                     to_numpy() @ old_aggload.to_numpy())
                old_profit = old_rev_t - sum(old_cost)
                new_rev_t = np.trace(new_price @ df_aggload.to_numpy())
                # print(new_price @ df_aggload.to_numpy())
                # print(df_price.iloc[iteration * len(account_lst):(iteration + 1) * len(account_lst), :].to_numpy() @
                #       old_aggload.to_numpy())
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
                #
                # try:
                #     dif = co2_change_lst[-1] - co2_change_lst[-2]
                # except:
                #     dif = co2_change_lst[-1]
                # # dif_lst.append(dif)
                # if dif > 0:
                #     break
                # else:
                #     iteration += 1
                # iteration += 1

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

                # if new_profit >= old_profit:
                #     iteration += 1
                # else:
                #     break

            df_profit = pd.DataFrame()
            df_profit['old_profit'] = old_iter_profits
            df_profit['new_profit'] = new_iter_profits
            df_profit['old_cost'] = old_sum_cost
            df_profit['new_cost'] = new_sum_cost
            df_profit['old_rev'] = old_iter_rev
            df_profit['new_rev'] = new_iter_rev

            df_profit.to_csv(f'./6.3_TRY_Case3/{date}/{date}_profit_profile.csv', index=False, header=True)

            df_cost = pd.DataFrame()
            df_cost['old_cost'] = old_iter_cost
            df_cost['new_cost'] = new_iter_cost
            df_cost['old_load'] = old_iter_loads
            df_cost['new_load'] = new_iter_loads

            df_cost.to_csv(f'./6.3_TRY_Case3/{date}/{date}_cost_profile.csv', index=False, header=True)

            df_co2 = pd.DataFrame()
            df_co2['co2'] = co2_change_lst

            df_co2.to_csv(f'./6.3_TRY_Case3/{date}/{date}_co2_profile.csv', index=False, header=True)
        except:
            pass

    print(f'Finish All! Time use {round(time.time() - t_start, 2)} s')





