import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import cvxpy as cp
import random


class UrbanScaleSimulation:

    def __init__(self, occ_path, load_file_path, date, bldg_num, iteration, case_num, df_price, bldg_lst):
        self.occ_path = occ_path
        self.load_file_path = load_file_path
        self.date = date
        self.bldg_num = bldg_num
        self.iteration = iteration
        self.case_num = case_num
        self.df_price = df_price
        self.bldg_lst = bldg_lst

        # CO2 factors
        self.co2 = [0.715, 0.72, 0.715, 0.71, 0.705, 0.69, 0.685, 0.69, 0.71, 0.74, 0.795, 0.87,
                    0.87, 0.865, 0.845, 0.81, 0.765, 0.73, 0.7, 0.675, 0.66, 0.66, 0.68, 0.7]
        # Rate
        self.rate = [0.077] * 3 + [0.2215] * 6 + [0.077] * 3 + [0.0619] * 6 + [0.077] * 6

        self.gurobi_individual_model()

    def load_file(self):
        file_path = os.path.join(self.load_file_path, self.date, self.bldg_num)

        # Open load file
        df_load = pd.read_csv(f'{file_path}/single_family_house_mtr_revised.csv')
        bldg_load = df_load.iloc[:, -2]

        new_bldg_load = bldg_load[-12:].tolist() + bldg_load[:12].tolist()

        month = self.date.split('_')[0]
        day = self.date.split('_')[1]
        try:
            hours = pd.date_range(start=f'2017-{month}-{day} 12:0', end=f'2017-{month}-{int(day) + 1} 11:0', freq='h')
        except:
            hours = pd.date_range(start=f'2017-{month}-{day} 12:0', end=f'2017-{int(month) + 1}-1 11:0', freq='h')
        df_load_new = pd.DataFrame({'Time': hours, 'Load': new_bldg_load})

        return df_load_new

    def ev_status(self):
        # Locate occ files
        occ_date_folder = os.path.join(self.occ_path, self.date)
        users_file = pd.read_csv(f'{occ_date_folder}/{self.bldg_num}.csv')

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

    def bat_initial(self):
        df_bat_init = pd.read_csv(f'./EV_bat/{self.date}_bat_init.csv')
        bat_init_lst = df_bat_init.iloc[:, 0].tolist()
        bat_init = random.choice(bat_init_lst)

        return bat_init

    def data_prepare(self):
        df_load_new = self.load_file()
        new_ev_user_occ = self.ev_status()
        df_day = df_load_new.copy()
        df_day['EV_sch'] = new_ev_user_occ
        base_load = df_day['Load'].tolist()

        bat_init = self.bat_initial()

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
                        EV_load[t] = base_load[t] + tl * new_ev_user_occ[t]
                        tl = tl - cp * new_ev_user_occ[t]
                    if tl >= cp:
                        EV_load[t] = base_load[t] + cp * new_ev_user_occ[t]
                        tl = tl - cp * new_ev_user_occ[t]
            else:
                EV_load[t] = base_load[t]

        dif = np.zeros(24)
        for i in range(24):
            dif[i] = EV_load[i] - base_load[i]

        df_day['EV_load'] = EV_load
        df_day['Charging'] = dif

        df_day['Price'] = self.rate

        save_path = r"C:\apps-su\Yuewei\EV_Opt_E+\7.14_df_day"

        if not os.path.exists(os.path.join(save_path, self.date, self.bldg_num)):
            os.makedirs(os.path.join(save_path, self.date, self.bldg_num))

        df_day.to_csv(f'./7.14_df_day/{self.date}/{self.bldg_num}/df_day.csv', header=True, index=False)

        return df_day

    def gurobi_individual_model(self):
        I = range(24)
        df_day = self.data_prepare()
        L_predict = df_day['Load'].tolist()
        occ = df_day['EV_sch'].tolist()
        ev_load = df_day['EV_load'].tolist()
        ev_price = df_day['Price'].tolist()
        price_signal = self.df_price.loc[f'{self.bldg_num}_iteration{self.iteration}'].tolist()
        df_day['price_outcome'] = price_signal
        rate_old = df_day['price_outcome'].tolist()

        # Efficiency
        miu = 0.9

        # Normalization
        rate_old_norm = []
        co2_norm = []
        ev_load_norm = []
        for i_ in I:
            rate_i = (rate_old[i_] - min(rate_old)) / (max(rate_old) - min(rate_old))
            co2_i = (self.co2[i_] - min(self.co2)) / (max(self.co2) - min(self.co2))
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

        # Define objective function
        if self.case_num == 'Case1':
            m.setObjective(gp.quicksum((load[i]-min(ev_load))/(max(ev_load)-min(ev_load)) *
                                       (rate_old[i]-min(rate_old))/(max(rate_old)-min(rate_old))
                                       for i in I) +
                           gp.quicksum(c[i] for i in I) +
                           24 * ((bat[-1] - 0.9) * 10) ** 2)
        elif self.case_num == 'Case2':
            m.setObjective(gp.quicksum((load[i])/(max(ev_load)) *
                                       (rate_old[i]-min(rate_old))/(max(rate_old)-min(rate_old))
                                       for i in I) +
                           3*gp.quicksum(c[i] for i in I) +
                           ((bat[-1] - 0.9) * 10) ** 2 +
                           9*gp.quicksum(co2_norm[i] *
                                         ((load[i]/max(ev_load)) - (ev_load[i]/max(ev_load)))
                                         for i in I))
        elif self.case_num == 'Case3':
            m.setObjective(3 * gp.quicksum((load[i]) / (max(ev_load)) *
                                           (rate_old[i] - min(rate_old)) / (max(rate_old) - min(rate_old))
                                           for i in I) +
                           5 * gp.quicksum(c[i] for i in I) +
                           ((bat[-1] - 0.9) * 10) ** 2 +
                           7 * gp.quicksum(co2_norm[i] *
                                           ((load[i]) / (max(ev_load)) - (ev_load[i] / max(ev_load)))
                                           for i in I))
        m.ModelSense = GRB.MINIMIZE

        # Add constraints
        bat_init = self.bat_initial()
        for i in I:
            m.addConstr(bat[i + 1] == bat[i] + a[i] * p[i] * miu / 60 * occ[i] - b[i] * occ[i] / (60 * miu))
        m.addConstr(bat[0] == bat_init / 60 + 0.1 * c[0])

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
        m.addConstr(max_load == gp.max_(load))

        m.update()
        m.params.NormAdjust = 0
        m.params.NonConvex = 2
        m.Params.timelimit = 100
        # m.Params.IterationLimit = 1
        m.optimize()

        # Log output file
        # Output variables
        x_list = []
        y_list = []

        for v in m.getVars():
            x_list.append(v.VarName)
            y_list.append(v.X)

        df_temp = pd.DataFrame({'VarName': x_list, 'Value': y_list})
        occ_range = 24

        a1 = df_temp['Value'][:occ_range].tolist()
        b1 = df_temp['Value'][occ_range:occ_range * 2].tolist()
        c1 = df_temp['Value'][occ_range * 2:occ_range * 3].tolist()
        P = df_temp['Value'][occ_range * 3:occ_range * 4].tolist()
        bat1 = df_temp['Value'][occ_range * 4 + 1:occ_range * 5 + 1].tolist()
        loads = df_temp['Value'][occ_range * 5 + 1:occ_range * 6 + 1].tolist()

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
        loads_day = loads
        charging_day = np.zeros(24)
        discharging_day = np.zeros(24)
        soc_day = np.ones(24)
        df_2['Optimized load'] = loads_day
        df_2['Measured load'] = df_day['EV_load'].tolist()
        df_2['Charging'] = charging_day
        df_2['Discharging'] = discharging_day

        df_2['Charging'] = ev_charging
        df_2['Discharging'] = ev_discharging

        df_2['EV power'] = df_2['Charging'] + df_2['Discharging']

        df_2['soc'] = soc_day
        df_2['soc'] = bat1

        df_2['Price_signal'] = price_signal

        df_2['EV_sch'] = occ

        if not os.path.exists(os.path.join(self.case_num, self.date, self.bldg_num)):
            os.makedirs(os.path.join(self.case_num, self.date, self.bldg_num))

        df_2.to_csv(f'./{self.case_num}/{self.date}/{self.bldg_num}/{self.bldg_num}_iteration{self.iteration}_profile.csv',
                    index=False, header=True)


class UrbanLevelGrid:
    def __init__(self, bldg_lst, case_num, date, iteration, df_price):
        self.bldg_lst = bldg_lst
        self.case_num = case_num
        self.date = date
        self.iteration = iteration
        self.df_price = df_price

        # CO2 factors
        self.co2 = [0.715, 0.72, 0.715, 0.71, 0.705, 0.69, 0.685, 0.69, 0.71, 0.74, 0.795, 0.87,
                    0.87, 0.865, 0.845, 0.81, 0.765, 0.73, 0.7, 0.675, 0.66, 0.66, 0.68, 0.7]
        # Rate
        self.rate = [0.077] * 3 + [0.2215] * 6 + [0.077] * 3 + [0.0619] * 6 + [0.077] * 6
        #
        # self.grid_main()

    def get_load(self, bldg_):
        df_load_ = pd.read_csv(f'./{self.case_num}/{self.date}/{bldg_}/{bldg_}_iteration{self.iteration}_profile.csv')
        loads = pd.DataFrame(df_load_['Optimized load'])
        old_load = pd.DataFrame(df_load_['Measured load'])

        return loads, old_load

    def agg_load(self):
        df_aggload = pd.DataFrame()
        old_aggload = pd.DataFrame()

        for bldg in self.bldg_lst:
            try:
                loads, old_load = self.get_load(bldg)
            except:
                loads = pd.Series([0] * 24)
                old_load = pd.Series([0] * 24)
            df_aggload = pd.concat([df_aggload, loads], axis=1)
            old_aggload = pd.concat([old_aggload, old_load], axis=1)

        return df_aggload, old_aggload

    def grid_model(self):
        # Time steps
        T = range(24)
        # How many accounts
        I = range(len(self.bldg_lst))
        df_aggload, old_aggload = self.agg_load()
        load_matrix = np.zeros([len(self.bldg_lst), 24 * len(self.bldg_lst)])
        for i in range(len(df_aggload.T)):
            load_matrix[i, i * 24:(i + 1) * 24] = df_aggload.T.iloc[i].to_numpy()
        rev_old = (self.df_price.iloc[:len(self.bldg_lst), :].to_numpy() @ old_aggload.to_numpy()).diagonal()

        old_rate = np.array(self.rate)

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

        if self.case_num == 'Case1':
            factor = cp.Variable(len(self.bldg_lst))
            obj = cp.Maximize(profit_obj - cp.sum(factor)*profit_old)
            # Case 1: Harm no profit
            constraints = [price >= 0.0575*0.8,
                           price <= 0.2215*1.2,
                           profit_obj >= profit_old,
                           factor >= 0,
                           factor <= 1]
            for i in I:
                load_val = load_matrix[i][i*24:(i+1)*24]
                price_vals = price[i*24:(i+1)*24]
                constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i] * (1+factor[i])]
        elif self.case_num == 'Case2':
            obj = cp.Maximize(profit_obj)
            constraints = [price >= 0.0575*0.8,
                           price <= 0.2215*1.2]
            for i in I:
                load_val = load_matrix[i][i*24:(i+1)*24]
                price_vals = price[i*24:(i+1)*24]
                constraints += [load_val@price_vals <= rev_old.reshape(len(I), 1)[i]]
        elif self.case_num == 'Case3':
            varphi = cp.Variable(len(self.bldg_lst))
            c = cp.Variable()
            obj = cp.Maximize(profit_obj +
                              c * profit_old -
                              cp.sum(varphi) * profit_old)

            constraints = [price >= 0.0575 * 0.8,
                           price <= 0.2215 * 1.2,
                           profit_obj >= c * profit_old,
                           varphi >= 0,
                           varphi <= 1,
                           c >= 0,
                           c <= 2]
            for i in I:
                load_val = load_matrix[i][i * 24:(i + 1) * 24]
                price_vals = price[i * 24:(i + 1) * 24]
                constraints += [load_val @ price_vals <= rev_old.reshape(len(I), 1)[i] * (1 + varphi[i])]

        prob = cp.Problem(obj, constraints)
        val = prob.solve()

        df_price = price.value.reshape(len(I), 24)

        return df_price

    def grid_main(self):
        df_aggload, old_aggload = self.agg_load()
        # df_aggload, old_aggload = UrbanLevelGrid.agg_load
        new_price = self.grid_model()
        new_price_df = pd.DataFrame(new_price)
        new_price_str = [x + f'_iteration{self.iteration + 1}' for x in self.bldg_lst]
        new_price_df.index = new_price_str
        df_new_price = pd.concat([self.df_price, new_price_df])

        lst_sum = df_aggload.sum(axis=1).tolist()
        old_load_sum = old_aggload.sum(axis=1).tolist()
        new_cost = np.zeros(24)
        old_cost = np.zeros(24)
        for t in range(24):
            new_cost[t] = 0.0001 * (self.rate[t] - 0.025) * lst_sum[t] * lst_sum[t]
            old_cost[t] = 0.0001 * (self.rate[t] - 0.025) * old_load_sum[t] * old_load_sum[t]

        new_profit = np.trace(new_price @ df_aggload.to_numpy()) - sum(new_cost)
        old_rev_t = np.trace(self.df_price.iloc[:len(self.bldg_lst), :].
                             to_numpy() @ old_aggload.to_numpy())
        old_profit = old_rev_t - sum(old_cost)
        new_rev_t = np.trace(new_price @ df_aggload.to_numpy())

        # Save everything in a dict
        output_dict = {'New load': lst_sum,
                       'Old load': old_load_sum,
                       'New cost': new_cost,
                       'Old cost': old_cost,
                       'New profit': new_profit,
                       'Old profit': old_profit,
                       'New rev': new_rev_t,
                       'Old rev': old_rev_t}

        return df_new_price, output_dict
