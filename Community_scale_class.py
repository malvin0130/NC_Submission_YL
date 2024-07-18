import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import cvxpy as cp


class SRPIndividual:
    def __init__(self, load_file_dir, df_price, case_num, month, date, iteration, account_lst, account_idx):
        self.load_file_dir = load_file_dir
        self.df_price = df_price
        self.case_num = case_num
        self.month = month
        self.date = date
        self.iteration = iteration
        self.account_lst = account_lst
        self.account_idx = account_idx

        # CO2 factors
        self.co2 = [0.715, 0.72, 0.715, 0.71, 0.705, 0.69, 0.685, 0.69, 0.71, 0.74, 0.795, 0.87,
                    0.87, 0.865, 0.845, 0.81, 0.765, 0.73, 0.7, 0.675, 0.66, 0.66, 0.68, 0.7]
        # Rate
        self.rate = [0.077] * 3 + [0.2215] * 6 + [0.077] * 3 + [0.0619] * 6 + [0.077] * 6

        self.account_id = self.account_lst[self.account_idx]
        self.df_day = pd.read_csv(f'{load_file_dir}/{month}/{date}/{self.account_id}.csv')

        self.gurobi_individual_model()

    def occ_schedule(self):
        self.df_day.index = pd.to_datetime(self.df_day.iloc[:, 0])
        occ_ = self.df_day[self.df_day['occ'] == 1]

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
        else:
            occ_range = 19
            start_index, end_index = self.df_day.index.values[0], self.df_day.index.values[18]

        return occ_range, start_index, end_index

    def data_prepare(self):
        self.df_day['Price'] = self.rate

        rev_measured = np.zeros(24)
        rev_old = np.zeros(24)

        measured_load = self.df_day['Load'].to_list()
        pred_result = self.df_day['Base_load'].to_list()

        for i in range(len(self.df_day)):
            rev_measured[i] = self.rate[i] * measured_load[i]
            rev_old[i] = self.rate[i] * pred_result[i]

        self.df_day['rev_measured'] = rev_measured
        self.df_day['rev_old'] = rev_old

    def gurobi_individual_model(self):
        occ_range, start_index, end_index = self.occ_schedule()

        self.data_prepare()
        occ_day = pd.DataFrame()
        occ_day['occ'] = np.ones(int(occ_range))
        occ_day.index = pd.date_range(start_index, end_index, freq='H')
        self.df_day['occ_temp'] = np.zeros(len(self.df_day))
        self.df_day.loc[occ_day.index, 'occ_temp'] = occ_day['occ']

        I = range(24)
        L_predict = self.df_day['Base_load'].tolist()
        ev_load = self.df_day['Load'].tolist()
        occ = self.df_day['occ_temp'].tolist()
        ev_price = self.df_day['Price'].tolist()

        price_ = self.df_price.iloc[self.iteration * len(self.account_lst) + self.account_idx].tolist()
        self.df_day['price_outcome'] = price_
        rate_old = self.df_day['price_outcome'].tolist()
        miu = 0.9
        peak_load = max(ev_load)
        min_load = min(ev_load)

        # Calculate bat init
        mask_temp = (self.df_day['charging'] == 1) & (self.df_day['occ'] == 1)
        df_charging = self.df_day.loc[mask_temp]
        try:  # Charging happened
            bat_use = df_charging.dif.sum()
            if bat_use < 48:
                bat_init = 60 * 0.9 - bat_use
            else:
                bat_init = 0
        except:  # No charging happened
            bat_init = 60 * 0.9

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
            rate_i = (rate_old[i_] - min(rate_old)) / (max(rate_old) - min(rate_old))
            co2_i = (self.co2[i_] - min(self.co2)) / (max(self.co2) - min(self.co2))
            ev_load_i = (ev_load[i_] - min(ev_load)) / (max(ev_load) - min(ev_load))

            rate_old_norm.append(rate_i)
            co2_norm.append(co2_i)
            ev_load_norm.append(ev_load_i)

        if self.case_num == 'Case1':
            m.setObjective(gp.quicksum((load[i] - min_load) / (peak_load - min_load) * rate_old_norm[i] for i in I) +
                           gp.quicksum(c[i] for i in I) +
                           24*((bat[-1] - 0.9) * 10) ** 2 -
                           (peak_load-max_load)/(peak_load-min(ev_load)))
        elif self.case_num == 'Case2':
            m.setObjective(gp.quicksum((load[i] - min_load) / (peak_load - min_load) * rate_old_norm[i] for i in I) +
                           gp.quicksum(c[i] for i in I)/24 +
                           ((bat[-1] - 0.9) * 10) ** 2 +
                           9*gp.quicksum(co2_norm[i] *
                                         ((load[i] - min_load)/(peak_load-min_load)-ev_load_norm[i])
                                         for i in I) -
                           9*(peak_load-max_load)/(peak_load-min(ev_load)))
        elif self.case_num == 'Case3':
            m.setObjective(
                5 * gp.quicksum((load[i] - min_load) / (peak_load - min_load) * rate_old_norm[i] for i in I) +
                gp.quicksum(c[i] for i in I) / 24 +
                ((bat[-1] - 0.9) * 10) ** 2 +
                5 * gp.quicksum(co2_norm[i] *
                                ((load[i] - min_load) / (peak_load - min_load) - ev_load_norm[i])
                                for i in I) -
                5 * (peak_load - max_load) / (peak_load - min_load))

        m.ModelSense = GRB.MINIMIZE

        # Add constraints
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
            m.addConstr(b[i] + (0.2 - 0.1 * c[i]) * 60 * occ[i] <= bat[i] * 60 * occ[i])
            m.addConstr(bat[i] + 0.1 * c[i] >= 0.2)
            m.addConstr(bat[i] - 0.1 * c[i] <= 0.9)
            m.addConstr(a[i] * b[i] == 0)
            m.addConstr(load[i] == L_predict[i] + a[i] * p[i] / miu - b[i] * miu)
            m.addConstr(load[i] >= 0)
            m.addConstr(load[i] <= peak_load)

        m.addConstr(max_load == gp.max_(load))

        m.update()
        m.params.NormAdjust = 0
        m.params.NonConvex = 2
        m.Params.timelimit = 100
        m.optimize()

        # Output variables
        x_list = []
        y_list = []

        for v in m.getVars():
            x_list.append(v.VarName)
            y_list.append(v.X)

        df_temp = pd.DataFrame({'VarName': x_list, 'Value': y_list})
        occ_range_ = 24

        a1 = df_temp['Value'][:occ_range_].tolist()
        b1 = df_temp['Value'][occ_range_:occ_range_ * 2].tolist()
        c1 = df_temp['Value'][occ_range_ * 2:occ_range_ * 3].tolist()
        P = df_temp['Value'][occ_range_ * 3:occ_range_ * 4].tolist()
        bat1 = df_temp['Value'][occ_range_ * 4 + 1:occ_range_ * 5 + 1].tolist()
        loads = df_temp['Value'][occ_range_ * 5 + 1:occ_range_ * 6 + 1].tolist()

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
        df_2.index = self.df_day.index

        loads_day = loads
        charging_day = np.zeros(24)
        discharging_day = np.zeros(24)
        soc_day = np.ones(24)
        df_2['Optimized load'] = loads_day
        df_2['Measured load'] = self.df_day['Load'].tolist()
        df_2['Charging'] = charging_day
        df_2['Discharging'] = discharging_day
        df_2['rev_measured'] = self.df_day['rev_measured'].tolist()

        df_2['Charging'] = ev_charging
        df_2['Discharging'] = ev_discharging

        df_2['EV power'] = df_2['Charging'] + df_2['Discharging']

        df_2['soc'] = soc_day
        df_2['soc'] = bat1
        df_2['Price_signal'] = price_

        co2_new = np.zeros(24)
        for i in range(24):
            co2_new[i] = self.co2[i] * df_2['EV power'].tolist()[i]

        save_path = os.path.join('Community_level', self.case_num, self.month, self.date, self.account_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df_2.to_csv(f'./{save_path}/{self.account_id}_iteration{self.iteration}_profile.csv',
                    index=False, header=True)


class SRPUrbanGrid:
    def __init__(self, month, date, case_num, account_lst, iteration, df_price):
        self.month = month
        self.date = date
        self.case_num = case_num
        self.account_lst = account_lst
        self.iteration = iteration
        self.df_price = df_price

        # Rate
        self.rate = [0.077] * 3 + [0.2215] * 6 + [0.077] * 3 + [0.0619] * 6 + [0.077] * 6

    def get_load(self, account_id):
        save_path = os.path.join('Community_level', self.case_num, self.month, self.date, account_id)
        df_load_ = pd.read_csv(f'./{save_path}/{account_id}_iteration{self.iteration}_profile.csv')
        loads = pd.DataFrame(df_load_['Optimized load'])
        old_load = pd.DataFrame(df_load_['Measured load'])

        return loads, old_load

    def agg_load(self):
        df_aggload = pd.DataFrame()
        old_aggload = pd.DataFrame()
        for account_id in self.account_lst:
            try:
                loads, old_load = self.get_load(account_id)
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
        I = range(len(self.account_lst))
        df_aggload, old_aggload = self.agg_load()
        load_matrix = np.zeros([len(self.account_lst), 24 * len(self.account_lst)])
        for i in range(len(df_aggload.T)):
            load_matrix[i, i * 24:(i + 1) * 24] = df_aggload.T.iloc[i].to_numpy()
        rev_old = (self.df_price.iloc[:len(self.account_lst), :].to_numpy() @ old_aggload.to_numpy()).diagonal()

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
            varphi = cp.Variable(len(self.account_lst))
            obj = cp.Maximize(profit_obj - cp.sum(varphi) * profit_old)

            constraints = [price >= 0.0575 * 0.8,
                           price <= 0.2215 * 1.2,
                           profit_obj >= profit_old,
                           varphi >= 0,
                           varphi <= 1]
            for i in I:
                load_val = load_matrix[i][i * 24:(i + 1) * 24]
                price_vals = price[i * 24:(i + 1) * 24]
                constraints += [load_val @ price_vals <=
                                rev_old.reshape(len(I), 1)[i] * (1 + varphi[i])]
        elif self.case_num == 'Case2':
            obj = cp.Maximize(profit_obj)

            constraints = [price >= 0.0575 * 0.8,
                           price <= 0.2215 * 1.2]
            for i in I:
                load_val = load_matrix[i][i * 24:(i + 1) * 24]
                price_vals = price[i * 24:(i + 1) * 24]
                constraints += [load_val @ price_vals <= rev_old.reshape(len(I), 1)[i]]
        elif self.case_num == 'Case3':
            varphi = cp.Variable(len(self.account_lst))
            c = cp.Variable()
            obj = cp.Maximize(profit_obj + c * profit_old -
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
        new_price_str = [x + f'_iteration{self.iteration + 1}' for x in self.account_lst]
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
        old_rev_t = np.trace(self.df_price.iloc[:len(self.account_lst), :].
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
