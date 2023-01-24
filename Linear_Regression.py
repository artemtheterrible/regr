import pandas as pd
import numpy as np

from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
from patsy import dmatrices

from math import sqrt, ceil
from itertools import combinations

import plotly.graph_objects as go
from plotly.subplots import make_subplots



class Linear_Regr():
	def __init__(self, results, df_data):
		# self.results = maybe_unwrap_results(results)
		self.results = results
		
		self.df_data = df_data
		self.indx = df_data.index

		self.y_true = self.results.model.endog
		self.y_predict = np.array(self.results.fittedvalues)
		self.xvar = self.results.model.exog
		self.xvar_names = self.results.model.exog_names
		self.yvar_name = self.results.model.endog_names

		self.residual = np.array(self.results.resid)
		influence = self.results.get_influence()
		self.residual_norm = influence.resid_studentized_internal
		self.leverage = influence.hat_matrix_diag
		self.build_df_main()
		
		# self.cooks_distance = influence.cooks_distance[0]
		# self.nparams = len(self.results.params)

	def formula (self):
		return self.results.model.formula


	def summary (self):
		return self.results.summary()


	def rse (self):
		return sqrt(self.results.mse_resid)


	def r_sq (self):
		return self.results.rsquared


	def conf_int_params (self):
		return self.results.conf_int()


	def params (self):
		return self.results.params


	def predict_df (self, new_data=None, includeIntervals=False):
		# If newData=None --> in-sample

		if new_data is None: 
			if includeIntervals:
				return self.results.get_prediction().summary_frame()
			else:
				return self.y_predict
		else:
			if includeIntervals:
				return self.results.get_prediction(new_data).summary_frame()
			else:
				return self.results.predict(new_data)

	def build_df_main (self):
		d = {'fitted': 'y_predict',
		    'resid': 'residual',
		    'std_resid': 'residual_norm',
		    'leverage': 'leverage'}
		df_res = pd.DataFrame({k:getattr(self, v) for k, v in d.items()},
		                      index = self.indx)
		y, X = dmatrices(self.formula(), self.df_data, return_type='dataframe')
		self.df_main = pd.concat([X,y,df_res],axis=1)


	def fig_residual(self, num_largest_residuals=5, standard_res_bool=False):

		res_col_name = 'std_resid' if standard_res_bool else 'resid'
		df = self.df_main[['fitted',res_col_name]].sort_values('fitted').copy()
		df['crv'] = lowess (df[res_col_name], df.fitted, is_sorted=True, return_sorted=False)

		largest_res = df.sort_values(by=res_col_name, key=abs, ascending=False)\
		              [:num_largest_residuals]

		x = df.fitted
		data=[go.Scatter(x=x, y=df[res_col_name], mode='markers',  text = df.index), 
		      go.Scatter(x=x, y=df.crv, mode='lines')]
		f=go.Figure(data)

		for indx, row in largest_res.iterrows():
		    f.add_annotation(x=row.fitted, y=getattr(row,res_col_name), 
		    				 text=indx, showarrow=True, arrowhead=1, opacity=0.9)

		if res_col_name == 'resid':
			title = 'Residuals vs Fitted'
			y_title = 'Residuals'
		else:
			title = 'Standardized Residuals vs Fitted'
			y_title = 'Standardized Residuals'

		f.update_layout(showlegend=False, title=title)
		f.update_xaxes(title='fitted values')
		f.update_yaxes(title=y_title)
		return f


	def fig_leverage(self, num_largest=5):

		df = self.df_main[['std_resid','leverage']].copy()

		largest_res_lev = df.sort_values(by='leverage', key=abs, ascending=False)\
		                  [:num_largest]
		largest_res_res = df.sort_values(by='std_resid', key=abs, ascending=False)\
		                  [:num_largest]

		data=[go.Scatter(x=df.leverage, y=df.std_resid, mode='markers',  text = df.index)]
		f=go.Figure(data)

		for indx, row in largest_res_lev.iterrows():
		    f.add_annotation(x=row.leverage, y=row.std_resid, 
		    				 text=indx, showarrow=True, arrowhead=1,
		    				 opacity=0.9, ax=20, ay=-30)

		for indx, row in largest_res_res.iterrows():
		    f.add_annotation(x=row.leverage, y=row.std_resid, 
		    				 text=indx, showarrow=True, arrowhead=1, font_color='red',
		    				 opacity=0.9, ax=-20, ay=-30)

		title = 'Residuals vs Leverage'
		y_title = 'Standardized Residuals'

		f.update_layout(showlegend=False, title=title)
		f.update_xaxes(title='Leverage')
		f.update_yaxes(title=y_title)
		return f


	def get_real_vars (self):

		def is_real_var (var):
		    if 'I(' in var: return False
		    if ':' in var: return False
		    if '**' in var: return False
		    if 'Intercept' in var: return False
		    if 'standardize' in var: return var
		    return var		
		
		real_var_names = []
		for var in self.xvar_names:
		    if is_real_var(var) != False: real_var_names.append(is_real_var(var))

		real_var_names = list(set(real_var_names))		
		return real_var_names


	def graph_outliers (self, num_largest, max_cols):

		large_lev_indx = self.df_main.sort_values('leverage', ascending=False).\
						 iloc[:num_largest].index

		l_xvars = self.get_real_vars()
		self.df_main['large_lev'] = False
		self.df_main.loc[large_lev_indx,'large_lev'] = True

		all_combo = list(combinations (l_xvars, 2))

		num_plots = len(all_combo)
		num_rows = ceil(num_plots / max_cols)
		f = make_subplots( rows=num_rows, cols=max_cols)
		
		l_d_ann = []
		ann_d_default = dict( showarrow=True, arrowhead=1, opacity=0.9, ax=20,ay=-20)
		cur_plot = 0

		for cur_row in range (1, num_rows+1):
		    for cur_col in range (1, max_cols+1):
		        cur_plot +=1
		        if cur_plot > num_plots: continue
		        x, y = all_combo[cur_plot-1]
		        
		        data=[]
		        df_reg = self.df_main.query('~large_lev')
		        data.append(go.Scatter(x=df_reg[x], y=df_reg[y], mode='markers', marker_size=3, 
		                               text=df_reg.index))
		        df_large_lev = self.df_main.query('large_lev')
		        data.append(go.Scatter(x=df_large_lev[x], y=df_large_lev[y], mode='markers', 
		                               marker_size=9, text = round(df_large_lev.leverage,3)))

		        for indx, row in df_large_lev.iterrows():
		            d = dict(x=getattr(row, x), y=getattr(row, y), text=indx, 
		                    xref=f'x{cur_plot}', yref=f'y{cur_plot}')
		            l_d_ann.append(ann_d_default|d)

		        f['layout'].update(annotations=l_d_ann)
		        for cnt in range(2):
		            f.add_trace(data[cnt], row=cur_row, col=cur_col)

		        f.update_xaxes(title_text=x, row=cur_row, col=cur_col)
		        f.update_yaxes(title_text=y, row=cur_row, col=cur_col)

		f.update_layout(showlegend=False)
		return f	


	def vif (self):
		vif_df = pd.DataFrame()
		vif_df["Features"] = self.xvar_names
		vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) 
		                        for i in range(self.xvar.shape[1])]
		return vif_df		





		