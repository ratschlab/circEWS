import os
import pandas as pd
import numpy as np
import pdb
import math
import pickle
import glob
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import seaborn as sns
sns.set(style="whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42

#enviroment variables
data_path = "/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/"
data_dir = "figs_nm/Fig3/shapley_values/"
input_data_file = os.path.join(data_path, data_dir, "feature_shap_values.h5")

#load variable name
vid2string = np.load("/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/visualisation/mid2string_v6.npy").item()

#final_dataframe = pd.read_hdf(input_data_file)
print ("Loading data from disk")
final_dataframe = pd.read_hdf(input_data_file)

#neg_shap = pd.read_hdf(pos_shap_file,'table')
# variable_count = 500
# columns_neg = []
# print ("Printing neg graphs")
# i= 0
# while len(columns_neg) < variable_count:
# 	column = neg_shap.keys()[i]
# 	if column[:8] == "RawShap_":
# 		columns_neg.append(neg_shap.keys()[i])
# 	i = i + 1

columns_neg = ["static_Age", "high_1_vm5_min", "static_Height", "low_entire_pm48_mean", "low_0_pm91_mean", "med_0_pm120_mean", "high_0_vm1_min"
	, "high_3_vm1_trend", "med_0_vm2_max", "high_1_vm3_min", "high_1_vm22_min",
	"plain_vm28", "vm29_measure_density", "high_3_vm62_trend", "low_entire_vm131_median",
	"med_1_vm132_trend", "vm136_0_dist-set", "plain_vm139", "low_1_vm176_min",
	"low_1_vm176_trend", "low_2_vm185_min", "low_1_vm188_median"]

variable_names = ["Patient age", "Minimal MAP last 1h", "Patient height", "ACE inhibitors mean dosage of stay", "Mean steroid dosage last 16h", "Fraction of last 12h with muscle relaxants adm.", "Minimal heart rate last 30min"
	, "Heart rate trend last 12h", "Maximal Tcentral last 12h", "Minimal ABPsystolic last 1h", "Minimal respiratory rate last 1h",
	"Current RASS", "Measurement density intra cranial pressure", "Ventilator peak pressure trend last 12h", "Median weight entire stay",
	"Base excess (blood gas) trend last 24h", "Arterial lactate shaplet 1", "Current arterial lactate", "Minimal C-reactive protein last 32h",
	"C-reactive protein trend last 32h", "Minimal platlet count last 48h", "Median erythrocyte MCV last 32h"]

show_mean = [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

custom_x_lim =	{
  "high_2_vm3_iqr": [0,25],
  "high_3_vm3_iqr": [0,16],
  "high_2_vm4_iqr": [0,10],
}

i = 0
for column in columns_neg:
	variable_name = variable_names[i]
	i = i + 1
	variable_id = str(i)
	x_axis = "Value"
	# if bool(re.search('[vp]m[0-9]{1,3}', column)):
	# 	variable_name = vid2string[re.search('[vp]m[0-9]{1,3}', column).group()]
	# 	variable_id = re.search('[vp]m[0-9]{1,3}', column).group()

	# if bool(re.search('_[0-9]{1}_', column)):
	# 	lenght = re.search('_[0-9]{1}_', column).group()[1]
	# 	if lenght == "0":
	# 		variable_name = variable_name + " short horizon"
	# 	if lenght == "1":
	# 		variable_name = variable_name + " med horizon"
	# 	if lenght == "2":
	# 		variable_name = variable_name + " long horizon"
	# 	if lenght == "3":
	# 		variable_name = variable_name + " very long horizon"

	# if bool(re.search('trend', column)):
	# 	variable_name = "Trend " + variable_name
	# 	x_axis = "Change"

	# if bool(re.search('plain', column)):
	# 	variable_name = "Current value " + variable_name

	# if bool(re.search('min', column)):
	# 	variable_name = "Minimum " + variable_name

	# if bool(re.search('max', column)):
	# 	variable_name = "Maximum " + variable_name

	# if bool(re.search('mean', column)):
	# 	variable_name = "Mean " + variable_name

	# if bool(re.search('median', column)):
	# 	variable_name = "Median " + variable_name

	# if bool(re.search('dist-set', column)):
	# 	variable_name = "Shaplet " + variable_name
	# 	x_axis = "Distance"

	# if bool(re.search('time_to_last', column)):
	# 	variable_name = "Time since measurement " + variable_name
	# 	x_axis = "Time delta"

	# if bool(re.search('density', column)):
	# 	variable_name = "Measurment density " + variable_name
	# 	x_axis = "Measurment density"
 
	# if bool(re.search('Age', column)):
	# 	variable_name = "Patient Age"

	# if bool(re.search('entire', column)):
	# 	variable_name = "Entire stay " + variable_name

	# if bool(re.search('iqr', column)):
	# 	variable_name = "IQR " + variable_name
	# 	x_axis = "IQR"

	# if variable_name == "":
	# 	variable_name = column[8:]

	column = "RawShap_" + column
	print ("{0}: {1} ({2})".format(i, variable_name, column[8:]))
	row_selection = ~final_dataframe[column[8:]].isna() & ~final_dataframe[column].isna()

	fraction=2000/np.sum(row_selection)
	data_sample = final_dataframe.loc[row_selection, [column[8:],column]].sample(frac=fraction)
	data = data_sample.loc[:, column[8:]]
	y_data = data_sample.loc[:, column]

	bins = np.linspace(min(data), max(data), 25)
	# bins = []
	# bin_count = 25
	# for j in range(0,bin_count):
	# 	offset = 0
	# 	step = (100 - 2 * offset) / bin_count
	# 	percentile = offset + step * j
	# 	percentile_value = np.percentile(data, percentile)
	# 	#linear interpolation for a special case (plotting issue)
	# 	if (percentile_value == 0):
	# 		percentile_value = 0.000001 * j
	# 	bins.append(float(percentile_value))
	# 	j = j + 1

	digitized = np.digitize(data_sample.loc[:, column[8:]], bins)

	print ("plotting 1")
	with sns.axes_style("white"):
		g = sns.jointplot(x=data, y=y_data, marker="+", alpha=0.7)
		ax = g.ax_joint
		ax.axhline(y=0, color='grey', alpha=0.5)

		if show_mean[i-1]:
			print ("plotting 2")
			for j in range(1, len(bins)): data[digitized == j] = bins[j] 
			sns.lineplot(x=data, y=y_data, ax=ax, ci='sd', color='orange')

		print ("saving")
		g.fig.subplots_adjust(top=0.91);
		g.fig.suptitle("{0}".format(variable_name), fontsize=14)
		ax.set_xlabel(x_axis)
		ax.set_ylabel("SHAP value")

		if column[8:] in custom_x_lim:
			ax.set_xlim(custom_x_lim[column[8:]][0],custom_x_lim[column[8:]][1]) 

		g.savefig(os.path.join(data_path, data_dir, 'scatter_plots', variable_id + column[8:] + ".pdf"), dpi=300, format='pdf')
		g.savefig(os.path.join(data_path, data_dir, 'scatter_plots', variable_id + column[8:]), dpi=300)