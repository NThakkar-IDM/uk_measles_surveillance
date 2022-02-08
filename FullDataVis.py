""" FullDataVis.py

Visualize the time series data as well as the age distributions. """
## Standard imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For getting the data
from utils.data import *

## Palette
#colors = ["#375E97","#FB6542","#FFBB00","#3F681C"]
#colors = ["#ff8700","#00ff07","#0078ff","#ff00f8"]
colors = ["#DF0000","#00ff07","#0078ff","#BF00BA"]

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return axes

if __name__ == "__main__":

	## Get the data
	population, df = get_uk_epi_data()
	pop_over_time = get_annual_uk_population(population)
	dist = get_age_at_inf_distribution()
	pyramid, _ = get_age_pyramid()
	
	## Add a population column for alignment
	pop_over_time["time"] += "_W02"
	df = df.merge(pop_over_time,how="left",on="time")
	df["population"] = df["population"].interpolate().fillna(method="bfill").astype(np.int64)

	## Make a full figure for everything
	fig = plt.figure(figsize=(12,6))
	case_ax = axes_setup(fig.add_subplot(2,6,(1,4)))
	birth_ax = axes_setup(fig.add_subplot(2,6,(7,10)))
	age_ax = axes_setup(fig.add_subplot(2,6,(5,6)))
	pyr_ax = axes_setup(fig.add_subplot(2,6,(11,12)))

	## Plot the cases
	case_ax.grid(color="grey",alpha=0.2)
	case_ax.plot(df["cases"]/1000,lw=3,color=colors[2])
	case_ax.set_ylim((0,None))
	case_ax.set_ylabel(r"Cases ($\times$1k)")
	case_ax.set_xticks(np.arange(0,len(df)+1,26*4))
	case_ax.set_xticklabels((np.arange(0,len(df)+1,26*4)/26 + 1944).astype(int))
	
	## And the yearly births
	birth_ax.grid(color="grey",alpha=0.2)
	birth_ax.plot(26*df["births"]/1000,lw=3,color=colors[0])
	birth_ax.set_ylabel(r"Births ($\times$1k)")
	birth_ax.set_xticks(np.arange(0,len(df)+1,26*4))
	birth_ax.set_xticklabels((np.arange(0,len(df)+1,26*4)/26 + 1944).astype(int))
	
	## Age at infection
	age_ax.grid(color="grey",alpha=0.2)
	age_ax.plot(dist["age"],100*dist["frac"],
				   lw=4,drawstyle="steps-post",color=colors[1])
	age_ax.set_ylim((0,None))
	age_ax.set_xticks(np.arange(0,25,5))
	age_ax.set_xlabel("Age (years)")
	age_ax.set_ylabel(r"% of cases")
	#age_ax.set_ylabel(r"Pr(age|inf.)")

	## Age pyramic
	pyr_ax.grid(color="grey",alpha=0.2)
	pyr_ax.plot(500*pyramid,
			  lw=4,drawstyle="steps-post",color=colors[3])
	pyr_ax.set_ylim((0,None))
	pyr_ax.set_yticks([0,2.5,5,7.5])
	pyr_ax.set_xticks(np.arange(0,101,20))
	pyr_ax.set_xlabel("Age (years)")
	pyr_ax.set_ylabel(r"% of pop.")
	#pyr_ax.set_ylabel(r"Pr(age)")

	## Details
	fig.tight_layout()
	case_ax.text(-0.18,1.,"a.",fontsize=18,color="k",transform=case_ax.transAxes)
	birth_ax.text(-0.18,1.,"b.",fontsize=18,color="k",transform=birth_ax.transAxes)
	age_ax.text(-0.4,1.,"c.",fontsize=18,color="k",transform=age_ax.transAxes)
	pyr_ax.text(-0.4,1.,"d.",fontsize=18,color="k",transform=pyr_ax.transAxes)
	fig.savefig("_plots\\full_data_vis.png")


	plt.show()