""" FullDataVis.py

Visualize the time series data as well as the age distributions. """
## Standard imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Custom global matplotlib parameters
## see http://matplotlib.org/users/customizing.html for details.
plt.rcParams["font.size"] = 20.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond","Time New Roman"]
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"

## Palette
#colors = ["#375E97","#FB6542","#FFBB00","#3F681C"]
#colors = ["#ff8700","#00ff07","#0078ff","#ff00f8"]
colors = ["#DF0000","#00ff07","#0078ff","#BF00BA"]


def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return axes

def GetRawData(data_dir="_data\\",fname="60measles.xls"):

	""" Use pandas excel I/O to get the dataset from the Excel files prepared by
	Kurt. """

	## Get each sheet individually
	cases = pd.read_excel(data_dir+fname,sheet_name="60measles",header=0)
	gps = pd.read_excel(data_dir+fname,sheet_name="60locations",header=0)
	births = pd.read_excel(data_dir+fname,sheet_name="60cities",header=0)

	## Kurt showed that the GPS needs a simple correction
	gps["LONGITUDE"] = -gps["LONGITUDE"]

	## Lower case column names
	cases.columns = [c.lower() for c in cases.columns]
	gps.columns = [c.lower() for c in gps.columns]
	births.columns = [c.lower() for c in births.columns]

	## And the city names
	births["city"] = births["city"].str.lower()
	gps["city"] = gps["city"].str.lower()

	return cases, gps, births

def GetAggregatedUKData(data_dir="_data\\",fname="60measles.xls"):

	## Get the 60 city UK data
	cases, _, births = GetRawData(data_dir,fname)
	population = births[["city","size"]]
	cities = population["city"]

	## Aggregate the case data accross cities
	cases["time"] = "19"+cases["year"].astype(str)+"_W"\
					 +("0"+cases["week"].astype(str)).str.slice(start=-2)
	df = pd.concat([cases["time"],
					cases[cities].sum(axis=1).rename("cases")],axis=1)

	## Do the same for the birth data
	births.columns = [c.replace("b","19") for c in births.columns]
	births = births.drop(columns=["city","size"]).sum(axis=0)
	df["births"] = np.round(df["time"].str.slice(stop=4).apply(births.get)/26).astype(np.int64)

	## And the total pop
	population = population["size"].sum()

	return population, df

def get_age_at_inf_distribution(data_dir="_data\\",fname="fine_clarkson_1982_scan.dat"):

	## Use python I/O
	df = []
	for line in open(data_dir+fname):
		if line.startswith("#"):
			continue
		df.append([np.float64(x) for x in line.split("\t")[:2]])
	df = pd.DataFrame(df,columns=["age","cumulative"])

	## Basic processing
	df["age"] = np.round(df["age"]*2)/2.

	## Compute the distribution
	df["frac"] = np.clip(df["cumulative"].diff().fillna(0),0,None)
	df["frac"] *= 1./(df["frac"].sum())

	## Move to left-age bins
	df["age"] = (df["age"]-0.5).astype(np.int64)
	df = df.loc[1:].reset_index(drop=True)

	return df[["age","frac"]]

def get_age_pyramid(fname="_data\\uk_age_pyramid_1950.csv"):

	## via pandas I/O
	df = pd.read_csv(fname,
					 header=0)

	## Some basic formatting
	df["age"] = df["Age"].apply(lambda s: s.split("-")[0].replace("+","")).astype(np.int64)
	df["total"] = df["M"]+df["F"]
	total = df["total"].sum()

	## Format, including moving coarsely to 1 year age bins, for alignment
	## with the regression.
	df = df[["age","total"]].set_index("age")["total"]
	df = df.reindex(np.arange(df.index[0],df.index[-1]+5)).fillna(method="ffill")/5
	df = df/total

	return df, total

def GetAnnualUKPop(pop60=None):
	total_uk_pop = pd.DataFrame([("1950",50616014),
								("1951",50601935),
								("1952",50651280),
								("1953",50750976),
								("1954",50890915),
								("1955",51063902),
								("1956",51265880),
								("1957",51495702),
								("1958",51754673),
								("1959",52045662),
								("1960",52370602),
								("1961",52727768),
								("1962",53109399),
								("1963",53500716),
								("1964",53882751),
								("1965",54240850),
								("1966",54568868)],columns=["time","population"])
	if pop60 is not None:
		to_60 = pop60/(total_uk_pop["population"].mean())
		total_uk_pop["population"] = (to_60*total_uk_pop["population"]).astype(np.int64)

	return total_uk_pop

if __name__ == "__main__":

	## Get the data
	population, df = GetAggregatedUKData()
	pop_over_time = GetAnnualUKPop(population)
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