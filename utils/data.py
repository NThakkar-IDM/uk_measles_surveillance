""" data.py

Functions to retrieve the data as pandas objects. """
import numpy as np
import pandas as pd

def get_uk_epi_data(data_dir="_data\\",fname="60measles.xls"):

	## Get the 60 city UK data
	cases, _, births = get_raw_uk60_data(data_dir,fname)
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

def get_raw_uk60_data(data_dir="_data\\",fname="60measles.xls"):

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

def get_annual_uk_population(pop60=None):
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