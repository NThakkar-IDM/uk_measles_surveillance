""" SurvivalAnalysis.py

With more concrete survival concepts + information like the age pyramid, let's
construct an slow-time-scale distribution that has nice uncertainty properties,
and leads to a better constrained transmission model. """
import sys

## Standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For getting the data
from utils.data import *

## For model fitting
from scipy.optimize import minimize

## For reference
#colors = ["#375E97","#FB6542","#FFBB00","#3F681C"]
colors = ["#DF0000","#00ff07","#0078ff","#BF00BA"]

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def fixed_s0_cost(theta):
	beta = 1./(1. + np.exp(-theta))
	f = beta*(S0*df["p_N_S0"]+df["exp_N_Bt"]).values
	ll = np.sum((f-df["phi_t"].values)**2)
	lp = np.dot(theta.T,np.dot(lam,theta))
	return ll+lp

def fixed_s0_grad_cost(theta):
	beta = 1./(1. + np.exp(-theta))
	f = beta*(S0*df["p_N_S0"]+df["exp_N_Bt"]).values
	grad = 2.*(f-df["phi_t"].values)*f*(1.-beta)
	grad += 2.* np.dot(lam,theta)
	return grad

def fixed_s0_hessian(theta):
	beta = 1./(1. + np.exp(-theta))
	f = (S0*df["p_N_S0"]+df["exp_N_Bt"]).values
	hess = np.diag(beta*(1.-beta)*f*((1.-2*beta)*(beta*f-df["phi_t"].values)+beta*(1-beta)*f))
	hess += lam
	return 2.*hess

if __name__ == "__main__":

	## Get the epi-data for the UK60 set
	population, df = get_uk_epi_data()

	## Move to annual time series to match the rough
	## resolution of the age distribution.
	df["time"] = df["time"].str.slice(stop=4)
	df = df.groupby("time").sum().reset_index()

	## Get annual population
	pop_series = get_annual_uk_population(population)
	df = df.merge(pop_series,how="left",on="time")
	df["population"] = df["population"].fillna(method="bfill").astype(np.int64)

	## Get the age-pyramid for calculation of the initial
	## susceptible population.
	pyramid, _ = get_age_pyramid()
	
	## Get the age at infection distribution, ripped from the
	## fine and clarkson, 1982 paper.
	dist = get_age_at_inf_distribution()
	dist["pr_mass"] = 1.*dist["frac"]
	dist["survival"] = 1.-np.cumsum(dist["pr_mass"])
	dist["hazard"] = dist["pr_mass"]/(dist["survival"]+dist["pr_mass"])
	dist["cumulative_hazard"] = np.cumsum(dist["hazard"])

	## Plot the key survival analysis objects
	fig, axes = plt.subplots(2,2,figsize=(12,8))
	for row in axes:
		for ax in row:
			axes_setup(ax)
			ax.grid(color="grey",alpha=0.2)

	## First is the probability mass function, aka the age distribution,
	## the probability of exit at time T.
	axes[0,0].plot(dist["age"],dist["pr_mass"],
				   lw=5,drawstyle="steps-post",color=colors[0])
	axes[0,0].set_ylim((0,None))
	axes[0,0].set_xlabel("Age (years)")
	axes[0,0].set_ylabel("P(age | infection)")

	## Then, the survival probability, aka the pr(T>=t), i.e. the prob that
	## death happens later, is 1 - the CMF assocatied with P(T=t)
	axes[0,1].plot(dist["age"],dist["survival"],
				   lw=5,drawstyle="steps-post",color=colors[1])
	axes[0,1].set_ylim((0,None))
	axes[0,1].set_xlabel("Age (years)")
	axes[0,1].set_ylabel("Survival function")

	## The hazard function is therefor P(T=t)/(P(T>= t)), aka
	## the PMF/Survival
	axes[1,0].plot(dist["age"],dist["hazard"],
				   lw=5,drawstyle="steps-post",color=colors[2])
	axes[1,0].set_ylim((0,None))
	axes[1,0].set_xlabel("Age (years)")
	axes[1,0].set_ylabel("Hazard function")

	## Finally accumulate hazard to construct the cuml. hazard function
	axes[1,1].plot(dist["age"],dist["cumulative_hazard"],
				   lw=5,drawstyle="steps-post",color=colors[3])
	axes[1,1].set_ylim((0,None))
	axes[1,1].set_xlabel("Age (years)")
	axes[1,1].set_ylabel("Cuml. hazard function")

	## Finish the details
	fig.tight_layout()
	fig.savefig("_plots\\uk_implied_survival.png")

	## Plot the age pyramid as well
	fig, axes = plt.subplots(1,2,figsize=(12,4.5))
	for ax in axes:
		axes_setup(ax)
		ax.grid(color="grey",alpha=0.2)
	axes[0].plot(dist["age"],dist["survival"],
				   lw=5,drawstyle="steps-post",color="#f9320c")
	axes[0].set_ylim((0,None))
	axes[0].set_xlabel("Age (years)")
	axes[0].set_ylabel("Survival function")
	axes[1].plot(500*pyramid,
			  lw=5,drawstyle="steps-post",color="#7200da")
	axes[1].set_ylim((0,None))
	axes[1].set_xlabel("Age (years)")
	axes[1].set_ylabel("Percent of population")
	fig.tight_layout()
	fig.savefig("_plots\\uk_age_pyramid.png")

	## Calculate 1944 population surviving, i.e. initially
	## susceptible. Variance here comes from binomial uncertainty and
	## a coarse approximation to uncertainty in the population estimate via MoE in
	## the ACS. 
	initial_pop	= df.iloc[0].loc["population"]
	S0 = (pyramid*initial_pop*dist["survival"]).fillna(0).sum()
	#S0_var = (pyramid*initial_pop*dist["survival"]*(1.-dist["survival"])).fillna(0).sum()
	S0_var = (initial_pop*pyramid*dist["survival"]*(1.-dist["survival"])+\
			 (0.02*initial_pop*pyramid*dist["survival"])**2).fillna(0).sum()
	S0_std = np.sqrt(S0_var)

	## Use the yearly hazard function to map birth-cohorts to expected time of 
	## infection. Start by creating the P(S|T) matrix, which maps birth time (T) to
	## exit time (S). 
	p_steps = dist["frac"].reindex(df.index).fillna(0)
	PST = np.tril(np.array([np.roll(p_steps.values,i) for i in np.arange(len(p_steps))]).T)
	exp_N_Bt = pd.Series(np.dot(PST,df.loc[0:,"births"].values),
						 index=df.loc[0:,"births"].index,name="exp_N_Bt")
	
	## Compute the contribution from S0
	p_a_in_S0 = p_steps.copy()
	#p_a_in_S0 = (dist["survival"]*pyramid).reindex(p_steps.index).fillna(0)
	p_a_in_S0 *= 1./p_a_in_S0.sum()
	fig, axes = plt.subplots(figsize=(6,4.5))
	axes_setup(axes)
	axes.grid(color="grey",alpha=0.2)
	axes.plot(p_a_in_S0,
			  lw=5,drawstyle="steps-post",color=colors[0])
	axes.plot(p_steps,
			  lw=5,drawstyle="steps-post",color=colors[1])
	axes.set_ylim((0,None))
	axes.set_xlabel("Age (years)")
	axes.set_ylabel(r"$p(a|\in S_0)$")
	fig.tight_layout()
	p_N_S0 = np.array([np.roll(p_steps.values,-i) for i in np.arange(len(p_steps))])
	p_N_S0 = p_N_S0 - np.rot90(np.triu(np.rot90(p_N_S0),k=1),k=3)
	p_N_S0 = pd.Series(np.dot(p_N_S0,p_a_in_S0.values),index=p_steps.index,name="p_N_S0")
	fig, axes = plt.subplots(figsize=(8,6))
	axes_setup(axes)
	axes.grid(color="grey",alpha=0.2)
	axes.plot(p_N_S0,color="k",lw=5,drawstyle="steps-post")
	axes.plot(dist["age"],dist["survival"],#/(dist["survival"].sum()),
			  lw=5,drawstyle="steps-post",color=colors[2])
	axes.set_ylim((0,None))
	axes.set_ylabel(r"P(infection|$\in$ S$_0$)")
	axes.set_xlabel("Time (years)")
	fig.tight_layout()
	fig.savefig("_plots\\uk_S0_exit_time_vs_survival.png")

	## Add to the data frame
	df = pd.concat([df,exp_N_Bt,p_N_S0],axis=1)
	df["phi_t"] = df["cases"].copy()

	## Compute the basic least-squares result, to get initial
	## guesses
	X = df[["exp_N_Bt"]].values+S0*df[["p_N_S0"]].values
	Y = (df["phi_t"]).values
	C = np.linalg.inv(np.dot(X.T,X))
	r = np.dot(C,np.dot(X.T,Y))[0]
	resid = Y-r*X[:,0]
	RSS = np.sum(resid**2)
	cov = C*RSS/len(Y)
	std = np.sqrt(np.diag(cov))
	r_std = std[0]
	print("\nVia standard least squares...")
	print("Reporting rate = {} +/- {}".format(r,2.*r_std))
	print("Survival S = {} +/- {}".format(S0,2.*S0_std))

	## Use a sampling approach to quantify uncertainty
	r_samples = np.random.normal(r,std[0],size=(1000,))
	S0_samples = np.random.normal(S0,S0_std,size=(len(r_samples)))
	fit_samples = r_samples*(df["exp_N_Bt"].values[:,np.newaxis]+S0_samples*df["p_N_S0"].values[:,np.newaxis])
	lst_sq_fit_low = np.percentile(fit_samples,2.5,axis=1)
	lst_sq_fit_high = np.percentile(fit_samples,97.5,axis=1)
	lst_sq_fit_mid = r*(df["exp_N_Bt"] + S0*df["p_N_S0"])

	## Test plot
	fig, axes = plt.subplots(figsize=(12,6))
	axes_setup(axes)
	axes.fill_between(df.index,0,df["cases"].values,
					  alpha=0.5,edgecolor="None",facecolor="grey")
	axes.plot(df["cases"],lw=4,color="grey",label="Observed cases by rash date")
	axes.fill_between(df.index,lst_sq_fit_low,lst_sq_fit_high,alpha=0.3,facecolor=colors[0],edgecolor="None")
	axes.plot(lst_sq_fit_mid,color=colors[0],lw=3,label="Initial guess for the regression")
	h,l =  axes.get_legend_handles_labels()
	axes.legend(h,l,frameon=False,fontsize=20)
	axes.set_ylabel("Number of people")
	axes.set_ylim((0,None))
	axes.set_xlabel("Time since {} (years)".format(df["time"].min()))
	fig.tight_layout()

	## Set up the regularization matrix for the random walk
	T = len(df)
	D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
	D2[0,2] = 1
	D2[-1,-3] = 1
	lam = np.dot(D2.T,D2)*((5**4)/8.)*(df["cases"].var())#/(np.pi**4)

	## Set up cost function to be passed to scipy.minimize, and
	## it's gradient to boost efficiency and stability. Then
	## solve the regression problem.
	x0 = np.array(len(df)*[np.log(r/(1.-r))])
	result = minimize(fixed_s0_cost,
					  x0=x0,
					  jac=fixed_s0_grad_cost,
					  #hess=fixed_s0_hessian,
					  method="BFGS",
					  )
	#print(result)
	df["rr"] = 1./(1. + np.exp(-result["x"]))
	df["fit"] = (df["rr"]*(df["exp_N_Bt"]+S0*df["p_N_S0"])).values
	sig_nu2 = np.sum((df["phi_t"]-df["fit"])**2)/len(df)
	hess = fixed_s0_hessian(result["x"])/sig_nu2
	cov = np.linalg.inv(hess)
	#cov = result["hess_inv"]
	df["rr"] = 1./(1. + np.exp(-result["x"]))
	df["rr_var"] = (np.diag(cov))*((df["rr"]*(1.-df["rr"]))**2)
	df["rr_std"] = np.sqrt(df["rr_var"])
	df["S0"] = S0*np.ones((T,))
	df["S0_var"] = S0_var*np.ones((T,))
	print("\nSo we find...")
	print("Reporting rate:")
	print(df[["time","rr","rr_std"]])

	## Compute the variance in the fits via sampling
	samples = np.random.multivariate_normal(result["x"],cov,size=(len(S0_samples),))
	samples = 1./(1. + np.exp(-samples))
	samples = samples*(df["exp_N_Bt"].values[np.newaxis,:]+S0_samples[:,np.newaxis]*df["p_N_S0"].values[np.newaxis,:])
	fit_low = np.percentile(samples,2.5,axis=0)
	fit_high = np.percentile(samples,97.5,axis=0)
	
	## Plot the fit and inference
	fig, axes = plt.subplots(2,1,sharex=True,figsize=(12,8))
	for ax in axes:
		axes_setup(ax)
	#axes[0].fill_between(df.index,(S0-2.*S0_std)*df["p_N_S0"]*df["rr"],(S0+2.*S0_std)*df["p_N_S0"]*df["rr"],
	#					 facecolor=colors[0],edgecolor="None",alpha=0.4,zorder=0)
	#axes[0].plot(S0*df["p_N_S0"]*df["rr"],color=colors[0],lw=4,label=r"Contribution from S$_0$",zorder=1)
	axes[0].fill_between(df.index,fit_low/1000,fit_high/1000,
						 facecolor="k",edgecolor="None",alpha=0.25,zorder=2)
	axes[0].plot(df["fit"]/1000,color="k",lw=4,label="Slow-timescale model",zorder=3)
	#axes[0].plot(df.index,lst_sq_fit_low,color="grey",ls="dashed",lw=2)
	#axes[0].plot(df.index,lst_sq_fit_high,color="grey",ls="dashed",lw=2)
	#axes[0].plot(lst_sq_fit_mid,color="grey",ls="dashed",lw=3,label="Constant reporting rate model")
	axes[0].plot(df["phi_t"]/1000,color=colors[2],lw=4,label="Observed cases",zorder=4)
	axes[0].set_ylabel(r"Yearly cases ($\times$1k)")
	#axes[0].set_ylim((0,None))
	axes[0].legend(loc=2,frameon=False,fontsize=20)
	axes[1].fill_between(df["rr"].index,100*(df["rr"]-2.*df["rr_std"]),100*(df["rr"]+2.*df["rr_std"]),
						 facecolor=colors[3],edgecolor="None",alpha=0.2)
	axes[1].fill_between(df["rr"].index,100*(df["rr"]-df["rr_std"]),100*(df["rr"]+df["rr_std"]),
						 facecolor=colors[3],edgecolor="None",alpha=0.4)
	axes[1].plot(100*df["rr"],color=colors[3],lw=6,label="Estimated reporting rate")
	#axes[1].axhline(0.23,ls="dashed",color="grey",lw=2)
	axes[1].axhline(100*r,ls="dashed",color="k",lw=3,label="Constant reporting model")
	#axes[1].axhline(100*(r-r_std),ls="dashed",color="grey",lw=2)
	#axes[1].axhline(100*(r+r_std),ls="dashed",color="grey",lw=2)
	axes[1].legend(frameon=False,fontsize=20)
	axes[1].set_ylabel("Reporting rate (%)")
	axes[1].set_xticks(np.arange(0,21,5))
	axes[1].set_xticklabels(np.arange(0,21,5)+int(df["time"].min()))
	#print(yticks)
	#sys.exit()
	#axes[1].set_xlabel("Time since {} (years)".format(df["time"].min()))
	#axes[0].set_ylim((0,None))
	#axes[1].set_ylim((0,None))
	fig.tight_layout()
	fig.savefig("_plots\\uk_coarse_inference.png")

	## Create an output df for use in the fine-timescale inference
	## procedures.
	print("\nOutput to pickle:")
	print(df)
	df.to_pickle("pickle_jar\\coarse_prior.pkl")
	
	plt.show()