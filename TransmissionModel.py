""" TransmissionModel.py

Fitting a transmission model to the UK60 data using the coarse regression as an input
based on age-at-infection information. """
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For data i/o
from utils.data import *

## For model fitting
from utils.models import StrongPriorModel
from scipy.optimize import minimize

## For goodness of fit
from sklearn.metrics import r2_score

## For reference
colors = ["#375E97","#FB6542","#FFBB00","#3F681C"]
#colors = ["#0078ff","#DF0000","#BF00BA","#00DF06"]

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def low_mid_high(samples):
	l0 = np.percentile(samples,2.5,axis=0)
	h0 = np.percentile(samples,97.5,axis=0)
	l1 = np.percentile(samples,25.,axis=0)
	h1 = np.percentile(samples,75.,axis=0)
	m = samples.mean(axis=0) 
	return l0, l1, m, h1, h0

def fit_quality(data,samples,verbose=True):

	## Compute a summary
	l0,l1,m,h1,h0 = low_mid_high(samples)

	## Compute scores
	score = r2_score(data,m)
	score50 = len(data[np.where((data >= l1) & (data <= h1))])/len(m)
	score95 = len(data[np.where((data >= l0) & (data <= h0))])/len(m)
	if verbose:
		print("R2 score = {}".format(score))
		print("With 50 interval: {}".format(score50))
		print("With 95 interval: {}".format(score95))
	
	return score, score50, score95

if __name__ == "__main__":

	## Get the data
	population, df = get_uk_epi_data()
	pop_over_time = get_annual_uk_population(population)

	## Add a population column
	pop_over_time["time"] += "_W02"
	df = df.merge(pop_over_time,how="left",on="time")
	df["population"] = df["population"].interpolate().fillna(method="bfill").astype(np.int64)

	## Plot the cases
	fig, axes = plt.subplots(figsize=(12,6))
	axes_setup(axes)
	axes.spines["left"].set_color(colors[0])
	axes.fill_between(df.index,0,df["cases"].values,
					  alpha=0.2,edgecolor="None",facecolor=colors[0])
	axes.plot(df["cases"],lw=4,color=colors[0])

	## Twin the axes and set up those spines
	axes2 = axes.twinx()
	axes2.spines["right"].set_position(("axes",1.025))
	axes2.spines["top"].set_visible(False)
	axes2.spines["left"].set_visible(False)
	axes2.spines["bottom"].set_visible(False)
	axes2.spines["right"].set_color(colors[1])
	axes2.plot(df["births"],lw=4,color=colors[1])

	## Details
	axes.set_ylabel("Reported measles cases",color=colors[0],labelpad=15)
	axes2.set_ylabel("Biweekly births",color=colors[1],labelpad=15)
	axes.set_ylim((0,None))
	axes.tick_params(axis="y",colors=colors[0])
	axes2.tick_params(axis="y",colors=colors[1])
	axes.set_xlabel("Time since 1944 (biweeks)")
	fig.tight_layout()
	fig.savefig("_plots\\fine_time_uk60_data.png")

	## Get the prior estimates at the annual scale
	coarse_prior = pd.read_pickle("pickle_jar\\coarse_prior.pkl")
	initial_S0 = coarse_prior.loc[0,"S0"]
	initial_S0_var = coarse_prior.loc[0,"S0_var"]#*500
	rr_prior = coarse_prior[["time","rr","rr_var"]].copy()
	rr_prior.columns = ["time","rr_p","rr_p_var"]
	
	## And interpolate them to the fine scale
	rr_prior["time"] += "_W26"
	df = df.merge(rr_prior,how="left",on="time")
	df[["rr_p","rr_p_var"]] = df[["rr_p","rr_p_var"]].interpolate().fillna(method="bfill")

	## Construct the posterior class and use it to solve the
	## inference problem
	neglp = StrongPriorModel(df,
							 initial_S0,
							 initial_S0_var,
							 beta_corr=3.)
	result = minimize(neglp,
					  jac=neglp.grad,
					  x0=neglp.theta0,
					  method="L-BFGS-B",
					  #bounds=[(None,None)]+(neglp.T+1)*[(1.e-4,1.)],
					  options={"maxfun":600000,
					  		   "maxiter":600000,
					  		   "ftol":1e-15,
					  		   "maxcor":100,
					  		   })

	## Let's talk about the results
	print("\nPeriodic enforcement implies...")
	print(result)
	cov = result["hess_inv"].todense()
	var = np.diag(cov)
	std = np.sqrt(var)
	print("Initial log S0 = {} +/- {}".format(neglp.theta0[0],np.sqrt(neglp.logS0var)))
	print("Converged log S0 = {} +/- {}".format(result["x"][0],std[0]))
	print("Which is {} +/- {}".format(np.exp(result["x"][0]),
									  np.exp(result["x"][0])*std[0]))

	## Use the result to finish model specification
	#result["x"][1:] = result["x"][1:].mean() 
	adj_cases = ((df["cases"].values+1.)/result["x"][1:])-1.
	E_t = adj_cases[1:]
	I_t = adj_cases[:-1]
	S_t = np.exp(result["x"][0])+np.cumsum(df["births"].values[:-1]-E_t)
	pop = df["population"].values[:-1]
	
	## Fit the transmission rate model
	Y_t = np.log(E_t)-np.log(S_t)
	X = np.hstack([neglp.X,np.log(I_t)[:,np.newaxis]])
	pRW2 = np.zeros((X.shape[1],X.shape[1]))
	pRW2[:-1,:-1] = neglp.pRW2
	C = np.linalg.inv(np.dot(X.T,X)+pRW2)
	beta_hat = np.dot(C,np.dot(X.T,Y_t))
	beta_t = np.dot(X,beta_hat)
	RSS = np.sum((Y_t-beta_t)**2)
	sig_eps = np.sqrt(RSS/neglp.T)
	print("sig_eps = {}".format(sig_eps))
	beta_cov = sig_eps*sig_eps*C
	beta_var = np.diag(beta_cov)
	beta_std = np.sqrt(beta_var)
	beta_t_std = np.sqrt(np.diag(np.dot(X,np.dot(beta_cov,X.T))))
	inf_seasonality = np.exp(beta_hat[:-1])
	inf_seasonality_std = np.exp(beta_hat[:-1])*beta_std[:-1]
	alpha = beta_hat[-1]
	alpha_std = beta_std[-1]
	print("alpha = {} +/- {}".format(alpha,2.*alpha_std))

	## Sample model trajectories
	num_samples = 10000
	eps_t = np.exp(sig_eps*np.random.normal(size=(num_samples,len(df))))
	S0_samples = np.random.normal(np.exp(result["x"][0]),
								  np.exp(result["x"][0])*std[0],
								  size=(num_samples,))
	traj_long = np.zeros((num_samples,2,len(df)))
	traj_long[:,0,0] = S0_samples
	traj_long[:,1,0] = I_t[0]
	traj_short = np.zeros((num_samples,2,len(df)))
	traj_short[:,0,0] = S0_samples
	traj_short[:,1,0] = I_t[0]
	skeleton = np.zeros((2,len(df)))
	skeleton[0,0] = np.exp(result["x"][0])
	skeleton[1,0] = I_t[0]
	for t in range(1,len(df)):

		## Get the transmission rate
		beta = inf_seasonality[(t-1)%26]

		## Compute the force of infection in each case
		lam_long = beta*traj_long[:,0,t-1]*(traj_long[:,1,t-1]**alpha)
		lam_short = beta*traj_short[:,0,t-1]*(I_t[t-1]**alpha)

		## Incorporate uncertainty across samples
		traj_long[:,1,t] = lam_long*eps_t[:,t-1]
		traj_long[:,0,t] = traj_long[:,0,t-1]+df.loc[t-1,"births"]-traj_long[:,1,t]
		traj_short[:,1,t] = lam_short*eps_t[:,t-1]
		traj_short[:,0,t] = traj_short[:,0,t-1]+df.loc[t-1,"births"]-traj_short[:,1,t]

		## Fill in the mean
		skeleton[1,t] = beta*skeleton[0,t-1]*(skeleton[1,t-1]**alpha)*np.exp(0.5*(sig_eps**2))
		skeleton[0,t] = skeleton[0,t-1]+df.loc[t-1,"births"]-skeleton[1,t]

		## Regularize for the 0 boundary 
		traj_long[:,:,t] = np.clip(traj_long[:,:,t],0.,None)
		traj_short[:,:,t] = np.clip(traj_short[:,:,t],0.,None)

	## Sample to get estimates of observed cases
	cases_short = np.random.binomial(np.round(traj_short[:,1,:]).astype(int),
									 p=result["x"][1:])

	## Test the goodness of fit
	print("\nGoodness of fit to cases:")
	fit_quality(df["cases"].values,cases_short)

	## Compute Reff
	classic_reff = (df["cases"]/(df["cases"].shift(1)**alpha)).dropna().groupby(lambda s: (s-1)%26).mean()
	model_reff = pd.Series(traj_short[:,1,:].mean(axis=0),index=df.index)
	model_reff = (model_reff/(model_reff.shift(1))).dropna().groupby(lambda s: (s-1)%26).mean()

	## Convert to per population metrics
	traj_long = 100*traj_long/(df["population"].values[np.newaxis,:])
	traj_short = 100*traj_short/(df["population"].values[np.newaxis,:])
	skeleton = 100*skeleton/(df["population"].values[np.newaxis,:])
	cases_short = cases_short/1000.

	## Summarize the result
	long_low, _, long_mid, _, long_high = low_mid_high(traj_long)
	short_low, _, short_mid, _, short_high = low_mid_high(traj_short)
	cases_low, _, cases_mid, _, cases_high = low_mid_high(cases_short)

	## Plot the results
	fig, axes = plt.subplots(4,1,sharex=True,figsize=(11,10))
	for ax in axes:
		axes_setup(ax)
	axes[0].fill_between(np.arange(len(df)),cases_low,cases_high,
						 alpha=0.3,facecolor=colors[3],edgecolor="None")
	axes[0].plot(cases_mid,color=colors[3],lw=3,label="Model fit")
	axes[0].plot(df["cases"]/1000,color="k",ls="None",
				 marker=".",markerfacecolor="k",markeredgecolor="k",markersize=5,
				 label="Data")
	axes[0].set_ylim((0,None))
	legend = axes[0].legend(loc=1,frameon=True,fontsize=18)
	legend.get_frame().set_linewidth(0.0)
	axes[0].set_ylabel(r"Cases ($\times$1k)")
	#axes[1].fill_between(np.arange(len(df)),long_low[1],long_high[1],
	#					 alpha=0.3,facecolor="grey",edgecolor="None")
	#axes[1].plot(long_mid[1],color="grey",lw=3)
	axes[1].fill_between(np.arange(len(df)),short_low[1],short_high[1],
						 alpha=0.3,facecolor=colors[1],edgecolor="None")
	axes[1].plot(short_mid[1],color=colors[1],lw=3)
	#axes[1].plot(short_mid[1],color="k",lw=3)
	#axes[1].plot(skeleton[1],color="k",lw=3)
	#axes[1].plot(100*I_t/pop,color=colors[1],lw=3)#,ls="dashed")
	axes[1].set_ylim((0,None))
	axes[1].set_ylabel(r"I$_t$ (%)")
	#axes[2].fill_between(np.arange(len(df)),long_low[0],long_high[0],
	#					 alpha=0.3,facecolor="grey",edgecolor="None")
	#axes[2].plot(long_mid[0],color="grey",lw=3)
	axes[2].fill_between(np.arange(len(df)),short_low[0],short_high[0],
						 alpha=0.3,facecolor=colors[0],edgecolor="None")
	axes[2].plot(short_mid[0],color=colors[0],lw=3)
	#axes[2].plot(skeleton[0],color="k",lw=4)
	#axes[2].plot(100*S_t/pop,color=colors[0],lw=3)
	#axes[2].plot(100*S2/pop,color="k",lw=2,ls="dashed")
	axes[2].set_ylabel(r"S$_t$ (%)")
	axes[3].plot(100*result["x"][1:],color=colors[2],lw=3)
	axes[3].plot(100.*df["rr_p"],color="grey",ls="dashed",lw=3)
	std = np.sqrt(df["rr_p_var"])
	axes[3].plot(100.*(df["rr_p"]-2.*std),color="grey",ls="dashed",lw=2)
	axes[3].plot(100.*(df["rr_p"]+2.*std),color="grey",ls="dashed",lw=2)
	#axes[3].set_ylim((0.,100.))
	axes[3].set_ylabel(r"r$_t$ (%)")
	axes[3].set_xticks(np.arange(0,len(df)+1,26*4))
	axes[3].set_xticklabels((np.arange(0,len(df)+1,26*4)/26 + 1944).astype(int))
	#axes[3].set_xlabel("Time since 1944 (biweeks)")
	fig.tight_layout()
	fig.savefig("_plots\\strong_prior.png")

	## A more complex figure for the reporting rate zoom
	## and other inferences.
	fig = plt.figure(figsize=(12,8))
	rr_ax = fig.add_subplot(2,2,(3,4))
	axes_setup(rr_ax)
	reff_ax = fig.add_subplot(2,2,(1,1))
	axes_setup(reff_ax)
	acc_ax = fig.add_subplot(2,2,(2,2))
	axes_setup(acc_ax)

	## Reporting rate timeline
	#rr_ax.spines["left"].set_color("#ff5f2e")
	rr_ax.plot(100*result["x"][1:],color="#FF3A2E",lw=4,zorder=5)
	#axes.plot(100*result["x"][1+2*26:],color=colors[2],lw=3,zorder=6)

	## Add annotations
	rr_ax.axvline(4.75,ymax=0.87,color="k",ls="dashed",lw=1) ## March 2, 1944
	rr_ax.text(4.75,83.,"NHS white paper\nis officially endorsed",
			  horizontalalignment="left",verticalalignment="bottom",
			  color="k",fontsize=18)
	rr_ax.axvline(13.5+26,ymax=0.66,color="k",ls="dashed",lw=1) ## July 6, 1945
	rr_ax.text(13.5+26,70.,"Labour Party wins\nthe general election",
			  horizontalalignment="center",verticalalignment="bottom",
			  color="k",fontsize=18)
	rr_ax.axvline(22.5+2*26,ymax=0.79,color="k",ls="dashed",lw=1) ## November 6, 1946
	rr_ax.text(22.5+2*26,78.,"NHS act of 1946\ngets royal assent",
			  horizontalalignment="center",verticalalignment="bottom",
			  color="k",fontsize=18)
	rr_ax.axvline(14+4*26,ymax=0.66,color="k",ls="dashed",lw=1) ## July 5, 1948
	rr_ax.text(14+4*26,70.,"NHS is established",
			  horizontalalignment="right",verticalalignment="bottom",
			  color="k",fontsize=18)

	## Twin the axes and set up those spines
	axes2 = rr_ax.twinx()
	axes2.spines["right"].set_visible(False)#.set_position(("axes",1.025))
	axes2.spines["top"].set_visible(False)
	axes2.spines["left"].set_visible(False)
	axes2.spines["bottom"].set_visible(False)
	axes2.spines["right"].set_color("#8283a7")
	axes2.fill_between(np.arange(len(df)),0,short_mid[1],
					   alpha=0.3,facecolor="grey",edgecolor=None,zorder=2)

	## Details
	rr_ax.set_ylabel("Reporting rate (%)")#,color="#ff5f2e",labelpad=15)
	#axes2.set_ylabel("Prevalence (%)",color="#8283a7",labelpad=15)
	axes2.set_ylim((0,None))
	#rr_ax.tick_params(axis="y",colors="#ff5f2e")
	#axes2.tick_params(axis="y",colors="#8283a7")
	axes2.set_yticks([])
	rr_ax.set_zorder(axes2.get_zorder()+1)
	rr_ax.patch.set_visible(False)
	rr_ax.set_xticks(np.arange(0,len(df)+1,26*1))
	rr_ax.set_xticklabels((np.arange(0,len(df)+1,26*1)/26 + 1944).astype(int))
	rr_ax.set_xlim((0,26*6))
	rr_ax.set_ylim((30,90))

	## Seasonality panel
	reff_ax.axvspan(0,0.5,
				 facecolor="grey",edgecolor="None",alpha=0.2)
	reff_ax.axvspan(13.5,17.5,
				 facecolor="grey",edgecolor="None",alpha=0.2)
	reff_ax.axvspan(6.5,7.5,
				 facecolor="grey",edgecolor="None",alpha=0.2)
	reff_ax.axvspan(21.5,22.5,
				 facecolor="grey",edgecolor="None",alpha=0.2)
	reff_ax.axvspan(24.5,25,
				 facecolor="grey",edgecolor="None",alpha=0.2,label="School holiday")
	#reff_ax.axhline(1,color="k",lw=3)
	S_bar = S_t.mean()#np.exp(result["x"][0])
	reff_ax.fill_between(np.arange(len(inf_seasonality)),
					  (S_bar)*(inf_seasonality-2.*inf_seasonality_std),
					  (S_bar)*(inf_seasonality+2.*inf_seasonality_std),
					  facecolor="#80ACFF",edgecolor="None",alpha=0.75,zorder=2)
	reff_ax.plot(S_bar*inf_seasonality,color="#004FDF",lw=4,zorder=3,label="Model")
	#reff_ax.fill_between(np.arange(len(inf_seasonality)),
	#				  (model_reff.values-2.*S_bar*inf_seasonality_std),
	#				  (model_reff.values+2.*S_bar*inf_seasonality_std),
	#				  facecolor="#80ACFF",edgecolor="None",alpha=0.75,zorder=2)
	#reff_ax.plot(model_reff.values,color="#004FDF",lw=4,zorder=3,label="Model")
	reff_ax.plot(classic_reff.values,color="k",lw=3,ls="dashed",label="Ref. 7")
	reff_ax.set_ylabel(r"Biweekly R$_e$")
	reff_ax.set_xlabel("Time of year (biweek)")
	reff_ax.set_xlim((0,25))
	reff_ax.legend(loc=3,frameon=False,fontsize=14)

	## Bifurcation panel
	## Calculate the bifurcation diagram as a function of biweekly
	## births
	num_rates = 1000
	num_steps = 26*200
	births = np.linspace(0.1e3,3.e4,num_rates)
	trajectories = np.zeros((2,num_steps,num_rates))
	trajectories[0,0,:] = np.exp(result["x"][0])
	trajectories[1,0,:] = I_t[0]
	for t in range(1,num_steps):
		beta = inf_seasonality[(t-1)%26]
		trajectories[1,t,:] = beta*trajectories[0,t-1,:]*(trajectories[1,t-1,:]**alpha)*np.exp(0.5*(sig_eps**2))
		trajectories[0,t,:] = trajectories[0,t-1,:]+births-trajectories[1,t,:]
	trajectories = trajectories[:,26*100:,:]

	## Plot the results
	i_t_10 = (100*trajectories[1,10::26,:]/(df["population"].mean()))
	s_t_10 = (100*trajectories[0,10::26,:]/(df["population"].mean()))
	i_t_20 = (100*trajectories[1,15::26,:]/(df["population"].mean()))
	s_t_20 = (100*trajectories[0,15::26,:]/(df["population"].mean()))
	#i_t = (100*trajectories[1,:,:]/(df["population"].mean())).reshape((26,100,1000)).max(axis=0)
	b_t = np.array(i_t_10.shape[0]*[births])		
	i_t_10 = i_t_10.reshape(-1)
	s_t_10 = s_t_10.reshape(-1)
	i_t_20 = i_t_20.reshape(-1)
	s_t_20 = s_t_20.reshape(-1)
	b_t = b_t.reshape(-1)
	acc_ax.grid(color="grey",alpha=0.2)
	#acc_ax.errorbar(s_t,i_t,
	#				yerr=2.*i_t*(np.exp(1.5*(sig_eps**2))-np.exp(0.5*(sig_eps**2))),
	#				ls="None",
	#				lw=1,color="#FFA32E",alpha=1)
	acc_ax.plot(s_t_20,i_t_20,c="#A600FF",ls="None",marker=".")
	acc_ax.plot(s_t_10,i_t_10,c="#FFA32E",ls="None",marker=".")
	acc_ax.set_xlabel(r"S$_t$ (%)")
	acc_ax.set_ylabel(r"I$_t$ (%)")
	acc_ax.set_ylim((0,None))
	acc_ax.set_xlim((4.75,10.25))
	acc_ax.set_xticks([5,6,7,8,9,10])
	acc_ax.plot([],c="#FFA32E",lw=4,label="Biweek 10")
	acc_ax.plot([],c="#A600FF",lw=4,label="Biweek 15")
	acc_ax.legend(loc=1,frameon=False,fontsize=14)
		
	## Finish up
	fig.tight_layout()
	reff_ax.text(-0.25,1.,"a.",fontsize=18,color="k",transform=reff_ax.transAxes)
	rr_ax.text(-0.11,1.,"c.",fontsize=18,color="k",transform=rr_ax.transAxes)
	acc_ax.text(-0.25,1.,"b.",fontsize=18,color="k",transform=acc_ax.transAxes)
	fig.savefig("_plots\\inferences.png")
	
	## Done
	plt.show()