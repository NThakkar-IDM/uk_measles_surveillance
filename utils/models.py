""" models.py

Functions and classes for inference. """

## Standard stuff
import numpy as np
import pandas as pd

## For matrix constructions
from scipy.sparse import diags

class StrongPriorModel(object):

	def __init__(self,df,S0,S0_var=None,beta_corr=3.):

		## Store some information about the model
		self.T = len(df)-1
		self.beta_corr = beta_corr
		self.C_t = df["cases"].values
		self.B_t = df["births"].values

		## Regularize S0?
		if S0_var is not None:
			self.logS0var = S0_var/(S0**2)
		else:
			self.logS0var = None

		## Initialize key pieces, like the periodic smoothing
		## matrix.
		D2 = np.diag(26*[-2])+np.diag((26-1)*[1],k=1)+np.diag((26-1)*[1],k=-1)
		D2[0,-1] = 1 ## Periodic BCs
		D2[-1,0] = 1
		self.pRW2 = np.dot(D2.T,D2)*((beta_corr**4)/4.)

		## The design matrices for the linear portion of the 
		## transmission regression problem.
		self.X = np.vstack((int(len(df)-1/len(self.pRW2))+1)*[np.eye(len(self.pRW2))])[:len(df)-1]
		self.C = np.linalg.inv(np.dot(self.X.T,self.X)+self.pRW2)
		self.H = np.dot(self.X,np.dot(self.C,self.X.T))

		## And the prior pieces
		self.r_hat = df["rr_p"].values#[1:]
		self.r_prec = np.diag(1./(df["rr_p_var"].values))#[1:]))

		## Set up the initial guess for the parameters
		self.theta0 = np.ones((self.T+1+1,))
		self.theta0[0] = np.log(S0)
		self.theta0[1:] = df["rr_p"].values#[1:]

	def __call__(self,theta):

		## Unpack the input
		S0 = np.exp(theta[0])
		r_t = theta[1:]
	
		## Compute the implied model compartments
		adj_cases = (self.C_t+1.)/r_t - 1.
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Solve the linear regression problem
		Y_hat = np.dot(self.H,Y_t)
		RSS = np.sum((Y_t-Y_hat)**2)
		lnp_beta = 0.5*self.T*np.log(RSS/self.T)

		## If you added S0 var
		if self.logS0var is not None:
			lnp_S0 = 0.5*((theta[0]-self.theta0[0])**2)/self.logS0var
		else:
			lnp_S0 = 0

		## Compute the r_t component
		lnp_rt = 0.5*np.sum(((r_t-self.r_hat)**2)*np.diag(self.r_prec))

		return lnp_beta + lnp_rt + lnp_S0

	def grad(self,theta):

		## Unpack the input
		S0 = np.exp(theta[0])
		r_t = theta[1:]

		## Compute the implied model compartments
		adj_cases = (self.C_t+1.)/r_t - 1.
		E_t = adj_cases[1:]
		I_t = adj_cases[:-1]
		S_t = S0 + np.cumsum(self.B_t[:-1]-E_t)
		Y_t = np.log(E_t)-np.log(I_t)-np.log(S_t)

		## Solve the linear regression problem and compute
		## The implied variance
		Y_hat = np.dot(self.H,Y_t)
		resid = Y_t-Y_hat
		var = np.sum(resid**2)/self.T

		## Compute the contribution from S_t
		dYdt0 = -S0/S_t 
		grad_t = np.dot(dYdt0-np.dot(self.H,dYdt0),resid)/var
		if self.logS0var is not None:
			grad_t += (theta[0]-self.theta0[0])/self.logS0var

		## Compute the contribution for r_t 
		dYtdrt = (I_t+1.)/(I_t*r_t[:-1])
		dYtdrt_plus1 = -(E_t+1.)/(E_t*r_t[1:])
		dYdr = diags([dYtdrt,dYtdrt_plus1],[0,1],shape=(self.T,self.T+1)).todense()
		dYdr[:,1:] += -np.tril(np.outer(1./S_t,(E_t+1.)/r_t[1:]))
		grad_r = np.dot((dYdr-np.dot(self.H,dYdr)).T,resid)/var
		grad_r += np.dot(self.r_prec,r_t-self.r_hat)
		
		## Combine and return
		jac = np.zeros((len(theta),))
		jac[0] = grad_t
		jac[1:] = grad_r
		return jac