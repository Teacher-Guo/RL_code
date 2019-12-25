import tensorflow as tf
import numpy as np
import gym
import scipy as sp
from gmm import *
# from general_utils import gauss_fit_joint_prior
def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg):
    """ Perform Gaussian fit to data with a prior. """
    # Build weights matrix.
    D = np.diag(dwts)
    # Compute empirical mean and covariance.
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    empsig = 0.5 * (empsig + empsig.T)
    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    mu = mun
    sigma = (N * empsig + Phi + (N * m) / (N + m) *
             np.outer(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    # Add sigma regularization.
    sigma += sig_reg
    # Conditioning to get dynamics.
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
    dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig
def traj_distr_kl_alt(new_mu, new_sigma, new_traj_distr, prev_traj_distr, tot=True):
    """
    This function computes the same quantity as the function above.
    However, it is easier to modify and understand this function, i.e.,
    passing in a different mu and sigma to this function will behave properly.
    """
    T, dX, dU = new_mu.shape[0], new_traj_distr.dX, new_traj_distr.dU
    kl_div = np.zeros(T)

    for t in range(T):
        K_prev = prev_traj_distr.K[t, :, :]
        K_new = new_traj_distr.K[t, :, :]

        k_prev = prev_traj_distr.k[t, :]
        k_new = new_traj_distr.k[t, :]

        sig_prev = prev_traj_distr.pol_covar[t, :, :]
        sig_new = new_traj_distr.pol_covar[t, :, :]

        chol_prev = prev_traj_distr.chol_pol_covar[t, :, :]
        chol_new = new_traj_distr.chol_pol_covar[t, :, :]

        inv_prev = prev_traj_distr.inv_pol_covar[t, :, :]
        inv_new = new_traj_distr.inv_pol_covar[t, :, :]

        logdet_prev = 2 * sum(np.log(np.diag(chol_prev)))
        logdet_new = 2 * sum(np.log(np.diag(chol_new)))

        K_diff, k_diff = K_prev - K_new, k_prev - k_new
        mu, sigma = new_mu[t, :dX], new_sigma[t, :dX, :dX]

        kl_div[t] = max(
                0,
                0.5 * (logdet_prev - logdet_new - new_traj_distr.dU +
                       np.sum(np.diag(inv_prev.dot(sig_new))) +
                       k_diff.T.dot(inv_prev).dot(k_diff) +
                       mu.T.dot(K_diff.T).dot(inv_prev).dot(K_diff).dot(mu) +
                       np.sum(np.diag(K_diff.T.dot(inv_prev).dot(K_diff).dot(sigma))) +
                       2 * k_diff.T.dot(inv_prev).dot(K_diff).dot(mu))
        )

    return np.sum(kl_div) if tot else kl_div
class Sample():
    def __init__(self,env, policy_net):
        self.env = env
        self.policy_net=policy_net
        self.gamma = 0.98
    def sample_episodes(self, num_episodes):
        #产生num_episodes条轨迹
        batch_obs=[]
        batch_actions=[]
        batch_rs =[]
        for i in range(num_episodes):
            observation = self.env.reset()
            #将一个episode的回报存储起来
            reward_episode = []
            while True:
                if RENDER:self.env.render()
                #根据策略网络产生一个动作
                state = np.reshape(observation,[1,4])
                action = self.policy_net.choose_action(state)
                observation_, reward, done, info = self.env.step(action)
                batch_obs.append(observation)
                batch_actions.append(action)
                reward_episode.append(reward)
                #一个episode结束
                if done:
                    #处理回报函数
                    reward_sum = 0
                    discouted_sum_reward = np.zeros_like(reward_episode)
                    for t in reversed(range(0, len(reward_episode))):
                        reward_sum = reward_sum*self.gamma + reward_episode[t]
                        discouted_sum_reward[t] = reward_sum
                    #归一化处理
                    discouted_sum_reward -= np.mean(discouted_sum_reward)
                    discouted_sum_reward/= np.std(discouted_sum_reward)
                    #将归一化的数据存储到批回报中
                    for t in range(len(reward_episode)):
                        batch_rs.append(discouted_sum_reward[t])
                    break
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.policy_net.n_features])
        batch_actions = np.reshape(batch_actions,[len(batch_actions),])
        batch_rs = np.reshape(batch_rs,[len(batch_rs),])
        return batch_obs, batch_actions,batch_rs
class DynamicsPriorGMM():
    def __init__(self):
        self.X = None
        self.U = None
        #高斯混合模型GMM在文件gmm.py中
        self.gmm = GMM()
        self._min_samp = 20
        self._max_samples = 20
        self._max_clusters = 50
        self._strength = 1.0
    def initial_state(self):
        mu0 = np.mean(self.X[:,0,:], axis=0)
        Phi = np.diag(np.var(self.X[:,0,:],axis=0))
        #Factor in multiplier
        n0 = self.X.shape[2] * self._strength
        m = self.X.shape[2] * self._strength
        Phi = Phi * m
        return  mu0, Phi, m, n0
    def update(self, X, U):
        T = X.shape[1] -1
        if self.X in None:
            self.X =X
        else:
            self.X = np.concatenate([self.X, X], axis=0)
        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)
        #从数据集中移除多余的样本
        start = max(0, self.X.shape[0]-self._max_samples+1)
        self.X = self.X[start:,:]
        self.U = self.U[start:,:]
        #计算总维数
        Do = X.shape[2]+U.shape[2]+X.shape[2]
        #创建数据集
        N = self.X.shape[0]
        xux = np.reshape(np.c_[self.X[:,:T,:], self.U[:,:T,:], self.X[:,1:(T+1),:]],[T*N,Do])
        #选择类的数目
        K = int(max(2, min(self._max_clusters, np.floor(float(N*T)/self._min_samp))))
        #更新GMM
        self.gmm.update(xux,K)
    def eval(self, Dim_x,Dim_u,pts):
        assert pts.shape[1] == Dim_x + Dim_u +Dim_x
        mu0, Phi, m, n0 = self.gmm.inference(pts)
        n0 = n0 * self._strength
        m = m * self._strength
        Phi *= m
        return mu0, Phi, m, n0
#无先验的动力学拟合x_{t+1} = Fm*[x_t;u_t]+fv; 每个时间步都有一个Fm和fv
class DynamicsLR():
    def __init__(self):
        self.Fm = None
        self.fv = None
        self.dyna_covar = None
    def fit(self,X,U):
        #N为轨迹的条数，T为时间步，dim_X为X的维数,dim_U为U的维数
        N, T, dim_X = X.shape
        dim_U = U.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")
        self.Fm = np.zeros([T, dim_X, dim_X+dim_U])
        self.fv = np.zeros([T, dim_X])
        self.dyna_covar = np.zeros([T, dim_X, dim_X])
        it = slice(dim_X+dim_U)
        ip = slice(dim_X+dim_U, dim_X+dim_U+dim_X)
        #利用最小二乘法回归拟合
        for t in range(T-1):
            #将状态和动作连成一个多维数组
            xux = np.c_[X[:,t,:],U[:,t,:],X[:,t+1,:]]
            xux_mean = np.mean(xux, axis=0)
            empsig = (xux-xux_mean).T.dot(xux-xux_mean)/N
            sigma = 0.5*(empsig + empsig.T)
            Fm = np.linalg.solve(sigma[it,it], sigma[it,ip]).T
            fv = xux_mean[ip] - Fm.dot(xux_mean[it])
            self.Fm[t,:,:] = Fm
            self.fv[t,:] = fv
            dyn_covar = sigma[ip, ip]-Fm.dot(sigma[it, it]).dot(Fm.T)
            self.dyna_covar[t,:,:] = 0.5*(dyn_covar + dyn_covar.T)
        return self.Fm, self.fv, self.dyna_covar
#带有先验的动力学拟合x_{t+1} = Fm[x_t,u_t]+fv；每个时间步都有一个Fm和fv
class DynamicsLRPrior():
    def __init__(self):
        self.Fm = None
        self.fv = None
        self.dyna_covar = None
        #先验为高斯
        self.prior =DynamicsPriorGMM()
    def update_prior(self, samples):
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X,U)
    def get_prior(self):
        return self.prior
    def fit(self, X, U):
        # N为轨迹的条数，T为时间步，dim_X为X的维数,dim_U为U的维数
        N, T, dim_X = X.shape
        dim_U = U.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")
        self.Fm = np.zeros([T, dim_X, dim_X + dim_U])
        self.fv = np.zeros([T, dim_X])
        self.dyna_covar = np.zeros([T, dim_X, dim_X])
        it = slice(dim_X + dim_U)
        ip = slice(dim_X + dim_U, dim_X + dim_U + dim_X)
        dwts = (1.0/N) * np.ones(N)
        # 利用最小二乘法回归拟合
        for t in range(T - 1):
            # 将状态和动作连成一个多维数组
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]
            mu0, Phi, mm, n0 = self.prior.eval(dim_X, dim_U, Ys)
            sig = np.zeros((dim_X+dim_U+dim_X, dim_X+dim_U+dim_X))
            Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys, mu0, Phi, mm,n0,dwts,dim_X+dim_U,dim_X,sig)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyna_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyna_covar
#定义计算代价
class Single_traj_cost():
    def __init__(self):
        self.sum_cost = 0
        self.w =0.01
    #计算动作代价
    def cost_action(self, sample):
        sample_u = sample.get_U()
        T = sample.T
        Dim_u = sample.dim_U
        Dim_x = sample.dim_X
        l = 0.5 * np.sum(sample_u ** 2, axis=1)
        lu = sample_u
        lx = np.zeros((T, Dim_x))
        luu = np.tile(np.diag(np.array(1,1)),[T,1,1])
        lxx = np.zeros((T, Dim_x, Dim_x))
        lux = np.zeros((T, Dim_u, Dim_x))
        return l, lx, lu, lxx, luu, lux
    #计算状态代价
    def cost_state(self, sample):
        w1 = 0.5
        T = sample.T
        Dim_u = sample.dim_U
        Dim_x = sample.dim_X
        tgt = sample.get_target()
        x = sample.get()
        dist = x -  tgt
        l = w1* np.sum(dist ** 2, axis = 1)
        lu =  np.zeros((T, Dim_u))
        luu = np.tille(np.zeros(2,2),[T,1,1])
        lx = np.zeros((T, Dim_x))
        lx[:,0:1] = 1
        lux = np.zeros((T, Dim_u,Dim_x))
        lxx = np.zeros((T, Dim_x, Dim_x))
        lxx[:,0,0]= 1
        lxx[:,1,1] = 1
        return l,lx, lu, lxx, luu, lux
    def eval_cost(self,sample):
        return self.cost_state(sample)+self.w*self.cost_action(sample)
class Batch_traj_cost():
    def __init__(self):
         self.single_traj_cost = Single_traj_cost()
    def eval_cost(self, sample_list):
        T, dim_X, dime_U = sample_list[0].T, sample_list[0].dim_X, sample_list[0].dim_U
        N = len(sample_list)
        cs = np.zeros((N,T))
        cc = np.zeros((N,T))
        cv = np.zeros((N, T, dim_X+dime_U))
        Cm = np.zeros((N, T, dim_X+dime_U, dim_X+dime_U))
        for n in range(N):
            sample =  sample_list[n]
            #单条轨迹的代价及其偏导数
            l, lx, lu, lx, lxx, luu, lux = self.single_traj_cost(sample)
            cc[n, :] = l
            cs[n, :] = l
            #组装矩阵
            cv[n,:,:] = np.c_[lx,lu]
            Cm[n,:,:,:] = np.concatenate((np.c_[lxx,np.transpose(lux,[0,2,1])],np.c_[lux, luu]),axis=1)
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X,U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff,axis=2)
            cv_update = np.sum(Cm[n,:,:,:]*rdiff_expand, axis=1)
            cc[n,:] += np.sum(rdiff*cv[n,:,:],axis=1)+0.5*np.sum(rdiff*cv_update,axis=1)
            cv[n,:,:] += cv_update
        traj_cc = np.mean(cc, 0)
        traj_cv = np.mean(cv, 0)
        traj_Cm = np.mean(Cm, 0)
        return traj_Cm, traj_cv,traj_cc
#定义policy fit，拟合线性策略，先验为高斯先验
class Policy_Prior_GMM():
    def __init__(self):
        self.X = None
        self.obs = None
        self.gmm = GMM()
        self._min_samp = 20
        self._max_samples = 20
        self._max_clusters = 50
        self._strength = 1.0
    #更新先验
    def update(self, samples, policy_opt, mode = 'add'):
        X, obs = samples.get_X(), samples.get_obs()
        if self.X is None or mode == 'replace':
            self.X = X
            self.obs = obs
        elif mode == 'add' and X.size > 0:
            self.X = np.concatenate([self.X, X], axis=0)
            self.obs = np.concatenate([self.obs, obs], axis=0)
            N = self.X.shape[0]
            if N > self._max_samples:
                start = N - self._max_samples
                self.X = self.X[start:,:,:]
                self.obs = self.obs[start:,:,:]
        #在样本点处评估策略，以便得到均值策略动作
        U = policy_opt.prob(self.obs.copy())[0]
        #创建数据集
        N, T = self.X.shape[:,2]
        d0 = self.X.shape[2] + U.shape[2]
        XU = np.reshape(np.concatenate([self.X, U], axis=2), [T*N, d0])
        #选择聚类的数目
        K = int(max(2, min(self._max_clusters, np.floor(float(N*T)/self.min_samp))))
        self.gmm.update(XU, K)
    def eval(self, Ts, Ps):
        #计算先验
        pts = np.concatenate((Ts, Ps), axis=1)
        mu0, Phi, m, n0 = self.gmm.inference(pts)
        n0 *= self._strength
        m *= self._strength
        Phi *= m
        return mu0, Phi, m, n0
    def fit(self, X, pol_mu, pol_sig):
        #拟合线性策略
        N, T, dim_X = X.shape
        dim_U = pol_mu.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")
        pol_sig = np.mean(pol_sig, axis = 0)
        pol_K = np.zeros([T, dim_U, dim_X])
        pol_k = np.zeros([T, dim_U])
        pol_S = np.zeros([T, dim_U, dim_U])
        #拟合策略线性化，利用最小二乘回归
        dwts = (1.0/N)*np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            mu0, Phi, mm, n0 = self.eval(Ts, Ps)
            sig_reg = np.zeros((dim_X+dim_U, dim_X+dim_U))
            if t == 0:
                sig_reg[:dim_X, :dim_X] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = gauss_fit_joint_prior(Ys, mu0, Phi,mm, n0, dwts, dim_X, dim_U,sig_reg)
#LQR轨迹优化
class TrajOptLQR():
    def __init__(self):
        self.cons_per_step = False
        self._use_prev_distr = False
        self._update_in_bwd_pass = True
        self.min_eta = 1e-8
        self.max_eta = 1e16
        self.max_itr = 50
    def update(self, algorithm):
        T = algorithm.T
        eta = algorithm.eta
        step_mult = algorithm.step_mult
        traj_info = algorithm.traj_info
        kl_step = algorithm.base_kl_step * step_mult
        pre_traj_distr = algorithm.pol_info.traj_distr()
        #LQR迭代过程
        for itr in range(self.max_itr):
            #后项传递，计算代价及新的控制策略
            traj_distr, eta = self.backward(pre_traj_distr,traj_info,eta, algorithm)
            #前向计算
            prev_mu, prev_sigma = self.forward(pre_traj_distr,traj_info)
            kl_div = traj_distr_kl_alt(prev_mu, prev_sigma, traj_distr, pre_traj_distr)
            con = kl_div - kl_step
            if self._conv_check(con, kl_step):
                break
        return traj_distr, eta
    def backward(self, prev_traj_distr, traj_info, eta, algorithm):
        T = prev_traj_distr.T
        dim_U = prev_traj_distr.dim_U
        dim_X = prev_traj_distr.dim_X
        if self._update_in_bwd_pass:
            traj_distr = prev_traj_distr.nans_like()
        else:
            traj_distr = prev_traj_distr.copy()
        idx_x = slice(dim_X)
        idx_u = slice(dim_X, dim_X + dim_U)
        #动力学系数
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        eta0 = eta
        fail = True
        #动态规划
        while fail:
            fail = False
            V = np.zeros((T, dim_X, dim_X))
            v = np.zeros((T,dim_X))
            Q = np.zeros((T, dim_X+dim_U, dim_X+dim_U))
            q = np.zeros((T, dim_X + dim_U))
            #代价系数
            fCm, fcv = algorithm.compute_costs(eta)
            for t in range(T-1, -1, -1):
                Q[t] = fCm[t,:,:]
                q[t] = fcv[t,:]
                if t< T-1:
                    Q[t]+=Fm[t,:,:].T.dot(V[t+1, :, :]).dot(Fm[t,:,:])
                    q[t]+=Fm[t,:,:].T.dot(v[t+1,:]+V[t+1,:,:].dot(fv[t,:]))
                Q[t] = 0.5*(Q[t]+Q[t].T)
                inv_term = Q[t, idx_u, idx_u]
                k_term = q[t,idx_u]
                K_term = Q[t, idx_u, idx_x]
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    fail = t if self.cons_per_step else True
                    break
                if self._update_in_bwd_pass:
                    #存储条件协方差，逆和cholesky分解
                    traj_distr.inv_pol_covar[t, :, :] = inv_term
                    traj_distr.pol_covar[t:, :, :] = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, np.eye(dim_U)))
                    traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(traj_distr.pol_covar[t,:,:])
                    #计算均值项
                    traj_distr.k[t,:] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, k_term))
                    traj_distr.K[t,:,:] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, K_term))
                else:
                    new_ipS[t,:,:] = inv_term
                    new_pS[t,:,:] = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, np.eye(dim_U)))
                    new_cpS[t,:,:] = sp.linalg.cholesky(new_pS[t,:,:])
                    # 计算均值项
                    new_k[t, :] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, k_term))
                    new_K[t, :, :] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, K_term))
                if (self.cons_per_step or not self._update_in_bwd_pass):
                    V[t,:,:] = Q[t, idx_x,idx_x]+ traj_distr.K[t].T.dot(Q[t,idx_u,idx_u]).dot(traj_distr.K[t])+(2*Q[t, idx_x, idx_u]).dot(traj_distr.K[t])
                    v[t,:] = q[t, idx_x].T +q[t,idx_u].T.dot(traj_distr.K[t])+traj_distr.k[t].T.dot(Q[t, idx_u,idx_u]).dot(traj_distr.K[t])+Q[t, idx_x, idx_u].dot(traj_distr.k[t])
                else:
                    V[t, :, :] = Q[t, idx_x, idx_x] + Q[t, idx_x, idx_u].dot(traj_distr.K[t,:,:])
                    v[t, :] = q[t, idx_x] + Q[t, idx_x, idx_u].dot(traj_distr.k[t,:])
                V[t, :, :]= 0.5 *(V[t,:,:]+V[t,:,:].T)
            if not self._update_in_bwd_pass:
                traj_distr.K, traj_distr.k = new_K, new_k
                traj_distr.pol_covar = new_pS
                traj_distr.inv_pol_covar = new_ipS
                traj_distr.chol_pol_covar = new_cpS
        return traj_distr, eta
    def _conv_check(self, con, kl_step):
        return abs(con) < 0.1*kl_step
    def forward(self, traj_distr, traj_info):
        T = traj_distr.T
        dim_U =  traj_distr.dim_U
        dim_X = traj_distr.dim_X
        idx_x = slice(dim_X)
        sigma = np.zeros((T, dim_X+dim_U, dim_X+dim_U))
        mu = np.zeros((T, dim_X+dim_U))
        #系统动力学
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar
        #设置初始协方差
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu
        #前向计算
        for t in range(T):
            sigma[t, :, :] = np.vstack([np.hstack([sigma[t, idx_x, idx_x], sigma[t, idx_x,idx_x].dot(traj_distr.K[t, :, :].T)]),
                                        np.hstack([traj_distr.K[t,:,:].dot(sigma[t, idx_x, idx_x]), traj_distr.K[t,:,:].dot(sigma[t,idx_x,idx_x]).dot(traj_distr.K[t,:,:].T)+traj_distr.pol_covar[t,:,:]])])
            mu[t,:] = np.hstack([mu[t, idx_x], traj_distr.K[t,:,:].dot(mu[t, idx_x])+traj_distr.k[t,:]])
            if t < T-1:
                sigma[t+1, idx_x, idx_x]= Fm[t,:,:].dot(sigma[t, :,:]).dot(Fm[t,:,:].T)+dyn_covar[t,:,:]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t,:])+fv[t,:]
        return mu, sigma
#监督相的策略优化
class PolicyOpt():
    def __init__(self, env, action_bound, lr = 0.0001, model_file=None):
        self.iterations = 5000
        self.batch_size = 200
        self.learning_rate = lr
        #输入特征的维数
        self.n_features = env.observation_space.shape[0]
        #输出动作空间的维数
        self.n_actions = 1
        #1.1 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        #1.2.第一层隐含层
        self.f1 = tf.layers.dense(inputs=self.obs, units=200, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                             bias_initializer=tf.constant_initializer(0.1))
        #1.3 第二层，均值
        mu = tf.layers.dense(inputs=self.f1, units=self.n_actions, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                                       bias_initializer=tf.constant_initializer(0.1))
        #1.3 第二层，标准差
        sigma = tf.layers.dense(inputs=self.f1, units=self.n_actions, activation=tf.nn.softplus, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                                       bias_initializer=tf.constant_initializer(0.1))
        #
        self.mu = 2*mu
        self.sigma =sigma
        # 定义带参数的正态分布
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        #根据正态分布采样一个动作
        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0],action_bound[1])
        #1.5 当前动作
        self.current_act = tf.placeholder(tf.float32, [None,1])
        self.current_reward = tf.placeholder(tf.float32, [None,1])
        #2. 构建损失函数
        log_prob = self.normal_dist.log_prob(self.current_act)
        self.loss = tf.reduce_mean(log_prob*self.current_reward+0.01*self.normal_dist.entropy())
        # self.loss += 0.01*self.normal_dist.entropy()
        #3. 定义一个优化器
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.loss)
        #4. tf工程
        self.sess = tf.Session()
        #5. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    #依概率选择动作
    def choose_action(self, state):
        action = self.sess.run(self.action, {self.obs:state})
        # print("greedy action",action)
        return action[0]
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
    #定义训练步
    def train_step(self, obs, tgt_mu, tgt_prc, tgt_wt):
        #obs: numpy数组，用于表示观测，注意与状态X的区别，数组大小为： N*T*dim_obs
        N, T = obs[:,2]
        dim_U, dim_obs =self.dim_U, self.dim_obs
        #保存原始的精度矩阵
        tgt_prc_orig =  np.reshape(tgt_prc, [N*T, dim_U, dim_U])
        #归一化
        tgt_wt *= (float(N*T)/np.sum(tgt_wt))
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n,t] = min(tgt_wt[n,t], 2*mn)
        tgt_wt /= mn
        #Reshape 输入
        obs = np.reshape(obs, (N*T, dim_obs))
        tgt_mu =  np.reshape(tgt_mu, (N*T, dim_U))
        tgt_prc = np.reshape(tgt_prc, (N*T, dim_U, dim_U))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1,1))
        tgt_prc = tgt_wt * tgt_prc
        batches_per_epoch = np.floor(N*T/self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)
        for i in range(self.iterations):
            start_idx = int(i*self.batch_size%(batches_per_epoch*self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.obs_tensor: obs[idx_i],
                         self.action_tensor:tgt_mu[idx_i],
                         self.precision_tensor: tgt_prc[idx_i]}
            train_loss = self.sess.run(self.loss_op, feed_dict)
            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f', i+1, average_loss/50)
                average_loss = 0
        #优化方差
        A = np.sum(tgt_prc_orig, 0)
        A = A/np.sum(tgt_wt)
        self.var = 1/np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))
        return self.policy
#定义mdgps类
class AlgorithmMDGPS():
    def __init__(self, traj_distr, pol_info, traj_info):
        self.pol_info = pol_info
        self.traj_info = traj_info
        self.T = traj_distr.T
        self.dim_U = traj_distr.dim_U
        self.dim_X = traj_distr.dim_X
        self.multiplier = 0.01
    def compute_cost(self, eta):
        #策略代价
        PKLm = np.zeros((self.T, self.dim_U+self.dim_X, self.dim_X+self.dim_U))
        PKLv = np.zeros((self.T, self.dim_X + self.dim_U))
        Cm, cv = np.copy(self.traj_info.Cm), np.copy(self.traj_info.cv)
        fCm = np.zeros((self.T, self.dim_U+self.dim_X, self.dim_X+self.dim_U))
        fcv = np.zeros((self.T, self.dim_U+self.dim_X))
        for t in range(self.T):
            #策略KL散度项
            #求解策略线性拟合的协方差的逆
            inv_pol_S = np.linalg.solve(self.pol_info.chol_pol_S[t, :, :],np.linalg.solve(self.pol_info.chol_pol_S[t,:,:].T, np.eye(self.dim_U)))
            KB, kB = self.pol_info.pol_K[t, :, :],self.pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),np.hstack([-inv_pol_S.dot(KB), inv_pol_S])])
            PKLv[t, :] = np.concatenate([KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)])
            fCm[t,:,:] = (Cm[t, :, :]+PKLm[t, :, :]*eta)/(eta+self.multiplier)
            fcv[t,:] = (cv[t,:] + PKLv[t,:]*eta)/(eta + self.multiplier)
        return fCm, fcv
def policy_train(env, brain, training_num):
    reward_sum = 0
    reward_sum_line = []
    training_time = []
    brain = brain
    env = env
    for i in range(training_num):
        temp = 0
        training_time.append(i)
        sampler = Sample(env, brain)
        # 采样10个episode
        train_obs, train_actions, train_rs = sampler.sample_episodes(1)
        # print("current trains_s",train_rs.shape)
        # print("max turn:%f,min turn:%f"%(max(train_rs),min(train_rs)))
        brain.train_step(train_obs, train_actions, train_rs)
        # 利用采样的数据进行梯度学习
        # for j in range(len(train_obs)):
        #     train_ob = np.reshape(train_obs[j,:],[1,3])
        #     train_action = np.reshape(train_actions[j,:],[1,1])
        #     train_r = np.reshape(train_rs[j,:],[1,1])
        #     brain.train_step(train_ob, train_action, train_r)
        # if i%50 == 0:
        #     RENDER = True
        # else:
        #     RENDER=False

        # print(loss.shape)
        # print("current experiments%d,current loss is %f"%(i, loss))
        if i == 0:
            reward_sum = policy_test(env, brain,RENDER,1)
        else:
            reward_sum = 0.95 * reward_sum + 0.05 * policy_test(env, brain,RENDER,1)
        # print(policy_test(env, brain))
        reward_sum_line.append(reward_sum)
        training_time.append(i)
        print("training episodes is %d,trained reward_sum is %f" % (i, reward_sum))
        # if reward_sum > -1:
        #     break
    brain.save_model('./current_bset_pg_pendulum')
    plt.plot(training_time, reward_sum_line)
    plt.show()
def policy_test(env, policy,RENDER,test_number):
    for i in range(test_number):
        observation = env.reset()
        reward_sum = 0
        # 将一个episode的回报存储起来
        while True:
            if RENDER:
                env.render()
            # 根据策略网络产生一个动作
            state = np.reshape(observation, [1, 3])
            action = policy.choose_action(state)
            observation_, reward, done, info = env.step(action)
            # print(reward)
            # reward_sum += (reward+8)/8
            reward_sum+=reward
            if done:
                # 处理回报函数
                break
            observation = observation_
    return reward_sum
#定义优化器
# class Optimizer()
#     def __init__(self):
if __name__=='__main__':
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    action_bound = [-env.action_space.high, env.action_space.high]
    brain = Policy_Net(env,action_bound)
    training_num = 1500
    policy_train(env, brain, training_num)
    reward_sum = policy_test(env, brain,True,10)
    print("test reward_sum is %f"%reward_sum)












