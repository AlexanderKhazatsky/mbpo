import numpy as np
import tensorflow as tf
import pdb

class FakeEnv:

    def __init__(self, model, config):
        self.model = model
        self.config = config

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self.model.random_inds(batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[model_inds, batch_inds]
        model_means = ensemble_model_means[model_inds, batch_inds]
        model_stds = ensemble_model_stds[model_inds, batch_inds]
        ####

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass



class FakeAdversarialEnv:

    def __init__(self, model, classifier, config):
        self.model = model
        self.classifier = classifier
        self.config = config
    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''

    def evaluate_kl(self, obs, act, rew, next_obs):
        model_trans = np.concatenate((obs, act, rew, next_obs - obs), axis=-1)
        ensemble_prob, _ = self.classifier.predict(model_trans, factored=True)
        prob = ensemble_prob[self.classifier._model_inds].mean(0)
        # num_models, batch_size, _ = ensemble_prob.shape
        # model_inds = self.classifier.random_inds(batch_size)
        # batch_inds = np.arange(0, batch_size)
        # prob = ensemble_prob[model_inds, batch_inds]
        #SWAP PROB HERE

        p, q = prob + self.classifier.eps, (1 - prob) + self.classifier.eps
        kl = np.log(q) - np.log(p)

        return kl, p, q

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        #### choose one model from ensemble, and get transition variables
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = self.model.random_inds(batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[self.model._model_inds].mean(0)
        #samples = ensemble_samples[model_inds, batch_inds]
        model_means = ensemble_model_means[model_inds, batch_inds]
        model_stds = ensemble_model_stds[model_inds, batch_inds]
        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)
        kl, p, q = self.evaluate_kl(obs, act, rewards, next_obs)
        ####

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'p': p, 'q': p, 'kl': kl}
        return next_obs, rewards, terminals, info

    ## Computational Graph

    def calc_kl(self, obs_ph, act_ph, deterministic):
        next_obs, rewards = self.step_ph(obs_ph, act_ph, deterministic=deterministic)
        dynamics_kl = self.kl_ph(obs_ph, act_ph, rewards, next_obs)
        return dynamics_kl

    # def calc_kl(self, obs_ph, act_ph, deterministic):
    #     assert len(obs_ph.shape) == len(act_ph.shape)

    #     inputs = tf.concat([obs_ph, act_ph], axis=1)
    #     ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
    #     ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
    #     ensemble_model_stds = tf.sqrt(ensemble_model_vars)

    #     if deterministic:
    #         ensemble_samples = ensemble_model_means
    #     else:
    #         ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        # total_kl = 0
        # for ind in self.model._model_inds:
        #     samples = tf.gather(ensemble_samples, ind)
        #     rewards, next_obs = samples[:,:1], samples[:,1:]
        #     expected_kl += self.kl_ph(obs_ph, act_ph, rewards, next_obs)

        # return total_kl / len(self.model._model_inds)

    def kl_ph(self, obs_ph, act_ph, rew_ph, next_obs_ph):
        model_trans = tf.concat([obs_ph, act_ph, rew_ph, next_obs_ph - obs_ph], axis=-1)
        ensemble_samples, _ = self.classifier.create_prediction_tensors(model_trans, factored=True)
        elite_samples = tf.gather(ensemble_samples, self.classifier._model_inds)
        prob = tf.reduce_mean(elite_samples, axis=0)
        # batch_size, input_size = tf.shape(elite_samples)[1], tf.shape(elite_samples)[2]
        # model_ind = tf.random_uniform((batch_size,), minval=0, maxval=self.classifier.num_elites, dtype=tf.int32)
        # cat_idx = tf.stack([model_ind, tf.range(0, batch_size)], axis=1)
        # prob = tf.gather_nd(elite_samples, cat_idx)

        p, q = prob + self.classifier.eps, (1 - prob) + self.classifier.eps
        return tf.math.log(q) - tf.math.log(p)

    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        elite_samples = tf.gather(ensemble_samples, self.model._model_inds)
        samples = tf.reduce_mean(elite_samples, axis=0)
        # batch_size, input_size = tf.shape(ensemble_model_means)[1], tf.shape(ensemble_model_means)[2]
        # model_ind = tf.random_uniform((batch_size,), minval=0, maxval=self.model.num_elites, dtype=tf.int32)
        # cat_idx = tf.stack([model_ind, tf.range(0, batch_size)], axis=1)
        # samples = tf.gather_nd(elite_samples, cat_idx)
        rewards, next_obs = samples[:,:1], samples[:,1:]

        return next_obs, rewards

    def close(self):
        pass