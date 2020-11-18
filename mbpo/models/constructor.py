import numpy as np
import tensorflow as tf

from mbpo.models.fc import FC
from mbpo.models.bnn import BNN

def construct_model(name='BNN', obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5,
	q_func=None, classifier=None, use_classifier=False, is_classifier=False, session=None):
	print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
	params = {'name': name, 'num_networks': num_networks, 'num_elites': num_elites, 'q_func': q_func,
		'classifier': classifier, 'use_classifier': use_classifier, 'is_classifier': is_classifier, 'sess': session}
	model = BNN(params)

	if is_classifier:
		model.add(FC(hidden_dim, input_dim=obs_dim*2+act_dim+rew_dim, activation="swish", weight_decay=0.000025))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(1, activation="sigmoid", weight_decay=0.0001))
		model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
	else:
		model.add(FC(hidden_dim, input_dim=obs_dim+act_dim, activation="swish", weight_decay=0.000025))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(obs_dim+rew_dim, weight_decay=0.0001))
		model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

	return model

def format_samples_for_training(samples):
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew, delta_obs), axis=-1)
	return inputs, outputs

def format_samples_for_classifier(env_samples, model_samples, balance=False):
	r_inputs = np.concatenate(format_samples_for_training(env_samples), axis=-1)
	m_inputs = np.concatenate(format_samples_for_training(model_samples), axis=-1)

	if balance:
		data_lim = min(r_inputs.shape[0], m_inputs.shape[0])
		r_ind = np.random.choice(r_inputs.shape[0], size=data_lim, replace=False)
		m_ind = np.random.choice(m_inputs.shape[0], size=data_lim, replace=False)
		r_inputs, m_inputs = r_inputs[r_ind], m_inputs[m_ind]

	r_labels = np.ones((r_inputs.shape[0], 1))
	m_labels = np.zeros((m_inputs.shape[0], 1))

	inputs = np.concatenate((r_inputs, m_inputs), axis=0)
	outputs = np.concatenate((r_labels, m_labels), axis=0)

	return inputs, outputs

# def format_samples_for_classifier(env_samples, model_samples, balance=False):
# 	r_inputs = np.concatenate(format_samples_for_training(env_samples), axis=-1)
# 	m_inputs = np.concatenate(format_samples_for_training(model_samples), axis=-1)

# 	if balance:
# 		data_lim = min(r_inputs.shape[0], m_inputs.shape[0])
# 		r_inputs, m_inputs = r_inputs[:data_lim], m_inputs[:data_lim]

# 	r_labels = np.ones((r_inputs.shape[0], 1))
# 	m_labels = np.zeros((m_inputs.shape[0], 1))

# 	inputs, outputs = [], []
# 	for i in range(r_inputs.shape[0] + m_inputs.shape[0]):
# 		if (i % 2) == 0:
# 			inputs.append(r_inputs[i // 2])
# 			outputs.append(r_labels[i // 2])
# 		else:
# 			inputs.append(m_inputs[i // 2])
# 			outputs.append(m_labels[i // 2])


# 	inputs = np.stack(inputs)
# 	outputs = np.stack(outputs)

# 	return inputs, outputs

def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))

if __name__ == '__main__':
	model = construct_model()
