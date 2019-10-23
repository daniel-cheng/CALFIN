import keras
from keras.callbacks import Callback
class AdditionalValidationSets(Callback):
	def __init__(self, validation_sets, verbose=0, batch_size=None):
		"""
		:param validation_sets:
		a list of 3-tuples (validation_data, validation_targets, validation_set_name)
		or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
		:param verbose:
		verbosity mode, 1 or 0
		:param batch_size:
		batch size to be used when evaluating on the additional datasets
		"""
		super(AdditionalValidationSets, self).__init__()
		self.validation_sets = validation_sets
		for validation_set in self.validation_sets:
			if len(validation_set) not in [2, 3]:
				raise ValueError()
		self.epoch = []
		self.history = {}
		self.verbose = verbose
		self.batch_size = batch_size

	def on_train_begin(self, logs=None):
		self.epoch = []
		self.history = {}

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epoch.append(epoch)

		# record the same values as History() as well
		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)

		# evaluate on the additional validation sets
		for validation_set in self.validation_sets:
			if len(validation_set) == 3:
				validation_data, validation_targets, validation_set_name = validation_set
				sample_weights = None
			elif len(validation_set) == 4:
				validation_data, validation_targets, sample_weights, validation_set_name = validation_set
			else:
				raise ValueError()

			results = self.model.evaluate(x=validation_data,
										  y=validation_targets,
										  verbose=self.verbose,
										  sample_weight=sample_weights,
										  batch_size=self.batch_size)

			for i, result in enumerate(results):
				if i == 0:
					valuename = validation_set_name + '_loss'
				else:
					if type(self.model.metrics[i-1]) is str:
						valuename = validation_set_name + '_' + self.model.metrics[i-1]
					else:
						valuename = validation_set_name + '_' + self.model.metrics[i-1].__name__
				if i == 2:
					print(' - ' + valuename + ': {:.4f}'.format(result), end='')
				self.history.setdefault(valuename, []).append(result)
			print()
