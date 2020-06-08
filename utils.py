# Config to choose the hyperparameters for everything
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

   
def standard_scale(x_train, x_test):
	scalar = StandardScaler()
	scalar.fit(x_train)
	x_train = scalar.transform(x_train)
	x_test = scalar.transform(x_test)
	return x_train, x_test