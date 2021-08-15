class PPR:
    def __init__(self, init_probs = 0.8, decay_probs = 0.05):
        self.vals = {}
        self.init_probs = init_probs
        self.decay_probs = decay_probs
        return
    def step(self):
        for key in self.vals:
            value = self.vals[key]
            rate = value['rate']
            if rate > 0: 
                value['rate'] = rate - self.decay_probs
                self.vals[key] = value
            else: 
                #do nothing
                self.vals[key]['rate'] = 0
                #del self.vals[key]
    def add(self, key, action):
        self.vals[key] = {'action': action, 'rate': self.init_probs}
    def get(self, key):
        if key in self.vals:
            return self.vals[key]['action'], self.vals[key]['rate']
        else:
            return None, 0
