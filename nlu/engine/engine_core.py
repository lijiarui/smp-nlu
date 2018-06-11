"""Engine的基类"""

class EngineCore(object):
    """Engine的基类"""

    def __init__(self, domain_implement, intent_implement, slot_implement):
        """初始化"""
        self.domain_implement = domain_implement
        self.intent_implement = intent_implement
        self.slot_implement = slot_implement

    def predict_domain(self):
        raise Exception('domain not implement')

    def predict_intent(self):
        raise Exception('intent not implement')

    def predict_slot(self):
        raise Exception('slot not implement')
