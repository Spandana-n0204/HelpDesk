class ContextManager:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        return self.history