
class cmenuitem:
    def __init__(self,parent):
        self.children = []
        self.parent = parent

class cmenu(cmenuitem):
    def __init__(self):
        self.root = cmenuitem(None)


root = cmenuitem
g = cmenu()