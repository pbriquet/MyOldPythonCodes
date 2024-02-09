class HB_Data:
    def __init__(self):
        self.data = []
        self.triple = []
        self.T = []
        self.t = []
        self.J = []
        self.J_ave = []
        self.ExpLat = []
        self.ExpLat_ave = []
        self.Fibro = []
        self.Fibro_ave = []
        self.dic_T = dict()
        self.dic_t = dict()

    def readData(self,data):
        for line in data:
            if(len(line) == 1):
                T = line[0]
                x = (T,[])
                self.dic_T[T] = dict()
                self.data.append(x)
            else:
                pairs_t_HB = line
                hb_list = []
                for i in xrange(len(pairs_t_HB)):
                    if(i == 0):
                        time = pairs_t_HB[0]
                        
                        self.dic_T[T][time] = []
                    else:
                        hb = pairs_t_HB[i]
                        hb_list.append(hb)
                        self.dic_T[T][time].append(hb)

                ave = reduce(lambda x, y: float(x) + float(y), hb_list) / float(len(hb_list))

                tup = (time, hb_list, ave)
                x[1].append(tup) 
                
        
        for T in self.data:
            for t in T[1]:
                HB_list = []
                for HB in t[1]:
                    self.T.append(T[0])
                    self.t.append(t[0])
                    self.HB.append(HB)
                    HB_list.append(HB)
                    self.triple.append((T[0],t[0],HB))
                self.HB_ave.append(reduce(lambda x, y: float(x) + float(y), HB_list) / float(len(HB_list)))
        
    def getArrayTimeHardness(self,T):
        x = []
        y = []

        for line in self.data:
            if(float(line[0]) == T):
                for times in line[1]:
                    for hb in times[1]:
                        x.append(times[0])
                        y.append(hb)

        return x, y

    def getArrayTimeHardnessAverage(self,T):
        x = []
        y = []

        for line in self.data:
            if(float(line[0]) == T):
                for times in line[1]:
                    x.append(times[0])
                    y.append(times[2])

        return x, y


    def getArrayTemperatureHardnessAverage(self,t):
        x = []
        y = []
        for line in self.data:
            for times in line[1]:
                if(float(times[0]) == t):
                    x.append(line[0])
                    y.append(times[2])
        return x,y
    