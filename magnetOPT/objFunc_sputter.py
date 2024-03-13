import sputterSimulator
import numpy as np
# import cupy as cp
import magpylib as magpy

class Erosion:
    def __init__(self, magN, seed):
        self.magN = magN
        self.seed = seed

    def MagnetGenerator(self):
        # N = 30
        np.random.seed(self.seed)
        dimX = np.random.rand(self.magN)*220
        dimY = np.random.rand(self.magN)*220
        polarization = np.random.rand(self.magN, 3)*np.pi
        magCollection = []

        for i in range(self.magN):
            mag = magpy.magnet.Cuboid(magnetization=[0, 0, 1.4], dimension=[10,10,10],position=[dimX[i],dimY[i],0])
            mag.rotate_from_euler(polarization[i], 'xyz', degrees=False)
            magCollection.append(mag)

        Allmag = magCollection[0]
        for i in range(self.magN-1):
            Allmag = magpy.Collection(magCollection[i+1], Allmag)
        return Allmag
    
    def Bfield_generator(self, TM, cellx, celly, cellz):
        # TM = 37
        xs = np.linspace(-220,220,cellx) #220
        ys = np.linspace(-220,220,celly) #220
        zs = np.linspace(TM,TM + 50,cellz) # 25

        coordinates = [np.array([x, y, z]) for x in xs for y in ys for z in zs] 
        Bs = self.MagnetGenerator().getB(coordinates)
        Bs = np.array(Bs).reshape(len(xs), len(ys), len(zs), 3)
        return Bs
    
    def electron_generator(self, N):
        # N = int(5e6)
        x_ca1 = np.random.rand(N)*0.44 - 0.22
        y_ca1 = np.random.rand(N)*0.44 - 0.22


        inipositions = []
        R = 0.22

        for l in range(N):
            if x_ca1[l]**2 + y_ca1[l]**2 <= 0.22**2:
                iniposition = np.array([x_ca1[l], y_ca1[l], 0])
                inipositions.append(iniposition)

        rng = np.random.default_rng()
        rng.shuffle(inipositions)

        N2 = np.array(inipositions).shape[0]

        inipositions_cp = np.array(inipositions)
        inivelosity_cp = np.zeros([N2, 3])

        return inipositions_cp ,inivelosity_cp
    

    def runSputter(self):
        Bs = self.Bfield_generator(37, 220, 220, 25)
        electron_pv = self.electron_generator(int(5e6))
        result = sputterSimulator.Sputter(540, 60, 0.3, 293, Bs)
        result.setXsec()
        result.ShealthPhi(11, 70, 540)
        erosion = result.runE(p0 = electron_pv[0], v0 = electron_pv[1], time = 1e-8)
        return erosion
    
    def erosion_profile(self):
        a = self.runSputter()
        x_erosion = a[1][1:,0]
        y_erosion = a[1][1:,1]
        # z_erosion = a[1][1:,2]
        r_hist = np.sqrt(x_erosion**2 + y_erosion**2)
        r_indice = r_hist < 0.22
        r_hist = r_hist[r_indice]
        bins = 100
        hist = np.histogram(r_hist, bins=bins)
        return hist[0]
