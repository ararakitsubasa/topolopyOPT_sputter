import depoML
import numpy as np
from joblib import Parallel, delayed
import torch

def unwrap_self(arg, **kwarg):
    return StepCoverage.coverDist(*arg, **kwarg)

class StepCoverage:
    def __init__(self, num):
        self.num = num
        self.TS = 0.11
        self.param = [1.6, -0.7] # al
        self.coordinates = np.linspace(0,0.21,3)
        self.device='cuda:0'

        '''
        Solve a step coverage problem

        Args:
            erosion bins : torch.Tensor, size = (N, bins), dtype = float64

        Returns:
            coverage : torch.Tensor, size=(bins, ), dtype=float64
        '''


    def sampleGen(self, num, sigma, mu):
        # bins = 100
        samples = np.multiply(np.random.randn(int(1e5), num), sigma) + mu
        sampleFlat = samples.reshape(-1)
        # dist = np.histogram(samples, bins, range=[0.0, 0.22])
        return sampleFlat
    
    def distGenerator(self, base, bins=200):
        n_dist = 20
        generate_random = base.cpu().numpy()

        sigma = (generate_random[:20] + 0.1)*0.1
        mu = generate_random[20:]*0.25
        a = self.sampleGen(n_dist, sigma, mu)
        dist = np.histogram(a, bins=bins, range=[0, 0.22], density=True)
        distGen = dist[0]*1e3
        distNorm = distGen*dist[1][1:]*2*np.pi     
        gen_dist_x = np.array([])
        gen_dist_y = np.array([])
        r = 0.22/bins
        print(np.sum(distGen))
        for i in range(bins):
            theta = np.random.rand(int(distNorm[i]))*2*np.pi
            gen_dist_x = np.concatenate((gen_dist_x, np.multiply((np.random.rand(int(distNorm[i]))*r + i*r), np.cos(theta))))
            gen_dist_y = np.concatenate((gen_dist_y, np.multiply((np.random.rand(int(distNorm[i]))*r + i*r), np.sin(theta))))

        return [gen_dist_x, gen_dist_y], distGen

    # def sampleGenTrain(self):
    #     test_random = np.random.rand(200)
    #     test_random_norm = test_random/(np.sum(test_random))
    #     test_random_norm = test_random_norm * 909090
    #     return test_random_norm

    def distGeneratorTrain(self, test_random_torch, bins=200):
        # a = self.sampleGenTrain()
        # test_random = np.random.rand(200)
        test_random = test_random_torch.cpu().numpy()
        test_random_norm = test_random/(np.sum(test_random))
        test_random_norm = test_random_norm * 909090
        distGen = test_random_norm
        distNorm = distGen*np.linspace(0, 0.22, 201)[1:]*2*np.pi     
        gen_dist_x = np.array([])
        gen_dist_y = np.array([])
        r = 0.22/bins
        for i in range(bins):
            theta = np.random.rand(int(distNorm[i]))*2*np.pi
            gen_dist_x = np.concatenate((gen_dist_x, np.multiply((np.random.rand(int(distNorm[i]))*r + i*r), np.cos(theta))))
            gen_dist_y = np.concatenate((gen_dist_y, np.multiply((np.random.rand(int(distNorm[i]))*r + i*r), np.sin(theta))))
        return [gen_dist_x, gen_dist_y], distGen

    def GeneratorInput(self, input):
        sampleList = []
        distList = []
        for i in range(input.shape[0]):
            # sample = self.distGeneratorTrain(input[i])
            sample = self.distGenerator(input[i])
            sampleList.append(sample[0])
            distList.append(sample[1])

        return sampleList, distList
    
    def generate_input(self, n, base=None):
        '''
        Generate random input data according to the base solution `base`.
        `new_random` is uniformly sampled with mean of `base` and range of 2.
        Not all components of `base` will be replaced by `new_random`. A random variable `p` is uniformly sampled in [0,1] which indicates the ratio
        of the components in `base` that will be mutated.
        Args:
            n : int
                number of total output samples
            base: torch.Tensor, size=(1, n_bar), dtype=float64
                base design, based on which new samples are generated
        Returns:
            inputs: torch.Tensor, size=(n, n_bar), dtype=float64
                new input samples, i.e., the areas of bars
        '''
        # generate_random = np.random.rand(40)
        new_random = base + torch.rand(n, 40, device=self.device)  # new input data, but only a portion will be adopted
        p = torch.linspace(0, 1, n, device=self.device)  # the ratio of components in a vector that adopts new input data (`new_random`)
        rand = torch.rand_like(new_random)
        mask = rand < p.unsqueeze(-1)  # indicates whether a component will change
        # print(mask)
        idx = torch.randperm(mask.shape[0])
        # print(idx)
        mask = mask[idx, :]  # randomly permute the mask
        inputs = base.repeat(n, 1)
        inputs = torch.where(mask, new_random, inputs)
        inputs.masked_fill_(inputs > 1., 1.)
        inputs.masked_fill_(inputs < 0., 0.)
        # inputs = torch.div(inputs.T, torch.sum(inputs, 1))
        # inputs = inputs.T
        # self.apply_discrete_thres(inputs)
        return inputs

    def coverDist(self, coordinates, Ero_dist_xy):
        test = depoML.depo(param = self.param, TS = self.TS, N = Ero_dist_xy[0].shape[0], sub_xy=coordinates)
        coverage = test.coverage(1, 125, Ero_dist_xy[0], Ero_dist_xy[1])
        return coverage
    
    def SCparallel(self, input, cores=60):
        num_jobs = cores  # Number of parallel jobs
        GenerateSamples = self.GeneratorInput(input)
        coverdist = Parallel(n_jobs=num_jobs, verbose=2)(
            delayed(self.coverDist)([x, 0], Ero_dist_xy) for Ero_dist_xy in GenerateSamples[0] for x in self.coordinates
        )
        outputs_all = torch.tensor(coverdist, device='cuda:0', dtype=torch.float32).reshape((self.num,3))
        outputs = torch.sum(outputs_all, dim=1)

        return outputs.reshape(self.num,1)
    
    def get_function(self, inverse=None):
        return self.SCparallel


if __name__ == '__main__':
    test = StepCoverage(10)
    print('run')
    a = test.SCparallel()
    print(a[0])