from constant import device
import torch
class Generate_Ising_Dataset:

    def __init__(self, L, proportion:float, beta:float) -> None:
        #self.lattice = lattice.to(device) 
        self.L = L 
        self.device = device 
        self.proportion = proportion
        

    def get_lattice_configuration(self)->torch.tensor:
      """This method initialize a spin tensor, where in each sites there is a plus 1 or -1, rnadomly chosen. Notice
        that 50% will be positive and 50% negative.

        Returns:
            torch.tensor: the spin tensor configutation
      """

      random_idx = torch.rand((self.L,self.L), device=self.device) # Specify device here
      config = torch.where(random_idx >= self.proportion, torch.tensor(1, device=self.device), torch.tensor(-1, device=self.device)).to(self.device)
      return config
    
    def get_neighbors(self,grid:torch.tensor,x:int,y:int)->torch.tensor:
      """This function computes the nearest neighbourhood sum in a 2D Ising lattice (i.e. S_i*S_j where i and j are nearest neighbors...)

        Args:
            x (int): x coordinate of a single spin site
            y (int): y coordinate of a single spin site
# 
        Returns:
            torch.tensor: Returns the interaction between nearest neighbors
      """

      neighbors = grid[(x + 1) % self.L, y] + grid[(x - 1) % self.L, y] + \
                  grid[x, (y + 1) % self.L] + grid[x, (y - 1) % self.L]
      return neighbors
    
    def metropolis(self,config:torch.tensor, beta:float, n_iter:int)->torch.tensor:
        """This function execute the metropolis algorithm on a spin lattice.

        Args:
            beta (float): Inverse of temperature

        Returns:
            torch.tensor: The lattice at equilibrium configuration
        """
        
        for _ in range(n_iter):

                i = torch.randint(0, self.L, (1,)).item()
                j = torch.randint(0, self.L, (1,)).item()
            
                neighbors = self.get_neighbors(grid = config, x = i,y = j) 
                    
                delta_E = 2 * config[i,j] * neighbors
                spin_flip_mask = (delta_E < 0) | (torch.rand((1,)) < torch.exp(-beta * delta_E))

        # Applicazione della maschera per flippare lo spin
                config[i, j] *= -1 if spin_flip_mask.item() else 1
                    
                
        return config
    

    def compute_energy(self)->torch.tensor:
      """ This function compute the average energy of the 2D ising model as \sum_<i,j>S_I*S_J, where 
      the nearest neighbor sum is computed through the get neighbor function

        Returns:
            torch.tensor: The full Hamiltonian of the Ising model
      """
      self.lattice = self.get_lattice_configuration()
      energy = []
      for _ in range(self.n_iter):
            i = torch.randint(0, self.L, (self.L,), device=self.device)
            j = torch.randint(0, self.L, (self.L,), device=self.device)


            neighbors = self.get_neighbors(i,j)
            delta_E = self.lattice[i, j] * neighbors
            energy.append(-2*delta_E.sum())
      return torch.tensor(energy)
    
    def simulate_ising(self, T: float)->torch.tensor:
        """Computes the metropolis algorithm for a specified number of times

        Args:
            T (float): External Bath Temperature
            n_iters (int, optional): Number of iteration. Defaults to 1000.

        Returns:
            torch.tensor: The spin configuration after N iteration
        """

        self.T = T
        self.beta = 1 / self.T
        return self.metropolis(self.beta)
    
    def generate_dataset(self, T_c: float, n_iters: int, batch_size=50, num_samples: int = 50, alpha: int = 5)->torch.tensor:
        """This function generate the Ising dataset as a 4D tensor and its corresponding labels 
        To be specific, the tensor is of the following form [T,N_b,H,W], where T are the temperatures, N_b are the number of lattices per
        temperature and H and W are the lattice dimensions.
        Labels represent if a phase is ordered or not so we can interpreting the problem of classifying the phase transition as a binary classification problem.

        Args:
            T_c (float): Critical temperature of the 2D ising Model
            n_iters (int): number of iteration
            batch_size (int, optional): Number of Temperatures. Defaults to 50.
            num_samples (int, optional): Number of lattices per single temperature. Defaults to 50.
            alpha (int, optional): Dimensionless parameter to control the temperature range. Defaults to 5.

        Returns:
            torch.tensor: Returns labels and data
        """
        self.T_c = T_c
        self.batch_size = batch_size
        self.num_samples = num_samples
        epsilon = 1e-5


        dataset_shape = (self.batch_size, self.L, self.L)
        dataset = torch.zeros(dataset_shape, device=self.device)

        t_min = epsilon
        t_max = alpha * self.T_c

        temperatures = torch.linspace(t_min, t_max, self.batch_size, device=self.device)
        label = torch.zeros(temperatures.shape[0])
        for t_ix, t in enumerate(temperatures):
            random_grid = self.get_lattice_configuration()
            for _ in range(self.num_samples):
                data = self.metropolis(config = random_grid, beta = 1/t, n_iter = 10000)
            dataset[t_ix, :, :] = data
            label[t_ix] = 1 if t > self.T_c else 0
                
        return label, dataset
