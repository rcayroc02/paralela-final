import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray
import time

class AStarPyCUDA:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        
        # CUDA kernel for A* search
        mod = SourceModule("""
        __global__ void astar_kernel(
            float *grid, 
            int *open_set, 
            int *closed_set, 
            float *g_scores, 
            float *f_scores, 
            int *came_from,
            int start_x, int start_y, 
            int goal_x, int goal_y, 
            int grid_size
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= grid_size || y >= grid_size) return;

            // Initialize scores
            int idx = y * grid_size + x;
            g_scores[idx] = INFINITY;
            f_scores[idx] = INFINITY;
            came_from[idx] = -1;
            open_set[idx] = 0;
            closed_set[idx] = 0;

            // Start node initialization
            if (x == start_x && y == start_y) {
                g_scores[idx] = 0.0f;
                f_scores[idx] = hypotf(x - goal_x, y - goal_y);
                open_set[idx] = 1;
            }

            // Basic heuristic: Manhattan distance
            __syncthreads();
        }

        __global__ void update_path_kernel(
    int *open_set, 
    int *closed_set, 
    float *g_scores, 
    float *f_scores, 
    int *came_from,
    int goal_x, int goal_y, 
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= grid_size || y >= grid_size) return;
    
    int current_idx = y * grid_size + x;
    
    // Only process nodes in the open set
    if (open_set[current_idx] == 0) return;
    
    // Neighbor offsets (8-directional movement)
    int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    // Mark current node as closed
    closed_set[current_idx] = 1;
    open_set[current_idx] = 0;
    
    // Check if goal reached
    if (x == goal_x && y == goal_y) {
        return;
    }
    
    // Explore neighbors
    for (int i = 0; i < 8; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        
        // Boundary and obstacle check
        if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) continue;
        
        int neighbor_idx = ny * grid_size + nx;
        
        // Skip if neighbor is an obstacle or closed
        if (closed_set[neighbor_idx] == 1) continue;
        
        // Movement cost (diagonal movement has higher cost)
        float movement_cost = (dx[i] == 0 || dy[i] == 0) ? 1.0f : 1.414f;
        
        // Calculate tentative g score
        float tentative_g_score = g_scores[current_idx] + movement_cost;
        
        // Heuristic (Manhattan distance)
        float heuristic = abs(nx - goal_x) + abs(ny - goal_y);
        
        // Only update if this path is better
        if (tentative_g_score < g_scores[neighbor_idx]) {
            // Update scores
            g_scores[neighbor_idx] = tentative_g_score;
            f_scores[neighbor_idx] = tentative_g_score + heuristic;
            
            // Update came from
            came_from[neighbor_idx] = current_idx;
            
            // Mark as open if not already
            open_set[neighbor_idx] = 1;
        }
    }
}
        """)

        self.astar_kernel = mod.get_function("astar_kernel")
        self.update_path_kernel = mod.get_function("update_path_kernel")

        # Preallocate device memory
        self.d_grid = cuda.mem_alloc(grid_size * grid_size * 4)
        self.d_open_set = cuda.mem_alloc(grid_size * grid_size * 4)
        self.d_closed_set = cuda.mem_alloc(grid_size * grid_size * 4)
        self.d_g_scores = cuda.mem_alloc(grid_size * grid_size * 4)
        self.d_f_scores = cuda.mem_alloc(grid_size * grid_size * 4)
        self.d_came_from = cuda.mem_alloc(grid_size * grid_size * 4)

    def find_path(self, start, goal, obstacles):
        # Start timing
        start_time = time.time()
        
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obs in obstacles:
            grid[obs[0], obs[1]] = 1.0

        # Copy data to device
        cuda.memcpy_htod(self.d_grid, grid)

        # Kernel configuration
        block_size = (16, 16, 1)
        grid_dim = (
            (self.grid_size + block_size[0] - 1) // block_size[0],
            (self.grid_size + block_size[1] - 1) // block_size[1],
            1
        )

        # Initial A* setup kernel
        self.astar_kernel(
            self.d_grid, 
            self.d_open_set, 
            self.d_closed_set, 
            self.d_g_scores, 
            self.d_f_scores, 
            self.d_came_from,
            np.int32(start[0]), np.int32(start[1]),
            np.int32(goal[0]), np.int32(goal[1]),
            np.int32(self.grid_size),
            block=block_size,
            grid=grid_dim
        )

        # Path update kernel (multiple iterations)
        iterations = self.grid_size * 2  # Arbitrary iteration limit
        for _ in range(iterations):
            self.update_path_kernel(
                self.d_open_set, 
                self.d_closed_set, 
                self.d_g_scores, 
                self.d_f_scores, 
                self.d_came_from,
                np.int32(goal[0]), np.int32(goal[1]),
                np.int32(self.grid_size),
                block=block_size,
                grid=grid_dim
            )

        # Get closed set to count expanded states
        closed_set = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        cuda.memcpy_dtoh(closed_set, self.d_closed_set)
        expanded_states = np.sum(closed_set)

        # Retrieve results
        came_from = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        cuda.memcpy_dtoh(came_from, self.d_came_from)

        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Calculate expansion rate (states per second)
        expansion_rate = expanded_states / execution_time if execution_time > 0 else 0

        # Get path
        path = self._reconstruct_path(came_from, start, goal)

        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Expanded States: {expanded_states}")
        print(f"Expansion Rate: {expansion_rate:.2f} states/second")

        return path

    def _reconstruct_path(self, came_from, start, goal):
        path = []
        current = goal
        while current != start:
            path.append(current)
            prev_x, prev_y = None, None
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if came_from[x, y] == current[0] * self.grid_size + current[1]:
                        prev_x, prev_y = x, y
                        break
                if prev_x is not None:
                    break
            
            if prev_x is None:
                break
            current = (prev_x, prev_y)
        
        path.append(start)
        return list(reversed(path))

    def __del__(self):
        # Manual memory cleanup
        self.d_grid.free()
        self.d_open_set.free()
        self.d_closed_set.free()
        self.d_g_scores.free()
        self.d_f_scores.free()
        self.d_came_from.free()

# Example usage
if __name__ == "__main__":
    grid_size = 100
    astar = AStarPyCUDA(grid_size)
    start = (0, 0)
    goal = (99, 99)
    obstacles = [(10, 10), (20, 20), (30, 30)]
    path = astar.find_path(start, goal, obstacles)